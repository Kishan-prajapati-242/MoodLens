import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import os

class MultiTaskEventClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Load transformer
        self.transformer = AutoModel.from_pretrained(config['model_name'])
        
        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        
        # Attention
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.transformer.config.hidden_size,
            num_heads=12,
            dropout=0.3,
            batch_first=True
        )
        
        # Shared layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific classifiers
        self.event_type_classifier = self._make_classifier(768, num_classes['event_type'])
        self.event_group_classifier = self._make_classifier(768, num_classes['event_group'])
        self.emotion_classifier = self._make_classifier(768, num_classes['emotion'])
        self.tense_classifier = self._make_classifier(768, num_classes['tense'])
        self.sarcasm_classifier = self._make_classifier(768, 2)
        
        # Regression heads
        self.sentiment_regressor = self._make_regressor(768)
        self.certainty_regressor = self._make_regressor(768)
    
    def _make_classifier(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def _make_regressor(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        attended, _ = self.text_attention(
            hidden_state, hidden_state, hidden_state,
            key_padding_mask=~attention_mask.bool()
        )
        text_features = (attended * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        shared_features = self.shared_layer(text_features)
        
        return {
            'event_type': self.event_type_classifier(shared_features),
            'event_group': self.event_group_classifier(shared_features),
            'emotion': self.emotion_classifier(shared_features),
            'tense': self.tense_classifier(shared_features),
            'sarcasm': self.sarcasm_classifier(shared_features),
            'sentiment_valence': self.sentiment_regressor(shared_features).squeeze(-1),
            'certainty': self.certainty_regressor(shared_features).squeeze(-1)
        }

class ModelHandler:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(os.path.join(models_dir, 'inference_config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(models_dir, 'tokenizer'))
        
        # Load label encoders
        self.label_encoders = {}
        encoder_files = {
            'event_type': 'le_event_type.pkl',
            'event_group': 'le_event_group.pkl',
            'emotion': 'le_emotion.pkl',
            'tense': 'le_tense.pkl'
        }
        
        for task, filename in encoder_files.items():
            filepath = os.path.join(models_dir, 'label_encoders', filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.label_encoders[task] = pickle.load(f)
        
        # Load model
        self.model = MultiTaskEventClassifier(self.config, self.config['num_classes'])
        checkpoint = torch.load(
            os.path.join(models_dir, 'best_model.pth'),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text, threshold=0.7):
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions for each task
            event_probs = torch.softmax(outputs['event_type'], dim=1)
            event_pred = torch.argmax(outputs['event_type'], dim=1)
            
            # Get top 3 event predictions
            top_probs, top_indices = torch.topk(event_probs[0], 3)
            top_events = []
            for idx, prob in zip(top_indices, top_probs):
                event_name = self.label_encoders['event_type'].inverse_transform([idx.item()])[0]
                top_events.append({
                    'event': event_name,
                    'confidence': prob.item()
                })
            
            # Get other predictions
            results = {
                'primary_event': top_events[0],
                'top_events': top_events,
                'event_group': self._get_classification_result(outputs['event_group'], 'event_group'),
                'emotion': self._get_classification_result(outputs['emotion'], 'emotion'),
                'tense': self._get_classification_result(outputs['tense'], 'tense'),
                'sarcasm': self._get_sarcasm_result(outputs['sarcasm']),
                'sentiment': outputs['sentiment_valence'][0].item(),
                'certainty': outputs['certainty'][0].item()
            }
            
            # Determine if multiple events
            if top_events[0]['confidence'] < threshold:
                results['interpretation'] = f"Could be {top_events[0]['event']} or {top_events[1]['event']}"
                results['multiple_events'] = True
            else:
                results['interpretation'] = f"Clear event: {top_events[0]['event']}"
                results['multiple_events'] = False
            
            return results
    
    def _get_classification_result(self, output, task):
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        prediction = self.label_encoders[task].inverse_transform(pred.cpu().numpy())[0]
        confidence = probs[0, pred[0]].item()
        return {
            'prediction': prediction,
            'confidence': confidence
        }
    
    def _get_sarcasm_result(self, output):
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        return {
            'prediction': 'TRUE' if pred.item() == 1 else 'FALSE',
            'confidence': probs[0, pred[0]].item()
        }