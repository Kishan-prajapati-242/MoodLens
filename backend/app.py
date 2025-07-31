import sys
# Simple numpy fix
import numpy
numpy._core = sys.modules['numpy.core']

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import os
import pickle
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI(title="Event Classifier API - Working Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom unpickler for label encoders
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        return super().find_class(module, name)

def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return CustomUnpickler(f).load()

# YOUR EXACT MODEL ARCHITECTURE FROM COLAB
class MultiTaskEventClassifier(nn.Module):
    def __init__(self, config, label_encoders):
        super().__init__()
        self.config = config
        self.label_encoders = label_encoders

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)

        # Freeze embeddings
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False

        # Attention for text features
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.transformer.config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )

        # Shared feature extractor
        self.shared_layer = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )

        # Task-specific classifiers
        self.event_type_classifier = self._make_classifier(
            config.hidden_dim, label_encoders.num_classes['event_type'])
        self.event_group_classifier = self._make_classifier(
            config.hidden_dim, label_encoders.num_classes['event_group'])
        self.emotion_classifier = self._make_classifier(
            config.hidden_dim, label_encoders.num_classes['emotion'])
        self.tense_classifier = self._make_classifier(
            config.hidden_dim, label_encoders.num_classes['tense'])
        self.sarcasm_classifier = self._make_classifier(
            config.hidden_dim, 2)

        # Regression heads
        self.sentiment_regressor = self._make_regressor(config.hidden_dim)
        self.certainty_regressor = self._make_regressor(config.hidden_dim)

    def _make_classifier(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 2, num_classes)
        )

    def _make_regressor(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Apply attention
        hidden_state = outputs.last_hidden_state
        attended, _ = self.text_attention(
            hidden_state, hidden_state, hidden_state,
            key_padding_mask=~attention_mask.bool()
        )

        # Pool
        text_features = (attended * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # Get shared features
        shared_features = self.shared_layer(text_features)

        # Get predictions from each head
        outputs = {
            'event_type': self.event_type_classifier(shared_features),
            'event_group': self.event_group_classifier(shared_features),
            'emotion': self.emotion_classifier(shared_features),
            'tense': self.tense_classifier(shared_features),
            'sarcasm': self.sarcasm_classifier(shared_features),
            'sentiment_valence': self.sentiment_regressor(shared_features).squeeze(-1),
            'certainty': self.certainty_regressor(shared_features).squeeze(-1)
        }

        return outputs

# Dummy classes needed for model creation
class MultiTaskLabelEncoders:
    def __init__(self):
        self.encoders = {}
        self.num_classes = {}

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Initialize
print("="*60)
print("Loading Event Classifier - Final Working Version")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config
with open('models/inference_config.json', 'r') as f:
    config_dict = json.load(f)

# Create config object with proper attributes
config = Config(config_dict)
config.model_name = config_dict.get('model_name', 'microsoft/deberta-v3-base')
config.max_length = config_dict.get('max_length', 256)
config.hidden_dim = 768
config.num_attention_heads = 12
config.dropout_rate = 0.3

print(f"✓ Config loaded: {config.model_name}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
print("✓ Tokenizer loaded")

# Load label encoders
print("\nLoading label encoders...")
label_encoders = {}
encoder_files = {
    'event_type': 'le_event_type.pkl',
    'event_group': 'le_event_group.pkl', 
    'emotion': 'le_emotion.pkl',
    'tense': 'le_tense.pkl'
}

for task, filename in encoder_files.items():
    filepath = os.path.join('models/label_encoders', filename)
    if os.path.exists(filepath):
        try:
            label_encoders[task] = load_pickle_file(filepath)
            print(f"✓ {task}: {len(label_encoders[task].classes_)} classes")
        except Exception as e:
            print(f"✗ Error loading {task}: {e}")

# Create dummy label_encoders for model initialization (EXACTLY like Colab)
dummy_label_encoders = MultiTaskLabelEncoders()
dummy_label_encoders.num_classes = config_dict['num_classes']

# Create model EXACTLY as in Colab
print("\nCreating model architecture...")
model = MultiTaskEventClassifier(config, dummy_label_encoders)

# Load YOUR TRAINED WEIGHTS
print("\nLoading model weights...")
try:
    # Load the clean weights from Colab
    state_dict = torch.load('models/model_weights_clean.pth', map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Model weights loaded successfully!")
    print("✓ YOUR TRAINED MODEL IS READY!")
    
    # DEBUG: Verify weights are actually loaded
    sample_weight = model.event_type_classifier[0].weight
    print(f"\nDEBUG INFO:")
    print(f"  Event classifier weight shape: {sample_weight.shape}")
    print(f"  Weight mean: {sample_weight.mean().item():.6f}")
    print(f"  Weight std: {sample_weight.std().item():.6f}")
    print(f"  First 5 weights: {sample_weight[0, :5].tolist()}")
    
except Exception as e:
    print(f"ERROR loading weights: {e}")
    print("Make sure model_weights_clean.pth is in the models/ folder!")
    raise  # Stop here - don't use random weights!

model.to(device)
model.eval()

print(f"\n✓ Model ready on {device}")
print(f"✓ Loaded {len(label_encoders)} label encoders")
print("="*60)

# API Models
class PredictionRequest(BaseModel):
    text: str
    threshold: float = 0.7

class PredictionResponse(BaseModel):
    primary_event: dict
    top_events: list
    event_group: dict
    emotion: dict
    tense: dict
    sarcasm: dict
    sentiment: float
    certainty: float
    interpretation: str
    multiple_events: bool

# Prediction function
def predict_text(text, threshold=0.7):
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Process event type predictions
        event_probs = torch.softmax(outputs['event_type'], dim=1)
        top_k = min(3, event_probs.shape[1])
        top_probs, top_indices = torch.topk(event_probs[0], top_k)
        
        # Get event names
        top_events = []
        for idx, prob in zip(top_indices, top_probs):
            if 'event_type' in label_encoders:
                event_name = label_encoders['event_type'].inverse_transform([idx.item()])[0]
            else:
                event_name = f"event_{idx.item()}"
            
            top_events.append({
                'event': event_name,
                'confidence': prob.item()
            })
        
        # Helper function for classifications
        def get_classification_result(output, task):
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            
            if task in label_encoders:
                prediction = label_encoders[task].inverse_transform(pred.cpu().numpy())[0]
            else:
                prediction = f"{task}_{pred.item()}"
            
            confidence = probs[0, pred[0]].item()
            return {
                'prediction': prediction,
                'confidence': confidence
            }
        
        # Get all predictions
        results = {
            'primary_event': top_events[0],
            'top_events': top_events,
            'event_group': get_classification_result(outputs['event_group'], 'event_group'),
            'emotion': get_classification_result(outputs['emotion'], 'emotion'),
            'tense': get_classification_result(outputs['tense'], 'tense'),
            'sarcasm': {
                'prediction': 'TRUE' if outputs['sarcasm'].argmax().item() == 1 else 'FALSE',
                'confidence': torch.softmax(outputs['sarcasm'], dim=1).max().item()
            },
            'sentiment': float(outputs['sentiment_valence'][0].item()),
            'certainty': float(outputs['certainty'][0].item())
        }
        
        # Interpretation
        if top_events[0]['confidence'] >= threshold:
            results['interpretation'] = f"Clear event: {top_events[0]['event']}"
            results['multiple_events'] = False
        else:
            if len(top_events) > 1:
                results['interpretation'] = f"Could be {top_events[0]['event']} or {top_events[1]['event']}"
            else:
                results['interpretation'] = f"Uncertain: {top_events[0]['event']}"
            results['multiple_events'] = True
        
        return results

# Endpoints
# Serve frontend files - ONLY mount on /static path, not root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')


# Keep the original API info endpoint  
@app.get("/api")
def api_info():
    return {
        "message": "Event Classifier API is running!",
        "model": config.model_name,
        "device": str(device),
        "encoders_loaded": list(label_encoders.keys())
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_text(request.text, request.threshold)
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nStarting server on 0.0.0.0:{port}")
    print("Your Event Classifier is ready! 🚀")
    uvicorn.run(app, host="0.0.0.0", port=port)