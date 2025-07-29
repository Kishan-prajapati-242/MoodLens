from transformers import AutoTokenizer
import os

# Create tokenizer directory if it doesn't exist
os.makedirs('models/tokenizer_new', exist_ok=True)

# Download the tokenizer
print("Downloading tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

# Save it locally
print("Saving tokenizer...")
tokenizer.save_pretrained('models/tokenizer_new')

print("Done! Tokenizer saved to models/tokenizer_new")
print("\nNow rename the folders:")
print("1. mv models/tokenizer models/tokenizer_old")
print("2. mv models/tokenizer_new models/tokenizer")