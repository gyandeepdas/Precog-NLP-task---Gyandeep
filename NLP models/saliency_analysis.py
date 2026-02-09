# Saliency Mapping for DistilBERT False Positives
# This script analyzes which words most strongly influenced DistilBERT to classify human text as AI

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Install Captum if needed
try:
    from captum.attr import LayerIntegratedGradients
    print("✓ Captum already installed")
except:
    print("Installing Captum...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'captum'])
    from captum.attr import LayerIntegratedGradients
    print("✓ Captum installed successfully")

# Load model and tokenizer
print("Loading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'Distilbert model'
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()

# Create a wrapper that returns only logits (Captum requirement)
def forward_func(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits

# Example text (replace with your false positive)
text = """as great a contrast to it as the front of a picture does to the back. It was one of the main arteries which conveyed the traffic of the City to the north and west. The roadway was blocked with the immense stream of commerce flowing in a double tide inward and outward, while the footpaths were black with the hurrying swarm of pedestrians. It was difficult to realise as we looked at the line of fine shops and stately business premises that they really abutted on the other side upon the faded and stagnant square which we had just quitted. “Let me see,” said Holmes, standing at the corner and glancing along the line, “I should like just to remember the order of the houses here. It is a hobby of mine to have an exact knowledge of London. There is Mortimer’s, the tobacconist, the little newspaper shop, the Coburg branch of the City and Suburban Bank, the Vegetarian Restaurant, and McFarlane’s carriage-building depot. That carries us right on to the other block. And now, Doctor, we’ve done our work, so it’s time we had some play. A sandwich and a cup of coffee, and then off to violin-land,
"""

print("="*80)
print("SALIENCY MAPPING: Word Attribution Analysis")
print("="*80)

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_class].item()

print(f"\nPredicted: {'AI' if pred_class == 1 else 'Human'} (Confidence: {confidence:.4f})")
print(f"\nAnalyzing text: {text[:200]}...\n")

# Set up Layer Integrated Gradients (use forward_func wrapper)
lig = LayerIntegratedGradients(forward_func, model.distilbert.embeddings)

# Baseline is [CLS] + [PAD] tokens
baseline_ids = torch.cat([
    input_ids[:, :1],  # Keep [CLS]
    torch.zeros_like(input_ids[:, 1:])  # PAD for rest
], dim=1).long()

# Calculate attributions
print("Computing attributions...")
attributions = lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,
    additional_forward_args=(attention_mask,),
    target=1,  # Target AI class
    n_steps=50
)

# Sum attributions across embedding dimension
attributions_sum = attributions.sum(dim=-1).squeeze(0)
attributions_sum = attributions_sum / torch.norm(attributions_sum)
attributions_norm = attributions_sum.detach().cpu().numpy()

# Get tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Aggregate subword tokens into complete words
word_attributions = {}
current_word = ""
current_attr = 0.0

for i, (token, attr) in enumerate(zip(tokens, attributions_norm)):
    if token in ['[CLS]', '[SEP]', '[PAD]']:
        continue
    
    # Handle subword tokens (##)
    if token.startswith('##'):
        current_word += token[2:]
        current_attr += attr
    else:
        # Save previous word if it exists and is not pure punctuation
        if current_word and not all(c in '.,!?;:\'"()[]{}=-' for c in current_word):
            if current_word in word_attributions:
                word_attributions[current_word] += current_attr
            else:
                word_attributions[current_word] = current_attr
        
        # Start new word
        current_word = token
        current_attr = attr

# Don't forget the last word
if current_word and not all(c in '.,!?;:\'"()[]{}=-' for c in current_word):
    if current_word in word_attributions:
        word_attributions[current_word] += current_attr
    else:
        word_attributions[current_word] = current_attr

# Sort by attribution magnitude (most positive = strongest push toward AI)
sorted_words = sorted(word_attributions.items(), key=lambda x: x[1], reverse=True)

# Display top contributing words
print("\nTop 20 Words Pushing Toward 'AI' Classification:")
print("-" * 80)
print(f"{'Rank':<6} {'Word':<25} {'Attribution':<15} {'Strength'}")
print("-" * 80)

count = 0
for word, attr in sorted_words:
    if attr > 0 and count < 20:  # Only show positive contributions
        bar_length = int(attr * 100)
        bar = '█' * min(bar_length, 40)
        print(f"{count+1:<6} {word:<25} {attr:+.4f}          {bar}")
        count += 1

# Also show top words pushing toward 'Human' (negative attribution)
print(f"\n{'='*80}")
print("Top 10 Words Pushing Toward 'Human' Classification:")
print("-" * 80)
print(f"{'Rank':<6} {'Word':<25} {'Attribution':<15} {'Strength'}")
print("-" * 80)

sorted_negative = sorted(word_attributions.items(), key=lambda x: x[1])
count = 0
for word, attr in sorted_negative:
    if attr < 0 and count < 10:
        bar_length = int(abs(attr) * 100)
        bar = '█' * min(bar_length, 40)
        print(f"{count+1:<6} {word:<25} {attr:+.4f}          {bar}")
        count += 1

# Create word importance text visualization
print(f"\n{'='*80}")
print("Text with Word Importance Markers:")
print("** = Strong push toward AI, * = Medium push toward AI")
print(f"{'='*80}\n")

output_text = []
for token, attr in zip(tokens, attributions_norm):
    if token in ['[CLS]', '[SEP]', '[PAD]']:
        continue
    
    # Clean up token (remove ##)
    clean_token = token.replace('##', '')
    
    # Add marker based on attribution strength
    if attr > 0.1:
        output_text.append(f"**{clean_token}**")  # Strong positive
    elif attr > 0.05:
        output_text.append(f"*{clean_token}*")   # Medium positive
    else:
        output_text.append(clean_token)

# Print formatted text
formatted_text = ' '.join(output_text)
print(formatted_text)
print(f"\n{'='*80}\n")
