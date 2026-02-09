"""
Genetic Algorithm to Mutate AI-generated Text
Uses WordNet synonyms and DistilBERT as fitness function
Goal: Make text appear less AI-generated
"""

import torch
import random
import spacy
from nltk.corpus import wordnet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download WordNet if needed
try:
    import nltk
    nltk.data.find('corpora/wordnet.zip')
except:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Load spaCy for better tokenization
print("Loading spaCy...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("✓ spaCy loaded")
except:
    print("Error: spaCy model not found. Install with:")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")
    exit(1)

# Load DistilBERT model
print("\nLoading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'Distilbert model'
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()

def get_ai_probability(text):
    """Calculate probability that text is AI-generated"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        ai_prob = probs[0][1].item()  # Class 1 = AI
    return ai_prob

def fitness_function(text):
    """Higher fitness = less AI-like (more human)"""
    ai_prob = get_ai_probability(text)
    return 1.0 - ai_prob  # Invert: we want LOW AI probability

def get_wordnet_pos(spacy_pos):
    """Convert spaCy POS tag to WordNet POS tag"""
    if spacy_pos in ['ADJ']:
        return wordnet.ADJ
    elif spacy_pos in ['VERB']:
        return wordnet.VERB
    elif spacy_pos in ['NOUN', 'PROPN']:
        return wordnet.NOUN
    elif spacy_pos in ['ADV']:
        return wordnet.ADV
    else:
        return None

def get_synonyms(word, pos_tag=None):
    """Get synonyms for a word from WordNet (single words only)"""
    synonyms = set()
    
    for syn in wordnet.synsets(word.lower(), pos=pos_tag):
        for lemma in syn.lemmas():
            synonym = lemma.name()
            # Only single words, no underscores or hyphens
            if '_' not in synonym and '-' not in synonym and synonym.lower() != word.lower():
                synonyms.add(synonym)
    
    return list(synonyms)

def mutate_text(text, mutation_rate=0.25):
    """Mutate text by replacing content words with synonyms"""
    doc = nlp(text)
    mutated_tokens = []
    mutations_made = 0
    
    for token in doc:
        # Only mutate content words (nouns, verbs, adjectives, adverbs)
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(token.text) > 3:
            if random.random() < mutation_rate:
                wn_pos = get_wordnet_pos(token.pos_)
                synonyms = get_synonyms(token.text, wn_pos)
                
                if synonyms:
                    synonym = random.choice(synonyms)
                    # Preserve capitalization
                    if token.text[0].isupper():
                        synonym = synonym.capitalize()
                    mutated_tokens.append(synonym)
                    mutations_made += 1
                else:
                    mutated_tokens.append(token.text)
            else:
                mutated_tokens.append(token.text)
        else:
            mutated_tokens.append(token.text)
    
    # Reconstruct text with proper spacing
    result = ""
    for i, token in enumerate(doc):
        result += mutated_tokens[i]
        # Add whitespace if the original had it
        if i < len(doc) - 1 and token.whitespace_:
            result += token.whitespace_
    
    return result, mutations_made

def genetic_algorithm(original_text, population_size=10, generations=25, mutation_rate=0.25):
    """Run genetic algorithm to evolve text to be less AI-like"""
    
    print(f"\n{'='*80}")
    print("GENETIC ALGORITHM: Mutating AI Text to Appear More Human")
    print(f"{'='*80}")
    print(f"\nOriginal Text:")
    print(f"  {original_text}")
    
    original_ai_prob = get_ai_probability(original_text)
    print(f"\n  AI Probability: {original_ai_prob:.4f} (Fitness: {1-original_ai_prob:.4f})")
    
    # Initialize population with the original text plus mutations
    population = [original_text]
    for _ in range(population_size - 1):
        mutated, _ = mutate_text(original_text, mutation_rate=0.4)
        population.append(mutated)
    
    best_overall = original_text
    best_overall_fitness = fitness_function(original_text)
    
    # Evolution
    for gen in range(generations):
        print(f"\n{'─'*80}")
        print(f"Generation {gen + 1}/{generations}")
        print('─'*80)
        
        # Evaluate fitness for all candidates
        fitness_scores = [(text, fitness_function(text)) for text in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Display best of this generation
        best_text, best_fitness = fitness_scores[0]
        ai_prob = 1.0 - best_fitness
        print(f"Best: AI Prob = {ai_prob:.4f} | Fitness = {best_fitness:.4f}")
        print(f"Text: {best_text[:150]}...")
        
        # Track best overall
        if best_fitness > best_overall_fitness:
            best_overall = best_text
            best_overall_fitness = best_fitness
            print(f"✓ NEW BEST! Improvement: {((original_ai_prob - ai_prob) / original_ai_prob * 100):.2f}%")
        
        # Elitism: Keep top 3 candidates
        survivors = [text for text, _ in fitness_scores[:3]]
        
        # Create next generation
        next_population = survivors.copy()
        
        # Breed and mutate to fill population
        while len(next_population) < population_size:
            parent = random.choice(survivors)
            child, _ = mutate_text(parent, mutation_rate)
            next_population.append(child)
        
        population = next_population
    
    # Final results
    print(f"\n{'='*80}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nOriginal Text:")
    print(f"  {original_text}")
    print(f"  AI Probability: {original_ai_prob:.4f}")
    
    print(f"\nBest Evolved Text:")
    print(f"  {best_overall}")
    
    final_ai_prob = get_ai_probability(best_overall)
    print(f"  AI Probability: {final_ai_prob:.4f}")
    
    improvement = ((original_ai_prob - final_ai_prob) / original_ai_prob) * 100
    print(f"\n  Improvement: {improvement:.2f}% reduction in AI probability")
    
    return best_overall, final_ai_prob

# Example AI-generated sentence
ai_sentence = """The implementation of advanced machine learning algorithms enables the system to effectively process and analyze large-scale datasets, thereby facilitating enhanced decision-making capabilities and operational efficiency."""

# Alternative example (shorter)
# ai_sentence = """The utilization of innovative technologies provides significant advantages for modern businesses seeking optimization and growth."""

# Run the genetic algorithm
if __name__ == "__main__":
    best_text, final_prob = genetic_algorithm(
        original_text=ai_sentence,
        population_size=10,
        generations=15,
        mutation_rate=0.3
    )
    
    print(f"\n{'='*80}")
    print("Want to test with your own AI text?")
    print("Edit the 'ai_sentence' variable at the bottom of this script!")
    print(f"{'='*80}\n")
