#Proof of Concept for NLSHBlock (Neural Locality Sensitive Hashing for Blocking)

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
import random


def serialize(item):
    serialized_text = ''.join([f'[COL]{attr}[VAL]{value}' for attr, value in item.items()])
    return serialized_text


def preprocess(texts):
    # Mocked Embeddings (Improved Simulation)
    np.random.seed(42)
    embeddings = np.random.rand(len(texts), 768)

    # Simulate similarity by making embeddings of similar products closer
    for i, text in enumerate(texts):
        if "calculator" in text:
            embeddings[i] *= 0.9  # Reduce distance for calculator-related products
        elif "workshop" in text:
            embeddings[i] *= 0.8  # Reduce distance for workshop-related products
        elif "spanish" in text:
            embeddings[i] *= 0.7  # Reduce distance for spanish-related products
    return embeddings


def augment_text(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def nlsh_loss(p, q, r, R=0.01, c=3.0):
    loss = max(R, np.abs(p - q).sum()) - min(c * R, np.abs(p - r).sum())
    return loss


def generate_candidates(embeddings, items, k=3):  # Adjusted k to work with larger datasets
    knn = NearestNeighbors(n_neighbors=min(k, len(items)), metric='euclidean')
    knn.fit(embeddings)

    candidate_pairs = {}
    for i, _ in enumerate(items):
        distances, indices = knn.kneighbors(embeddings[i].reshape(1, -1))
        candidate_pairs[i] = indices.flatten().tolist()

    return candidate_pairs


def evaluate(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return precision, recall, f1


# Improved Synthetic Dataset (Larger and More Diverse)
sample_items = [
    {'Product Name': 'sharp printing calculator', 'Manufacturer': 'sharp', 'Price': '45.63'},
    {'Product Name': 'instant immersion spanish deluxe 2.0', 'Manufacturer': 'topics entertainment', 'Price': '39.99'},
    {'Product Name': 'adventure workshop 4th-6th grade 7th edition', 'Manufacturer': 'encore software', 'Price': '29.99'},
    {'Product Name': 'basic scientific calculator', 'Manufacturer': 'casio', 'Price': '19.99'},
    {'Product Name': 'spanish language course', 'Manufacturer': 'learnco', 'Price': '49.99'},
    {'Product Name': 'english workshop edition', 'Manufacturer': 'studyco', 'Price': '39.99'},
]

texts = [serialize(item) for item in sample_items]
embeddings = preprocess(texts)

# Generate candidates
candidates = generate_candidates(embeddings, sample_items)
print("Candidate Pairs:", candidates)

# Evaluation (Simulated true labels for testing)
true_labels = [1, 0, 1, 1, 0, 1]  # Example true labels (match=1, no match=0)
predicted_labels = [1 if i in candidates[0] else 0 for i in range(len(sample_items))]
precision, recall, f1 = evaluate(true_labels, predicted_labels)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
