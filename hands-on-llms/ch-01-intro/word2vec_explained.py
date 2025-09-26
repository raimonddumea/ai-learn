"""
Word2Vec Implementation and Explanation
Based on Chapter 1 of "Hands-On Large Language Models" by Jay Alammar

This script demonstrates the core concepts of Word2Vec:
1. EMBEDDINGS: How words are represented as vectors
2. NEURAL NETWORKS: The architecture that learns these representations
3. PARAMETERS: The weights that are learned during training
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random


print("=" * 80)
print("WORD2VEC: Learning Word Embeddings")
print("=" * 80)

# ============================================================================
# STEP 1: TEXT PREPROCESSING AND VOCABULARY
# ============================================================================

print("\n1. PREPARING OUR TEXT DATA")
print("-" * 40)

corpus = [
    "the cat sits on the mat",
    "the dog plays in the park",
    "cat and dog are pets",
    "the mat is on the floor",
    "pets play in the park"
]

print("Sample corpus:")
for sentence in corpus:
    print(f"  - {sentence}")

def preprocess_corpus(sentences: List[str]) -> Tuple[List[List[str]], Dict[str, int], Dict[int, str]]:
    """Convert text to tokens and create vocabulary mappings."""
    tokenized = []
    vocab = set()

    for sentence in sentences:
        tokens = sentence.lower().split()
        tokenized.append(tokens)
        vocab.update(tokens)

    idx_to_word = {idx: word for idx, word in enumerate(sorted(vocab))}
    word_to_idx = {word: idx for idx, word in idx_to_word.items()}

    return tokenized, word_to_idx, idx_to_word

tokenized_corpus, word_to_idx, idx_to_word = preprocess_corpus(corpus)
vocab_size = len(word_to_idx)

print(f"\nVocabulary size: {vocab_size}")
print(f"Word to index mapping (first 5): {dict(list(word_to_idx.items())[:5])}")


# ============================================================================
# STEP 2: UNDERSTANDING EMBEDDINGS
# ============================================================================

print("\n2. WHAT ARE EMBEDDINGS?")
print("-" * 40)
print("""
Embeddings are dense vector representations of words.
Instead of one-hot encoding (sparse, high-dimensional):
  'cat' = [0, 0, 1, 0, 0, 0, ...]  (only one 1 in a huge vector)

We use embeddings (dense, low-dimensional):
  'cat' = [0.2, -0.4, 0.7, 0.1]  (small vector with meaningful values)
""")

embedding_dim = 4  # Small dimension for visualization

print(f"Each word will be represented as a {embedding_dim}-dimensional vector")


# ============================================================================
# STEP 3: THE NEURAL NETWORK ARCHITECTURE
# ============================================================================

print("\n3. THE WORD2VEC NEURAL NETWORK")
print("-" * 40)
print(f"""
Word2Vec uses a simple neural network with:
  - Input layer: One-hot encoded word (size: {vocab_size})
  - Hidden layer: Word embedding (size: {embedding_dim})
  - Output layer: Probability over all words (size: {vocab_size})

This is the Skip-Gram architecture:
  Given a word -> Predict its context words
""")


# ============================================================================
# STEP 4: PARAMETERS (WEIGHTS) OF THE NETWORK
# ============================================================================

print("\n4. NETWORK PARAMETERS")
print("-" * 40)

class SimpleWord2Vec:
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Word2Vec model with random weights.

        PARAMETERS:
        - W1: Input to hidden weights (vocab_size x embedding_dim)
              These become our word embeddings!
        - W2: Hidden to output weights (embedding_dim x vocab_size)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialize parameters with small random values
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.1

        print(f"Initialized parameters:")
        print(f"  W1 shape: {self.W1.shape} - These are our word embeddings!")
        print(f"  W2 shape: {self.W2.shape} - Output layer weights")
        print(f"  Total parameters: {self.W1.size + self.W2.size}")

    def forward_pass(self, word_idx: int) -> np.ndarray:
        """
        Forward pass through the network.

        Steps:
        1. Create one-hot input vector
        2. Multiply by W1 to get hidden layer (embedding)
        3. Multiply by W2 to get output
        4. Apply softmax to get probabilities
        """
        # Step 1: One-hot encode input word
        x = np.zeros(self.vocab_size)
        x[word_idx] = 1

        # Step 2: Hidden layer = input @ W1
        # This extracts the embedding for the word!
        h = x @ self.W1  # Shape: (embedding_dim,)

        # Step 3: Output layer = hidden @ W2
        u = h @ self.W2  # Shape: (vocab_size,)

        # Step 4: Softmax to get probabilities
        y_pred = self.softmax(u)

        return y_pred, h

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Convert scores to probabilities."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

    def get_embedding(self, word: str, word_to_idx: Dict[str, int]) -> np.ndarray:
        """Get the embedding vector for a word."""
        word_idx = word_to_idx[word]
        return self.W1[word_idx]


# ============================================================================
# STEP 5: CREATING TRAINING DATA (SKIP-GRAM)
# ============================================================================

print("\n5. CREATING TRAINING PAIRS")
print("-" * 40)

def create_skip_gram_pairs(tokenized_corpus: List[List[str]],
                          word_to_idx: Dict[str, int],
                          window_size: int = 2) -> List[Tuple[int, int]]:
    """
    Create (center_word, context_word) pairs for training.

    For each word, look at surrounding words within window_size.
    """
    pairs = []

    for sentence in tokenized_corpus:
        for i, center_word in enumerate(sentence):
            center_idx = word_to_idx[center_word]

            # Look at words within the window
            for j in range(max(0, i - window_size),
                          min(len(sentence), i + window_size + 1)):
                if i != j:  # Skip the center word itself
                    context_word = sentence[j]
                    context_idx = word_to_idx[context_word]
                    pairs.append((center_idx, context_idx))

    return pairs

training_pairs = create_skip_gram_pairs(tokenized_corpus, word_to_idx)
print(f"Created {len(training_pairs)} training pairs")
print("\nSample training pairs (center_word -> context_word):")
for i in range(min(5, len(training_pairs))):
    center_idx, context_idx = training_pairs[i]
    print(f"  {idx_to_word[center_idx]:8} -> {idx_to_word[context_idx]}")


# ============================================================================
# STEP 6: TRAINING THE MODEL (SIMPLIFIED)
# ============================================================================

print("\n6. TRAINING PROCESS")
print("-" * 40)

model = SimpleWord2Vec(vocab_size, embedding_dim)

def train_step(model: SimpleWord2Vec,
               center_idx: int,
               context_idx: int,
               learning_rate: float = 0.01) -> float:
    """
    One training step using gradient descent.

    This is simplified - real Word2Vec uses more efficient techniques.
    """
    # Forward pass
    y_pred, h = model.forward_pass(center_idx)

    # Create target (one-hot for context word)
    y_true = np.zeros(model.vocab_size)
    y_true[context_idx] = 1

    # Calculate error
    error = y_pred - y_true
    loss = -np.log(y_pred[context_idx] + 1e-10)  # Cross-entropy loss

    # Backpropagation (simplified)
    # Update W2
    dW2 = np.outer(h, error)
    model.W2 -= learning_rate * dW2

    # Update W1 (only the row for the input word)
    dW1 = error @ model.W2.T
    model.W1[center_idx] -= learning_rate * dW1

    return loss

print("Training for a few iterations...")
print("(In practice, Word2Vec trains for millions of iterations)")

num_epochs = 50
for epoch in range(num_epochs):
    random.shuffle(training_pairs)
    total_loss = 0

    for center_idx, context_idx in training_pairs:
        loss = train_step(model, center_idx, context_idx)
        total_loss += loss

    if epoch % 10 == 0:
        avg_loss = total_loss / len(training_pairs)
        print(f"  Epoch {epoch:3d}: Average Loss = {avg_loss:.4f}")

print("\nTraining complete!")


# ============================================================================
# STEP 7: EXAMINING LEARNED EMBEDDINGS
# ============================================================================

print("\n7. THE LEARNED WORD EMBEDDINGS")
print("-" * 40)
print("After training, W1 contains our word embeddings:")
print("Each row is a word's vector representation\n")

# Show embeddings for some words
sample_words = ["cat", "dog", "pet", "play", "the"]
existing_words = [w for w in sample_words if w in word_to_idx]

print(f"Embeddings ({embedding_dim}-dimensional vectors):")
for word in existing_words[:5]:
    embedding = model.get_embedding(word, word_to_idx)
    print(f"  {word:8} : [{', '.join([f'{x:6.3f}' for x in embedding])}]")


# ============================================================================
# STEP 8: COMPUTING WORD SIMILARITY
# ============================================================================

print("\n8. WORD SIMILARITY USING EMBEDDINGS")
print("-" * 40)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2 + 1e-10)

print("Words with similar meanings should have similar embeddings.")
print("Cosine similarity ranges from -1 to 1 (1 = very similar)\n")

if len(existing_words) >= 2:
    word1, word2 = existing_words[0], existing_words[1]
    emb1 = model.get_embedding(word1, word_to_idx)
    emb2 = model.get_embedding(word2, word_to_idx)
    similarity = cosine_similarity(emb1, emb2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")


# ============================================================================
# STEP 9: VISUALIZING EMBEDDINGS
# ============================================================================

print("\n9. VISUALIZING WORD EMBEDDINGS")
print("-" * 40)

# For visualization, reduce to 2D using PCA
from sklearn.decomposition import PCA

# Get all word embeddings
all_embeddings = model.W1
words = [idx_to_word[i] for i in range(vocab_size)]

# Reduce to 2D
if embedding_dim > 2:
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    print(f"Reduced {embedding_dim}D embeddings to 2D for visualization")
else:
    embeddings_2d = all_embeddings

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Add labels
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=9, alpha=0.7)

plt.title("Word Embeddings Visualization (2D Projection)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("word2vec_embeddings.png", dpi=100)
print("Saved visualization to 'word2vec_embeddings.png'")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 80)
print("KEY CONCEPTS SUMMARY")
print("=" * 80)

print("""
1. EMBEDDINGS:
   - Dense vector representations of words
   - Capture semantic meaning in continuous space
   - Similar words have similar vectors

2. NEURAL NETWORK:
   - Simple architecture: Input -> Hidden -> Output
   - Hidden layer activations become word embeddings
   - Trained to predict context words from center word

3. PARAMETERS:
   - W1 (input->hidden): The embedding matrix we want!
   - W2 (hidden->output): Weights for prediction
   - Learned through backpropagation and gradient descent

The magic: After training, W1 contains meaningful word vectors that
capture relationships between words based on how they're used in context!
""")

print("\nThis simplified implementation demonstrates the core ideas.")
print("Real Word2Vec uses optimizations like negative sampling and")
print("hierarchical softmax for efficiency with large vocabularies.")