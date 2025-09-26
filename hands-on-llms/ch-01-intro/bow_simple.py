#!/usr/bin/env python3
"""
Bag-of-Words (BoW) Implementation from Scratch

This script demonstrates how to build a simple Bag-of-Words text vectorizer
without using external NLP libraries (except numpy for arrays).

What it does:
- Builds a vocabulary from a collection of text documents
- Converts text into numerical vectors based on word frequencies
- Each document becomes a vector where each position represents a word count

Example output:
    "The cat sat" -> [0, 1, 0, 0, ..., 1, 1]
    where positions correspond to words in the vocabulary

Usage:
    python3 bow_simple.py

Requirements:
    - numpy
"""
import numpy as np
from collections import Counter
import re

class SimpleBagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.vocab_size = 0

    def preprocess_text(self, text):
        """Simple text preprocessing: lowercase and remove punctuation"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def fit(self, documents):
        """Build vocabulary from a list of documents"""
        all_words = set()
        for doc in documents:
            words = self.preprocess_text(doc)
            all_words.update(words)

        # Create word-to-index mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.vocab_size = len(self.vocabulary)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Vocabulary: {list(self.vocabulary.keys())}")

    def transform(self, documents):
        """Convert documents to BoW vectors"""
        vectors = []

        for doc in documents:
            words = self.preprocess_text(doc)
            word_counts = Counter(words)

            # Create vector of zeros
            vector = np.zeros(self.vocab_size)

            # Fill in word counts
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] = count

            vectors.append(vector)

        return np.array(vectors)

if __name__ == "__main__":
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A cat and a dog are pets",
        "I love my pet cat"
    ]

    # Create and train BoW model
    bow = SimpleBagOfWords()
    bow.fit(documents)

    # Transform documents
    vectors = bow.transform(documents)

    print("\nDocument vectors:")
    for i, (doc, vector) in enumerate(zip(documents, vectors)):
        print(f"Doc {i+1}: '{doc}'")
        print(f"Vector: {vector.astype(int)}")
        print()

    # Transform new documents
    print("New document vectors:")
    new_docs = ["The cat loves the park", "A dog sat"]
    new_vectors = bow.transform(new_docs)
    for doc, vector in zip(new_docs, new_vectors):
        print(f"'{doc}' -> {vector.astype(int)}")