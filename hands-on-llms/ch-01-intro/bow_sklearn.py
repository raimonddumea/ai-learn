#!/usr/bin/env python3
"""
Bag-of-Words using Scikit-Learn's CountVectorizer

This script demonstrates how to use scikit-learn's built-in CountVectorizer
for creating Bag-of-Words representations of text documents.

What it does:
- Uses CountVectorizer to automatically build vocabulary and vectorize text
- Shows the word-to-index mapping created by the vectorizer
- Converts documents into sparse matrix format (memory efficient)

Key differences from manual implementation:
- Excludes single-letter words by default (no 'a', 'i')
- Returns sparse matrices instead of dense arrays
- Offers many configuration options (ngrams, stop words, etc.)

Usage:
    python3 bow_sklearn.py

Requirements:
    - scikit-learn
"""
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A cat and a dog are pets",
        "I love my pet cat"
    ]

    # Create and fit CountVectorizer
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents)

    # Show vocabulary
    print("Vocabulary (alphabetically sorted):")
    print(f"{vectorizer.get_feature_names_out()}")
    print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")

    # Show word-to-index mapping
    print("\nWord to index mapping:")
    for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
        print(f"  {word}: {idx}")

    # Show document vectors
    print("\nDocument vectors:")
    for i, (doc, vector) in enumerate(zip(documents, vectors.toarray())):
        print(f"Doc {i+1}: '{doc}'")
        print(f"Vector: {vector}")
        print()

    # Transform new documents
    print("New document vectors:")
    new_docs = ["The cat loves the park", "A dog sat"]
    new_vectors = vectorizer.transform(new_docs)
    for doc, vector in zip(new_docs, new_vectors.toarray()):
        print(f"'{doc}' -> {vector}")