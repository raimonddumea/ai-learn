#!/usr/bin/env python3
"""
Document Similarity using Bag-of-Words and Cosine Similarity

This script demonstrates how to measure similarity between text documents
using Bag-of-Words vectorization and cosine similarity metrics.

What it does:
- Converts documents to BoW vectors using CountVectorizer
- Calculates cosine similarity between all document pairs
- Identifies the most and least similar documents
- Shows common words between document pairs

Cosine Similarity:
- Measures the angle between two vectors (not their magnitude)
- Range: 0.0 (completely different) to 1.0 (identical)
- Example: "cat sat" and "cat ran" would have high similarity due to shared "cat"

Usage:
    python3 bow_similarity.py

Requirements:
    - scikit-learn
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A cat and a dog are pets",
        "I love my pet cat"
    ]

    # Create BoW vectors
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between all documents
    similarities = cosine_similarity(vectors)

    print("Document Similarity Matrix:")
    print("(1.0 = identical, 0.0 = no common words)")
    print("\n    ", end="")
    for i in range(len(documents)):
        print(f"Doc{i+1:2}", end="  ")
    print()

    for i in range(len(documents)):
        print(f"Doc{i+1}", end=" ")
        for j in range(len(documents)):
            print(f"{similarities[i][j]:5.2f}", end=" ")
        print()

    # Show pairwise similarities with document text
    print("\nPairwise Document Similarities:")
    print("-" * 50)
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            similarity = similarities[i][j]
            print(f"\nDoc {i+1} vs Doc {j+1}: {similarity:.3f}")
            print(f"  '{documents[i]}'")
            print(f"  '{documents[j]}'")

            # Explain similarity
            if similarity > 0:
                # Find common words
                words1 = set(documents[i].lower().split())
                words2 = set(documents[j].lower().split())
                common = words1.intersection(words2)
                print(f"  Common words: {common}")

    # Find most and least similar document pairs
    print("\n" + "=" * 50)
    print("Summary:")

    # Flatten similarity matrix (excluding diagonal)
    sim_scores = []
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            sim_scores.append((similarities[i][j], i, j))

    sim_scores.sort(reverse=True)

    print(f"\nMost similar pair: Doc {sim_scores[0][1]+1} & Doc {sim_scores[0][2]+1} (similarity: {sim_scores[0][0]:.3f})")
    print(f"Least similar pair: Doc {sim_scores[-1][1]+1} & Doc {sim_scores[-1][2]+1} (similarity: {sim_scores[-1][0]:.3f})")