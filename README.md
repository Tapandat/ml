# Mathematical Correctness Checker

An ML-powered system that evaluates the correctness of mathematical solutions/answers using **Sentence-BERT (SBERT)** embeddings and **cosine similarity**, rather than relying on brittle exact-match or rule-based comparison.

## Overview

Traditional answer-checking systems rely on exact string matching or handcrafted rules, which fail when a mathematically correct answer is expressed in a different but equivalent form (different notation, ordering, phrasing, or intermediate steps). This project addresses that gap by using semantic similarity to judge whether a submitted solution is mathematically equivalent to a reference solution.

## Problem Statement

Automated grading and self-assessment tools for math problems often mark correct-but-differently-phrased answers as wrong. This project builds a correctness checker that understands the *meaning* of a mathematical answer/solution rather than just its surface text, enabling more robust and fair evaluation.

## Approach

1. **Text Representation** — Encode both the reference (ground-truth) solution and the candidate (user-submitted) solution into dense vector embeddings using **SBERT (Sentence-BERT)**.
2. **Similarity Scoring** — Compute **cosine similarity** between the two embeddings to quantify how semantically close the candidate solution is to the reference.
3. **Correctness Decision** — Apply a similarity threshold to classify the candidate solution as *correct*, *partially correct*, or *incorrect*.

## Tech Stack

- **Language:** Python
- **NLP / Embeddings:** Sentence-BERT (SBERT)
- **ML Tooling:** scikit-learn, PyTorch
- **Similarity Metric:** Cosine similarity

## Project Structure

```
math-correctness-checker/
├── data/               # Sample problems & reference solutions
├── src/
│   ├── embed.py        # SBERT embedding generation
│   ├── similarity.py   # Cosine similarity scoring
│   └── evaluate.py     # End-to-end correctness evaluation pipeline
├── notebooks/          # Experimentation & threshold tuning
├── requirements.txt
└── README.md
```

## How It Works

```
Reference Solution ─┐
                     ├─► SBERT Encoder ─► Embeddings ─► Cosine Similarity ─► Correctness Verdict
Candidate Solution ──┘
```

## Getting Started

### Prerequisites
```bash
pip install sentence-transformers scikit-learn torch
```

### Usage
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

reference = "The derivative of x^2 is 2x"
candidate = "d/dx(x^2) = 2x"

emb = model.encode([reference, candidate])
score = cosine_similarity([emb[0]], [emb[1]])[0][0]

print(f"Similarity Score: {score:.2f}")
print("Correct" if score > 0.75 else "Incorrect")
```

## Results

The embedding-based approach outperforms exact-match and keyword-based baselines by correctly identifying semantically equivalent solutions expressed with different wording, notation, or step ordering.

## Future Work

- Extend beyond sentence-level similarity to structured/symbolic comparison (e.g., SymPy-based equivalence checking) for higher precision on equation-heavy answers.
- Fine-tune SBERT on a math-specific corpus to improve domain sensitivity.
- Add a partial-credit scoring mechanism based on step-wise similarity.

## Author

**Tapan Datta**
[GitHub](https://github.com/Tapandat) · [LinkedIn](https://linkedin.com/in/tapan-datta-473958319)
