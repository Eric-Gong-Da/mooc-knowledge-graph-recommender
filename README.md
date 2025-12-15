# MOOC Recommendation System Based on Knowledge Graph Embeddings

This project implements a Massive Open Online Course (MOOC) recommendation system using knowledge graph embeddings. The system leverages meta-path random walks on a heterogeneous knowledge graph to learn user and content embeddings, then performs course recommendations using K-nearest neighbors (KNN) and Random Forest algorithms.

## Authors

- Gong Da - The Education University of Hong Kong (s1153651@s.eduhk.hk)
- Li Haolin - The Education University of Hong Kong (s1153657@s.eduhk.hk)
- Chan Cheuk Ying - The Education University of Hong Kong (s1155604@s.eduhk.hk)
- Chan Ka Man - The Education University of Hong Kong (s1155229@s.eduhk.hk)
- Zhu Jiayin - The Education University of Hong Kong (s1153658@s.eduhk.hk)

## Project Structure

```
mooc_recommendation_system/
├── sample_data/           # Sample relation data (first 100 lines of each file)
├── src/                   # Source code
│   ├── generate_kg_triples.py       # Convert raw relations to knowledge graph triples
│   ├── random_walk_embedding.py     # Generate embeddings using meta-path random walks
│   ├── knn_course_recommendation.py # KNN-based course recommendation
│   └── knn_rf_recommendation.py     # Combined KNN + Random Forest recommendation
├── docs/                  # Documentation and academic paper
│   ├── paper.tex          # LaTeX source for academic paper
│   └── paper.pdf          # Compiled academic paper with complexity analysis
├── complexity_analysis/   # Runtime complexity analysis
│   ├── analyze_complexity.py        # Script to evaluate runtime complexity
│   └── results/           # Generated plots and analysis report
└── README.md              # This file
```

## Dataset Information

The original dataset is from MOOCCube and contains large relation files. For demonstration purposes, this repository includes only sample data (first 100 lines of each relation file) to keep the repository size manageable.

**Original dataset source:** http://moocdata.cn/data/MOOCCube

**Note:** The full dataset contains millions of records. The sample data is provided for demonstration only. For actual experiments, please download the complete dataset from the official source.

## Core Architecture

The system consists of three main processing stages:

1. **Knowledge Graph Construction** (`generate_kg_triples.py`)
   - Converts raw relation files from `sample_data/` directory into structured triple format
   - Outputs processed triples to `kg_triples/` directory
   - Handles multiple relation types: user-course enrollment, user-video watching, course-concept teaching, concept hierarchy, prerequisite dependencies, and video-concept coverage

2. **Embedding Generation** (`random_walk_embedding.py`)
   - Performs meta-path random walks on the knowledge graph following the pattern: User → Video → Concept → Video → ...
   - Uses Word2Vec (skip-gram) to learn 128-dimensional embeddings from random walk sequences
   - Supports checkpoint-based resumable processing for large-scale data
   - Outputs user embeddings to files like `user_embeddings_128d.txt`

3. **Course Recommendation** (`knn_course_recommendation.py` and `knn_rf_recommendation.py`)
   - Uses KNN with cosine similarity to find similar users based on embeddings
   - Implements both pure KNN and combined KNN + Random Forest approaches
   - Aggregates courses from similar users to generate recommendations
   - Filters out courses already taken by the target user

## Key Implementation Details

### Meta-Path Random Walk Strategy
The random walk alternates between entity types (User → Video → Concept → Video → ...) to capture heterogeneous relationships in the knowledge graph. This preserves semantic information about user learning behaviors.

### Data Format
- Relation files: tab-separated format `entity1_id\tentity2_id`
- Triple files: `head\trelation\ttail`
- Embedding files: `user_id\tdim1 dim2 ... dim128` (space-separated floats)

## Requirements

- Python 3.6+
- gensim
- scikit-learn
- numpy
- tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate knowledge graph triples:
```bash
python src/generate_kg_triples.py
```

2. Generate user embeddings:
```bash
python src/random_walk_embedding.py
```

3. Run course recommendation:
```bash
python src/knn_course_recommendation.py
```

Or for combined KNN + Random Forest approach:
```bash
python src/knn_rf_recommendation.py
```

## Complexity Analysis

We performed a comprehensive runtime complexity analysis of each system component:

1. **Knowledge Graph Construction**: O(E) where E is the number of relationships
2. **Meta-Path Random Walk Generation**: O(U × W × L × D) where U=users, W=walks per user, L=walk length, D=average degree
3. **Embedding Training**: O(W × L × V) where W=walks, L=average walk length, V=vocabulary size
4. **KNN Recommendation**: O(N × log(N) × D) for efficient implementations where N=users, D=embedding dimension

Detailed analysis results and visualizations are available in:
- `complexity_analysis/report.pdf` (Dedicated complexity analysis report with Big O notation)
- `docs/paper.pdf` (Academic paper with complexity analysis section)
- `complexity_analysis/results/` (Generated plots and detailed report)

## Evaluation

The system includes evaluation metrics to measure recommendation quality, including precision, recall, and F1-score.