# HyperGAT with Syntactic Hyperedges

Enhanced Hypergraph Attention Network (HyperGAT++) for text classification incorporating **syntactic hyperedges** derived from dependency parsing and POS tagging.

## Overview

This project extends the HyperGAT architecture by introducing **syntactic hyperedges** as a third hyperedge type alongside sequential and semantic hyperedges. The model captures grammatical structures in text to improve classification performance.

### Key Innovations

1. **Dependency-based Syntactic Hyperedges** - Construct hyperedges from dependency parse trees (head-dependent relationships: nsubj, dobj, amod, nmod, etc.)
2. **POS-based Syntactic Hyperedges** - Group words by grammatical categories (nouns, verbs, adjectives, adverbs)
3. **Dual Attention Mechanism** - Node-level and hyperedge-level attention for hierarchical aggregation
4. **Inductive Learning** - Document-level graphs enable processing unseen documents without retraining

## Architecture

```
Document → Hypergraph Construction → HyperGAT Layers → Classification
                │
                ├─ Sequential Hyperedges (sentence-level)
                ├─ Semantic Hyperedges (LDA topics)
                └─ Syntactic Hyperedges (NEW)
                   ├─ Dependency-based (parse tree)
                   └─ POS-based (grammatical categories)
```

### Model Components

- **Embedding Layer**: GloVe (300d) for MR/Tweet; Xavier initialization for others
- **Hypergraph Attention Layers** (2 layers):
  - Layer 1: 300 → 300 dim, residual + LeakyReLU(0.1)
  - Layer 2: 300 → hiddenSize dim, dropout(0.3) + LayerNorm
- **Classification**: Masked mean pooling → Linear projection → Softmax

## Datasets

| Dataset | Documents | Classes | Vocab Size | Avg Length | Max Length |
|---------|-----------|---------|------------|------------|------------|
| SearchSnippets | 12,295 | 8 | 3,973 | 13.35 | 37 |
| Pascal-Flickr | 4,834 | 20 | 733 | 3.54 | 14 |
| GoogleNews | 11,108 | 152 | 2,445 | 5.05 | 11 |
| Biomedical | 19,448 | 20 | 3,497 | 6.73 | 27 |
| StackOverflow | 16,407 | 20 | 1,971 | 4.51 | 17 |
| Tweet | 2,472 | 89 | 5,076 | 8.48 | 20 |

## Installation

```bash
pip install torch numpy scikit-learn nltk gensim
```

Download GloVe embeddings (for MR/Tweet datasets):
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/
```

## Usage

### Training

```bash
python run.py --dataset SearchSnippets --model hypergat_syntactic --syntactic_type dependency
```

Arguments:
- `--dataset`: SearchSnippets, Pascal_Flickr, GoogleNews, Biomedical, StackOverflow, Tweet
- `--model`: hypergat_syntactic (with syntactic hyperedges), hypergat (baseline)
- `--syntactic_type`: dependency, pos
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_size`: Hidden dimension (default: 300)

### Preprocessing

```bash
python preprocess.py --dataset SearchSnippets
python generate_lda.py --dataset SearchSnippets --num_topics 50
```

## Hyperedge Construction

### Sequential Hyperedges
Each sentence forms a hyperedge connecting all words within it.

### Semantic Hyperedges
LDA with k topics → top-n words per topic form k hyperedges per document.

### Syntactic Hyperedges (Proposed)

**Dependency-based:**
```
E_syn = {w_head, w_dep1, w_dep2, ...}
```
Example: "Apple unveils new iPhone" → {unveils, Apple, iPhone}, {iPhone, new}, {camera, better}

**POS-based:**
```
E_pos = {w_i | POS(w_i) = t}
```
Groups: Nouns, Verbs, Adjectives, Adverbs

## Training Protocol

- **Optimizer**: Adam (weight_decay=1e-6)
- **Learning Rate**: 1e-3, decay 0.1 every 3 epochs
- **Loss**: Class-weighted Cross-Entropy
- **Regularization**: Dropout(0.3), LayerNorm
- **Early Stopping**: Patience=3, tolerance=0.001
- **Max Epochs**: 10

## Results

See `mtp_report.txt` for detailed results (Tables 5.2, 5.3, 5.4).

### Model Variants Compared
- HyperGAT (baseline: sequential + semantic)
- HyperGAT + Dependency syntactic
- HyperGAT + POS syntactic
- HyperGAT + Both syntactic types

## Project Structure

```
├── model.py              # Main HyperGAT model definition
├── layers.py             # Hypergraph attention layers
├── preprocess.py         # Data preprocessing & hyperedge construction
├── generate_lda.py       # LDA topic modeling for semantic hyperedges
├── run.py                # Training & evaluation script
├── utils.py              # Utility functions
├── data/                 # Datasets & precomputed files
│   *_corpus.txt          # Raw documents
│   *_labels.txt          # Labels
│   *_LDA.p               # Precomputed LDA models
│   vocab_dic.pkl         # Vocabulary mapping
│   labels_dic.pkl        # Label mapping
└── mtp_report.txt        # Thesis report (full text)
```

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{davender2025hypergat,
  title={Semantic Encoding of Textual Documents in a Graph Neural Network},
  author={Davender},
  year={2025},
  school={Indian Institute of Technology Guwahati}
}
```

## License

Academic research project. Contact author for usage permissions.

## Author

**Davender** (Roll No. 232123106)  
Department of Mathematics, IIT Guwahati  
Supervisor: Dr. Ashok Singh Sairam