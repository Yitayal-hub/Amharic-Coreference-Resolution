# Amharic Neural Coreference Resolution with Multi-Head Attention and Named Entity Recognition

This repository contains the official implementation of the paper **"Amharic neural coreference resolution with multi-head attention and named entity recognition"** by Yitayal Abate, Yaregal Assabie, and Wolfgang Menzel, published in **Scientific Reports (2026)**.

The model performs **coreference resolution** for Amharic, a morphologically rich Semitic language, using a neural architecture that integrates **multi-head attention (MHA)** and a **named entity recognition (NER)** module.

---

## 📋 Overview

Coreference resolution is a fundamental task in natural language processing that identifies all textual expressions referring to the same real-world entity. This task presents particular challenges for Amharic due to its rich morphology, pro-drop nature, flexible word order, and limited annotated resources compared to widely studied languages.

### Key Contributions

1. **Neural coreference resolution system** for Amharic integrating multi-head attention (MHA) and NER.
2. **Multi-head attention** at two critical stages: pre-BiLSTM contextualization and within the coreference layer.
3. **NER module** with BiLSTM and Conditional Random Field (CRF) to enhance mention detection.
4. **Comprehensive architecture** comprising preprocessing, morphological analysis, contextualization, span generation, mention scoring and pruning, and iterative coreference resolution.
5. **Custom-built Amharic coreference dataset** with 250 documents.

---

## 📁 Repository Contents

```
.
├── Amcoref_datasets.json              # Complete annotated dataset
├── Amcoref_Source_Code/               # All source code files
│   ├── coref_model.py                 # Main coreference resolution model
│   ├── coref_ops.py                   # Coreference operations (CPU)
│   ├── coref_ops_gold_mentions.py     # Operations for gold mentions mode
│   ├── coref_kernels.cc               # Custom TensorFlow kernels
│   ├── coref_kernels_gold_mentions.cc # Gold mentions kernels
│   ├── util.py                        # Utility functions
│   ├── metrics.py                     # Evaluation metrics
│   ├── conll.py                       # CoNLL evaluation scripts
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── filter_embeddings.py           # Embedding filtering utility
│   ├── get_char_vocab.py              # Character vocabulary builder
│   ├── experiments.conf               # Configuration file
│   ├── requirements.txt               # Python dependencies
│   ├── setup_all.sh                   # Full setup script
│   └── setup_training.sh              # Training setup script
└── README.md                          # This file
```

---

## 🏗️ Model Architecture

The system architecture consists of the following main components:

### 1. Preprocessing & Morphological Analysis
- Text cleaning, sentence segmentation, and word tokenization.
- Morphological segmentation using **HornMorphoAX** to break words into constituent morphemes.

### 2. Morphological Embeddings
- **Word Embeddings**: 300-dimensional pre-trained Word2Vec vectors.
- **Character Embeddings**: 50-dimensional embeddings processed by BiLSTM for morpheme-level representations.

### 3. Contextualization Layer
- **Multi-Head Attention**: Applied before BiLSTM to provide initial global contextualization.
- **BiLSTM**: 200-dimensional hidden size (per direction) with 0.4 dropout.

### 4. NER Module
- **BiLSTM + CRF** architecture for named entity recognition.
- Predicts entity types (PER, ORG, LOC) to guide mention detection.
- NER predictions are used to generate additional candidate spans.

### 5. Span Generation & Representation
- Regular candidate spans with maximum width of **20** tokens.
- NER-based candidate generation.
- Span embeddings combine start/end embeddings, span width embedding, and head attention.

### 6. Mention Scoring & Pruning
- Feedforward neural network (FFNN) scores each candidate span.
- Top 25% of spans selected as candidate mentions.
- **Coarse-to-fine pruning** for antecedent selection (max 50 antecedents).

### 7. Coreference Resolution Layer
- **Multi-head attention** for antecedent scoring (4 heads, 64 attention size).
- Iterative refinement with entity-level representations.
- Softmax over antecedents (including dummy antecedent) determines final clusters.

### 8. Loss Functions
- **B³ Loss**: Entity-level optimization.
- **Antecedent Loss**: Softmax cross-entropy for coreference linking.
- **Mention Loss**: Sigmoid cross-entropy with OHEM.
- **NER Loss**: CRF loss (weight: 0.15).
- **L2 Regularization**: Applied to trainable variables (excluding biases).

---

## 📊 Dataset

The dataset consists of **250 documents** collected from web sources and expanded from a previous pronominal anaphora dataset.

| Split | Percentage | # Documents |
|-------|------------|-------------|
| Training | 80% | 200 |
| Development | 10% | 25 |
| Testing | 10% | 25 |

**Key Characteristics**:
- Annotated with coreference clusters (including singletons).
- Includes NER tags (PER, ORG, LOC).
- Morphologically segmented using HornMorphoAX.
- Annotated in CoNLL-U format, converted to JSON lines.
- Validated by native Amharic speakers under expert supervision.

### Data Format

Each line in the JSON file contains:

```json
{
  "doc_key": "document_id",
  "sentences": [["token1", "token2", ...], ...],
  "speakers": [["speaker1", "speaker2", ...], ...],
  "clusters": [[[start, end], [start, end], ...], ...],
  "ner": [["PER", "O", "ORG", ...], ...]
}
```

- `start` and `end` are **inclusive** indices in the **flattened** document.
- `clusters` contains lists of mentions that refer to the same entity.
- `ner` contains NER tags for each token.

---

## 🔧 Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Character Embedding | Dimension | 50 |
| Character LSTM | Size | 100 |
| Word Embedding | Dimension | 300 |
| NER Embedding | Dimension | 20 |
| Contextual Embedding | BiLSTM Hidden Size | 200 |
| Contextual Embedding | Layers | 2 |
| FFNN | Hidden Size | 100 |
| FFNN | Depth | 2 |
| Multi-Head Attention | Heads | 4 |
| Multi-Head Attention | Attention Size | 64 |
| NER | LSTM Size | 200 |
| NER | CRF | Enabled |
| Training | Optimizer | Adam |
| Training | Learning Rate | 0.0005 |
| Training | Dropout (FFNN) | 0.3 |
| Training | Dropout (LSTM) | 0.4 |
| Training | Dropout (Embedding) | 0.5 |
| Training | Steps / Epochs | 10,000 (~50 epochs) |
| Training | L2 Lambda | 0.01 |
| Training | NER Loss Weight | 0.15 |
| Pruning | Max Span Width | 20 |
| Pruning | Top Span Ratio | 0.25 |
| Pruning | Max Top Antecedents | 50 |
| Pruning | Max NER Span Width | 10 |
| Morphological Analyzer | HornMorphoAX | 92.3% accuracy, 98% coverage |

Full configuration can be found in [`experiments.conf`](https://github.com/Yitayal-hub/Amharic-Coreference-Resolution/blob/main/Amcoref_Source_Code/experiments.conf).

---

## 📈 Results

The model was evaluated using standard coreference metrics: **MUC**, **B³**, **CEAF₄**, and the **CoNLL average F1**.

### Overall Performance (Coarse-to-Fine Pruning)

| Metric | Precision (%) | Recall (%) | F1 (%) |
|--------|---------------|------------|--------|
| **MUC** | 68.72 | 75.88 | **72.02** |
| **B³** | 26.83 | 72.11 | **38.33** |
| **CEAF₄** | 31.36 | 19.67 | **23.25** |
| **CoNLL** | — | — | **44.60** |

### Development Set Performance

| Metric | Value |
|--------|-------|
| **Maximum F1** | 68.58% |

### Ablation Study: Impact of Multi-Head Attention

| Model | CoNLL F1 (%) |
|-------|---------------|
| **With MHA and NER** | **44.60** |
| **With MHA Only** | 44.12 |
| **Without MHA** | 42.49 |

### Ablation Study: Impact of NER Features

| Model | CoNLL F1 (%) |
|-------|---------------|
| **With Gold NER (NERG)** | **45.72** |
| **With Predicted NER (NERP)** | **44.60** |
| **Without NER** | 44.12 |

### Comparative Performance

| Language / System | CoNLL F1 (%) | Dataset Size | Context |
|-------------------|--------------|--------------|---------|
| **Our Model (Amharic)** | **44.60** | 250 documents | Low-resource, morphologically rich, pro-drop |
| Turkish [86] | 59.3 | 33 documents | Low-resource; first neural models (BERTurk) |
| Persian [88] | 76.16 | 400 documents | Medium-resource; end-to-end model |
| Arabic [2] | 63.9 | 447 documents | Medium-resource; AraBERT |
| English [63] | >80 | ~3,000 documents | High-resource; extensively studied |

**Key Findings**:
- Multi-head attention improves CoNLL F1 by **1.63 percentage points**.
- Gold NER information improves CoNLL F1 by **1.12 percentage points** over predicted NER.
- The model achieves competitive performance for a low-resource language.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.0+
- Other dependencies listed in [`requirements.txt`](https://github.com/Yitayal-hub/Amharic-Coreference-Resolution/blob/main/Amcoref_Source_Code/requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/Yitayal-hub/Amharic-Coreference-Resolution.git
cd Amharic-Coreference-Resolution

# Install dependencies
pip3 install -r Amcoref_Source_Code/requirements.txt
```

### Setup

Build custom TensorFlow kernels:

```bash
bash -x -e Amcoref_Source_Code/setup_all.sh
```

### Data Preparation

1. **Download Amharic Word2Vec embeddings** from:  
   [https://sparknlp.org/2022/03/14/w2v_cc_300d_am_3_0.html](https://sparknlp.org/2022/03/14/w2v_cc_300d_am_3_0.html)

2. **Prepare embeddings**:

```bash
bash -x -e Amcoref_Source_Code/setup_training.sh
```

3. **Filter embeddings** for your dataset:

```bash
python3 Amcoref_Source_Code/filter_embeddings.py cc.am.300.vec train.jsonlines dev.jsonlines test.jsonlines
```

4. **Generate character vocabulary**:

```bash
python3 Amcoref_Source_Code/get_char_vocab.py
```

### Training

```bash
python3 Amcoref_Source_Code/train.py <experiment_name> --logdir <log_directory>
```

**Example**:

```bash
python3 Amcoref_Source_Code/train.py train_am_coref --logdir logs/amharic_coref
```

The training script uses the configuration from `experiments.conf`.

### Evaluation

```bash
python3 Amcoref_Source_Code/evaluate.py <experiment_name>
```

### Output Format

The system outputs predictions in **JSON Lines** format with the following structure:

```json
{
  "doc_key": "document_id",
  "clusters": [[[start, end], ...], ...],
  "sentences": [...],
  "predicted_ner": ["PER", "O", "ORG", ...],
  ...
}
```

---

## 🧩 Implementation Notes

### Gold Mentions Mode

The model supports two modes via the `use_gold_mentions` flag:
- **True**: Uses gold-standard mentions for training.
- **False**: The model must learn to propose mentions from scratch.

### Multi-Head Attention

Multi-head attention is used in two places:
1. **Pre-LSTM Attention**: Provides initial global contextualization of input embeddings.
2. **Antecedent Attention**: Refines "slow antecedent scores" within the coreference layer.

### NER Integration

The NER module:
1. Predicts named entities (PER, ORG, LOC) using BiLSTM-CRF.
2. Generates additional candidate spans from predicted NER tags.
3. Provides explicit semantic type embeddings to filter incompatible coreference links.

### Character Encoding

The model uses a **BiLSTM** for character-level representations, replacing the traditional CNN used in earlier work, to better capture Amharic's morphological complexity.

### Custom TensorFlow Kernels

The repository includes custom C++ kernels for efficient span extraction operations. These need to be compiled as part of the setup process.

---

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{abate2026amharic,
  title={Amharic neural coreference resolution with multi-head attention and named entity recognition},
  author={Abate, Yitayal and Assabie, Yaregal and Menzel, Wolfgang},
  journal={Scientific Reports},
  year={2026},
  doi={10.1038/s41598-026-50661-5}
}
```

---

## 📜 License

### Code License

**Note**: This repository does not currently contain a license file. Please contact the authors for permission regarding usage, modification, and distribution.

### Paper License

The paper is published under a **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

This license:
- Permits **non-commercial** use, sharing, distribution, and reproduction.
- Requires **appropriate credit** to the original author(s) and source.
- Does **not** permit sharing adapted material derived from this article.

For details, visit: [http://creativecommons.org/licenses/by-nc-nd/4.0/](http://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## 🙏 Acknowledgements

- **Addis Ababa University** for financial support.
- **University of Hamburg** for collaboration.
- **HornMorphoAX** developers (Michael Gasser) for morphological analysis tools.

---

## 📧 Contact

For questions or collaborations, please contact:

- **Yitayal Abate**: yitayal.abate@aau.edu.et
- **Yaregal Assabie**: yaregal.assabie@aau.edu.et
- **Wolfgang Menzel**: Department of Informatics, University of Hamburg

---

## References

1. Abate, Y., Assabie, Y. & Menzel, W. Amharic neural coreference resolution with multi-head attention and named entity recognition. *Sci Rep* (2026). https://doi.org/10.1038/s41598-026-50661-5
2. Lee, K., He, L., Lewis, M., & Zettlemoyer, L. (2017). End-to-end Neural Coreference Resolution. *EMNLP 2017*.
3. Lee, K., He, L., & Zettlemoyer, L. (2018). Higher-order Coreference Resolution with Coarse-to-Fine Inference. *NAACL 2018*.
4. Aloraini, A., Yu, J., and Poesio, M. (2020). Neural Coreference Resolution for Arabic. *CRAC 2020*.
5. Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., & Levy, O. (2020). SpanBERT. *TACL*.
