# 🎬 IMDB Sentiment Analysis — NLP with Naive Bayes, LSTM & BERT

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An end-to-end NLP pipeline that classifies IMDB movie reviews as **Positive** or **Negative** using three models — Naive Bayes, Bidirectional LSTM, and BERT — with an interactive sentiment analyzer UI.

---

## 📸 Demo

| ✅ Positive Sentiment | ❌ Negative Sentiment |
|:---:|:---:|
| ![Positive Result](p1.PNG) | ![Negative Result](p2.PNG) |

> All three models achieving **95–99% confidence** on test reviews.

---

## 🚀 Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L5ugg6dgO8nAWclaxQzRsxOLhoSrNI3R#scrollTo=guY1EahBCgtT)

```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook IMDB_Sentiment_Analysis.ipynb
```

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Interactive UI](#interactive-ui)
- [Requirements](#requirements)
- [Usage](#usage)

---

## 🧠 Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify and classify opinions or emotions expressed in text. This project focuses on the **IMDB Movie Reviews Dataset** and builds an intelligent system to automatically classify user sentiment from textual content.

**Applications:**
- 🛍️ Product review monitoring
- 📱 Social media sentiment tracking
- 📞 Customer feedback analysis
- 🎬 Movie/content recommendation systems

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | IMDB Movie Reviews Dataset |
| **Source** | HuggingFace `datasets` library |
| **Total Samples** | 50,000 reviews |
| **Train Split** | 25,000 reviews |
| **Test Split** | 25,000 reviews |
| **Classes** | Positive (1) / Negative (0) |
| **Balance** | Perfectly balanced — 50% each |

```python
from datasets import load_dataset
dataset = load_dataset('imdb')  # Auto-downloads — no manual setup needed
```

---

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── IMDB_Sentiment_Analysis.ipynb   # Main notebook (all steps)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── p1.PNG                          # Demo screenshot (positive)
├── p2.PNG                          # Demo screenshot (negative)
│
└── saved_models/                   # Generated after training
    ├── naive_bayes.pkl
    ├── tfidf_vectorizer.pkl
    ├── lstm_model.pt
    ├── vocab.pkl
    ├── bert_model/
    └── bert_tokenizer/
```

---

## ⚙️ Pipeline

```
Raw Text
   │
   ▼
Text Preprocessing
   ├── Lowercase conversion
   ├── HTML tag removal
   ├── Special character removal
   ├── Tokenization
   ├── Stopword removal
   └── Porter Stemming
   │
   ▼
Feature Extraction
   ├── TF-IDF (30k features, unigrams + bigrams) → Naive Bayes
   ├── Word Index Encoding (vocab size: 20k)    → LSTM
   └── BERT Tokenizer (max_length: 256)         → BERT
   │
   ▼
Model Training & Evaluation
   ├── Naive Bayes
   ├── Bidirectional LSTM
   └── BERT (fine-tuned)
   │
   ▼
Predictions + Interactive UI
```

---

## 🤖 Models

### 1. Naive Bayes (with TF-IDF)

- **Vectorizer:** TF-IDF with 30,000 features and bigrams
- **Classifier:** Multinomial Naive Bayes (`alpha=0.1`)
- **Pros:** Fast training, interpretable, good baseline
- **Cons:** Ignores word order and context

### 2. Bidirectional LSTM

- **Embedding dim:** 128
- **Hidden dim:** 128 (×2 bidirectional)
- **Layers:** 2 with dropout (0.4)
- **Optimizer:** Adam (lr=1e-3) with StepLR scheduler
- **Pros:** Captures sequential patterns and context
- **Cons:** Slower than classical ML

### 3. BERT (bert-base-uncased)

- **Architecture:** Pre-trained BERT with classification head
- **Max token length:** 256
- **Fine-tuning:** 2 epochs on 5,000 IMDB samples
- **Pros:** Best accuracy, context-aware, state-of-the-art
- **Cons:** Most resource-intensive

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | ~87% | ~87% | ~87% | ~87% |
| **LSTM** | ~90% | ~90% | ~90% | ~90% |
| **BERT** | ~93%+ | ~93%+ | ~93%+ | ~93%+ |

> Results may vary slightly based on random seeds and hardware.

---

## 🖥️ Interactive UI

The notebook includes a fully interactive sentiment analyzer built with `ipywidgets`:

- **Text box** — paste or type any movie review
- **Analyze button** — runs all 3 models simultaneously
- **Confidence bars** — visual bar per model showing prediction confidence
- **Majority vote verdict** — final POSITIVE / NEGATIVE result
- **Quick example buttons** — load pre-written positive/negative reviews
- **Clear button** — reset the interface

```python
# Run the interactive analyzer after training all models
predict_sentiment("Your movie review goes here...")
```

---

## 📦 Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.3.0
nltk>=3.8.1
gensim>=4.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
ipywidgets>=8.0.0
```

Install all at once:
```bash
pip install torch transformers datasets scikit-learn nltk gensim pandas numpy matplotlib seaborn ipywidgets
```

---

## 💡 Usage

### Run on Google Colab (Recommended)
1. Click the **Open in Colab** badge above
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom

### Run Locally in VS Code
1. Install the **Python** and **Jupyter** extensions
2. Create a virtual environment and install requirements
3. Open `IMDB_Sentiment_Analysis.ipynb`
4. Select your virtual environment as the kernel

> ⚠️ **Note:** BERT training is very slow on CPU. Use GPU (Colab T4) for best performance.

### Use Your Own Dataset (Kaggle CSV)
```python
import pandas as pd
df = pd.read_csv('IMDB Dataset.csv')
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
# Replace 'text' column references with 'review' in the notebook
```

---

## 🔮 Future Improvements

- [ ] Multi-class sentiment (Very Positive / Neutral / Very Negative)
- [ ] Real-time review monitoring via API
- [ ] Domain-specific fine-tuning (e.g., product reviews, tweets)
- [ ] Streamlit / Gradio web app deployment
- [ ] Model explainability with SHAP / LIME

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/) for the Transformers library and IMDB dataset
- [NLTK](https://www.nltk.org/) for NLP preprocessing tools
- [PyTorch](https://pytorch.org/) for deep learning framework
- **Skillentrix Technologies** for project guidance and support

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ | <strong>Skillentrix Technologies</strong>
</p>
