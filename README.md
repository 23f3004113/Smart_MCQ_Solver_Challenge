# Smart MCQ Solver Challenge

## Student Details

* **Name:** Ayush Sur
* **Roll Number:** 23f3004113
* **Program:** IIT Madras BS Degree in Data Science and Applications
* **Course:** Introduction to Deep Learning and Generative AI Project
* **Term:** May Term 2026 (T2-2026)
* **Kaggle ID:** `ayushsur2003`

---

# Project Overview

This repository contains the final implementation of the **Smart MCQ Solver Challenge**, developed as part of the IIT Madras BS Degree in Data Science and Applications.

The objective of the competition is to predict and rank the most probable answers for multiple-choice questions. Each question contains **five candidate options (A, B, C, D, E)**, and the system must return the **top three ranked options**.

The primary competition evaluation metric is **Mean Average Precision at 3 (MAP@3)**.

The project follows an end-to-end NLP and Deep Learning workflow:

1. Exploratory Data Analysis
2. Data preprocessing and feature construction
3. TF-IDF representation
4. Custom neural network trained from scratch
5. Fine-tuning of BERT
6. Fine-tuning of RoBERTa
7. Validation using Accuracy, Macro F1 and MAP@3
8. Top-3 prediction generation
9. Kaggle submission generation
10. Model comparison and final model selection
11. RoBERTa-based deployment

The final model selected for submission is **RoBERTa-base**.

---

# Problem Statement

For every MCQ, the model receives:

* A question/prompt
* Five candidate answers: A, B, C, D and E

The model produces a score for each of the five choices. These scores are converted into a ranking, and the three highest-ranked options are submitted.

The competition therefore evaluates both:

* **Classification performance**
* **Ranking quality**

MAP@3 is particularly important because the correct answer receives partial credit when it appears at rank 2 or rank 3.

---

# Dataset

The dataset consists of:

| Split      | Samples | Purpose          |
| ---------- | ------: | ---------------- |
| Training   |   1,406 | Model training   |
| Validation |     352 | Model evaluation |
| Test       |     500 | Kaggle inference |

The training data contains the correct answer label, while the test data is used to generate the Kaggle submission.

The final notebook uses a stratified **80:20 train-validation split** with:

```text
random_state = 42
stratify = label
```

This produces:

```text
Training   : 1,406
Validation :   352
Test       :   500
```

---

# Data Preprocessing

The final preprocessing pipeline was implemented directly in the final notebook.

## 1. Missing Value Checks

Missing values were checked separately for the training and test datasets.

## 2. Duplicate MCQ Removal

Duplicate questions were examined and duplicate MCQs were removed using the complete set of question and answer-option columns.

The preprocessing first removes duplicate rows based on:

```text
prompt, A, B, C, D, E
```

Duplicate prompts are subsequently removed to ensure that the same prompt does not appear multiple times in the training data.

## 3. Combined Text Construction

For the TF-IDF model, the question and all five candidate options are combined into a single text representation.

The final construction is conceptually:

```text
prompt
A
B
C
D
E
```

The same combined-text construction is applied to both training and test data.

## 4. Label Encoding

The answer labels are encoded as:

```text
A → 0
B → 1
C → 2
D → 3
E → 4
```

The reverse mapping is also maintained for generating Kaggle predictions.

## 5. Duplicate and Leakage Checks

The notebook also checks:

* Duplicate prompts
* Duplicate combined text
* Conflicting labels for identical combined text
* Test prompts appearing in the training set
* Test combined text appearing in the training set

These checks were used to verify the dataset before model training.

---

# Model Architecture

The final project contains three model families.

```text
                         Smart MCQ Solver
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
        TF-IDF + NN          BERT           RoBERTa
        Scratch Model      Pretrained       Pretrained
              |                |                |
              +----------------+----------------+
                               |
                               v
                       Top-3 Ranking
                               |
                               v
                            MAP@3
```

---

# Model 1 — TF-IDF + Neural Network

The first deep learning model is a custom feed-forward neural network trained from scratch using TF-IDF features.

This model replaced the earlier Logistic Regression approach.

## TF-IDF Configuration

The final notebook uses:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    max_features=100000,
    min_df=2,
    sublinear_tf=True
)
```

Although the maximum vocabulary size is configured as 100,000, the fitted vectorizer produces **27,489 actual features** for the training data.

The TF-IDF matrices are converted to `float32` dense NumPy arrays before being passed to PyTorch.

## Neural Network Architecture

```text
TF-IDF
27,489 features
      |
      v
Linear
27,489 → 256
      |
     ReLU
      |
Dropout = 0.30
      |
      v
Linear
256 → 64
      |
     ReLU
      |
Dropout = 0.20
      |
      v
Linear
64 → 5
      |
      v
Five class logits
      |
      v
Top-3 ranking
```

The network is implemented using PyTorch.

## Training Configuration

| Parameter      | Value            |
| -------------- | ---------------- |
| Input features | 27,489           |
| Hidden layer 1 | 256              |
| Hidden layer 2 | 64               |
| Output classes | 5                |
| Activation     | ReLU             |
| Dropout 1      | 0.30             |
| Dropout 2      | 0.20             |
| Loss           | CrossEntropyLoss |
| Optimizer      | Adam             |
| Learning rate  | 1e-3             |
| Weight decay   | 1e-4             |
| Batch size     | 32               |
| Epochs         | 30               |

The best checkpoint is selected using validation MAP@3.

---

# Model 2 — BERT Multiple Choice

The second model is a pretrained **BERT-base-uncased** multiple-choice model.

## Model

```text
bert-base-uncased
```

The model is loaded using:

```python
AutoModelForMultipleChoice
```

## MCQ Representation

Each question is converted into five prompt-option pairs:

```text
Prompt + Option A
Prompt + Option B
Prompt + Option C
Prompt + Option D
Prompt + Option E
```

The resulting representation has the form:

```text
batch × choices × sequence_length
```

For this project:

```text
choices = 5
sequence_length = 128
```

Therefore, one MCQ is represented as:

```text
[5, 128]
```

## Tokenization

BERT uses its pretrained WordPiece tokenizer:

```python
AutoTokenizer.from_pretrained("bert-base-uncased")
```

Configuration:

```text
max_length = 128
truncation = True
padding = "max_length"
```

## Training Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Model                   | bert-base-uncased |
| Maximum sequence length | 128               |
| Learning rate           | 2e-5              |
| Weight decay            | 0.01              |
| Train batch size        | 16                |
| Evaluation batch size   | 16                |
| Epochs                  | 15                |
| Seed                    | 42                |
| Data seed               | 42                |
| Best metric             | MAP@3             |

Training is performed using the Hugging Face `Trainer` API.

Weights & Biases is used to track the experiment.

---

# Model 3 — RoBERTa Multiple Choice

The third and final model is **RoBERTa-base**.

## Model

```text
roberta-base
```

The model is loaded using:

```python
AutoModelForMultipleChoice
```

## Tokenization

RoBERTa uses its byte-level BPE tokenizer:

```python
AutoTokenizer.from_pretrained("roberta-base")
```

Each MCQ is again represented as five prompt-option sequences:

```text
Prompt + Option A
Prompt + Option B
Prompt + Option C
Prompt + Option D
Prompt + Option E
```

Each sequence is truncated/padded to:

```text
128 tokens
```

Therefore, one MCQ has the shape:

```text
[5, 128]
```

## Training Configuration

| Parameter               | Value                      |
| ----------------------- | -------------------------- |
| Model                   | roberta-base               |
| Task                    | 5-class MCQ classification |
| Maximum sequence length | 128                        |
| Learning rate           | 1.5e-5                     |
| Weight decay            | 0.01                       |
| Train batch size        | 16                         |
| Evaluation batch size   | 16                         |
| Epochs                  | 20                         |
| Seed                    | 42                         |
| Data seed               | 42                         |
| Best metric             | MAP@3                      |

Training is performed using the Hugging Face `Trainer` API and experiment metrics are logged to Weights & Biases.

---

# Evaluation

The following metrics are used:

* Accuracy
* Macro F1
* MAP@3

## MAP@3

MAP@3 is the primary competition metric.

For each question, only the top three predictions are considered.

If the correct answer is:

```text
Rank 1 → score = 1
Rank 2 → score = 1/2
Rank 3 → score = 1/3
Not in Top-3 → score = 0
```

The final MAP@3 is the mean of these scores across all evaluated questions.

---

# Final Validation Results

The final notebook compares the three model families using the same validation split.

| Model                   | Accuracy | Macro F1 |    MAP@3 |
| ----------------------- | -------: | -------: | -------: |
| TF-IDF + Neural Network | 1.000000 | 1.000000 | 1.000000 |
| RoBERTa-base            | 1.000000 | 1.000000 | 1.000000 |
| BERT-base               | 0.997159 | 0.997019 | 0.998580 |

These are **validation results**, not Kaggle leaderboard scores.

The final notebook explicitly selects:

```text
RoBERTa-base
```

as the final model.

---

# Final Kaggle Submission

The final inference pipeline generates three model-specific submission files:

```text
submission_model1.csv
submission_model2.csv
submission_model3.csv
```

Corresponding to:

```text
submission_model1.csv → TF-IDF + Neural Network
submission_model2.csv → BERT-base
submission_model3.csv → RoBERTa-base
```

The final model selection logic selects:

```text
RoBERTa-base
```

and copies its submission to:

```text
submission.csv
```

The submission follows the required Kaggle format:

```text
id,prediction
```

where `prediction` contains the three ranked answer choices separated by spaces.

Example:

```text
A C E
```

---

# Kaggle Result

The best recorded Kaggle submission achieved:

| Metric           |           Result |
| ---------------- | ---------------: |
| Best MAP@3       |      **0.76599** |
| Leaderboard Rank |           **51** |
| Submission       | `submission.csv` |

The Kaggle leaderboard score is kept separate from the validation metrics reported above.

---

# Model Comparison

The project demonstrates the progression from sparse lexical features to contextual Transformer representations:

```text
TF-IDF
  |
  v
Sparse lexical representation
  |
  v
TF-IDF + Neural Network
  |
  v
Nonlinear learned representation
  |
  v
BERT
  |
  v
Contextual Transformer representation
  |
  v
RoBERTa
  |
  v
Final selected model
```

The TF-IDF model provides a computationally simple baseline and scratch deep learning approach.

BERT and RoBERTa provide contextual representations through Transformer self-attention and are formulated naturally as multiple-choice classification models.

---

# Experiment Tracking

Weights & Biases is used for experiment tracking.

Project:

```text
DL-23f3004113-notebook-t22026
```

The three principal runs are:

```text
TFIDF-NeuralNetwork
BERT-FineTuning
RoBERTa-FineTuning
```

The experiments track common evaluation metrics including:

* Accuracy
* Macro F1
* MAP@3

---

# Error Analysis

The final notebook includes:

* Classification reports
* Confusion matrices
* Incorrect prediction inspection
* Validation prediction analysis

For incorrect predictions, the prompt, candidate options, ground-truth answer and predicted answer can be inspected.

This allows errors to be examined in terms of:

* Lexical overlap
* Contextual understanding
* Incorrect option ranking
* Model-specific prediction behaviour

---

# Deployment

A RoBERTa-based Smart MCQ Solver was also deployed using Hugging Face Spaces.

The application accepts:

* A question
* Five answer options

and returns the predicted answer and top-3 probabilities.

Deployment: 
https://ayushsur2003-smart-mcq-roberta-demo.hf.space/?__theme=system&deep_link=H6W6tcSqHg4


```text
Hugging Face Space:
ayushsur2003/smart-mcq-roberta-demo
```

---

# Repository Structure

The repository follows a modular structure separating datasets, notebooks, source code, model-related prediction artifacts and reports.

```text
Smart_MCQ_Solver_Challenge/
│
├── data/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── models/
│   ├── .gitkeep
│   ├── bert_pretrained_submission.csv
│   ├── roberta_submission.csv
│   └── tfidf_nn_predictions.csv
│
├── notebooks/
│   ├── Final_Notebook.ipynb
│   ├── Milestone-1.ipynb
│   ├── Milestone-2.ipynb
│   ├── Milestone-3.ipynb
│   ├── Milestone-4.ipynb
│   └── Milestone-5.ipynb
│
├── reports/
│   ├── 23f3004113_DG_T22026.pdf
│   └── model_comparison.md
│
├── src/
│   ├── Roberta_model.py
│   ├── bert_model.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── tfidf_baseline.py
│   ├── tfidf_nn.py
│   └── utils.py
│
├── README.md
└── requirements.txt
```

### Important Note

The `models/` directory currently contains prediction/submission artifacts rather than the full BERT/RoBERTa checkpoint directories.

The final notebook trains and saves model checkpoints during execution, while the repository keeps the code and generated prediction artifacts required for the project submission.

---

# Source Code Organization

## `src/preprocessing.py`

Contains reusable preprocessing functionality used to prepare the MCQ data.

## `src/tfidf_baseline.py`

Contains the classical TF-IDF baseline implementation.

This is retained as a baseline/reference approach and is **not the final neural model**.

## `src/tfidf_nn.py`

Contains the custom TF-IDF + Neural Network architecture.

The model follows:

```text
27,489 → 256 → 64 → 5
```

## `src/bert_model.py`

Contains BERT multiple-choice model loading and related functionality.

## `src/Roberta_model.py`

Contains RoBERTa model loading and related functionality.

## `src/inference.py`

Contains inference and submission-generation functionality.

## `src/utils.py`

Contains reusable utility functions, including evaluation-related functionality.

---

# Technologies Used

## Programming Language

* Python

## Machine Learning / Deep Learning

* PyTorch
* scikit-learn
* NumPy
* Pandas

## NLP / Transformers

* Hugging Face Transformers
* Hugging Face Datasets
* Tokenizers

## Experiment Tracking

* Weights & Biases

## Development Environment

* Kaggle Notebooks
* Jupyter Notebook
* GitHub

## Deployment

* Hugging Face Spaces

---

# Key Learnings

This project provided practical experience with:

* NLP preprocessing
* TF-IDF representations
* Sparse feature spaces
* Feed-forward neural networks
* PyTorch training loops
* Transformer architectures
* BERT and RoBERTa tokenization
* Multiple-choice Transformer modeling
* Hugging Face `Trainer`
* Model evaluation
* MAP@3 ranking
* Experiment tracking with Weights & Biases
* Kaggle inference and submission generation
* Model deployment

The project demonstrates the progression from traditional lexical methods to neural and contextual Transformer-based approaches.

---

# Future Work

Potential improvements include:

* Better hyperparameter optimization
* Calibration of model probabilities
* More extensive error analysis
* Ensemble methods
* Retrieval-Augmented Generation
* Instruction-based approaches
* Larger Transformer architectures
* Improved ranking strategies

These are considered future improvements and are **not part of the final submitted pipeline**.

---

# Author

**Ayush Sur**

IIT Madras BS Degree in Data Science and Applications

Roll Number: `23f3004113`

Kaggle ID: `ayushsur2003`
