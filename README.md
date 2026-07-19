# Smart MCQ Solver Challenge

## Student Details

- **Name:** Ayush Sur
- **Roll Number:** 23f3004113
- **Program:** IIT Madras BS Degree in Data Science and Applications
- **Course:** Introduction to Deep Learning and Generative AI Project (May Term 2026 - T2 Term)

---

## Project Overview

This repository contains the implementation of the **Smart MCQ Solver Challenge**, a Deep Learning and Generative AI project conducted as part of the IIT Madras BS Degree program.

The objective of the project is to build intelligent systems capable of predicting and ranking the most probable answers for multiple-choice questions using Natural Language Processing (NLP), Information Retrieval, Transformer-based architectures, and parameter-efficient fine-tuning techniques.

The evaluation metric used in the competition is **Mean Average Precision at 3 (MAP@3)**.

---

## Repository Structure

```text
Smart_MCQ_Solver_Challenge/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── milestone1.ipynb
│   ├── milestone2.ipynb
│   ├── milestone3.ipynb
│   └── milestone4.ipynb
│
├── src/
│
├── models/
│
├── reports/
│
├── README.md
│
└── requirements.txt
```

---

## Milestone Progress

### Milestone 1 (Completed)

#### Topics Covered

- Frequency Distribution Analysis
- Text Cleaning and Preprocessing
- Stop Word Removal
- Vocabulary Size Computation
- TF-IDF Vectorization
- Cosine Similarity
- MAP@3 Evaluation Metric
- Majority Class Baseline
- TF-IDF Similarity Ranking Pipeline

#### Tasks Completed

- Calculated answer distribution statistics
- Computed vocabulary size after preprocessing
- Generated TF-IDF feature space
- Measured cosine similarity between prompts and answer options
- Implemented MAP@3 calculations
- Developed Majority Class Baseline
- Built a TF-IDF-based ranking pipeline
- Submitted Milestone 1 responses

**Status:** Completed

---

### Milestone 2 (Completed)

#### Topics Covered

- Hugging Face Datasets
- Hugging Face Transformers
- Sentence Transformers (MiniLM)
- Dense Vector Retrieval
- Zero-shot Classification
- FLAN-T5 Small Language Model
- Prompt Engineering
- Modular Python Package Design

#### Tasks Completed

- Loaded datasets using the Hugging Face Datasets library
- Created combined text features using the `.map()` function
- Computed dataset statistics and text lengths
- Implemented semantic retrieval using MiniLM sentence embeddings
- Ranked answer options using cosine similarity
- Compared TF-IDF and MiniLM retrieval performance
- Implemented Zero-shot Classification using `facebook/bart-large-mnli`
- Compared Softmax and Independent Sigmoid (`multi_label=True`) scoring
- Loaded and used `google/flan-t5-small`
- Generated answers through prompt engineering
- Refactored reusable code into modular Python files
- Organized the project into reusable components
- Submitted Milestone 2 responses

**Status:** Completed

---

### Milestone 3 (Completed)

#### Topics Covered

- Transformer Architectures
- Self-Attention Mechanism
- Tokenization Strategies
- BERT Embeddings
- Multiple Choice Question Modeling
- Hugging Face Tokenizers

#### Tasks Completed

- Explored transformer-based architectures for MCQ solving
- Implemented tokenization pipelines for multiple-choice inputs
- Constructed formatted prompt-option pairs
- Performed tensor reshaping for multiple-choice models
- Analyzed model input representations
- Prepared data pipelines for transformer-based inference

**Status:** Completed

---

### Milestone 4 (Completed)

#### Topics Covered

- AutoModelForMultipleChoice
- Multiple Choice Tokenization
- LoRA (Low-Rank Adaptation)
- Parameter-Efficient Fine-Tuning (PEFT)
- Hugging Face Trainer API
- Tiny LoRA Fine-Tuning
- Softmax-Based Inference
- Multiple Choice Classification with BERT

#### Tasks Completed

- Encoded answer labels for multiple-choice classification
- Generated tokenized inputs for five answer options
- Prepared batch tensors for multiple-choice transformer models
- Performed forward passes using `AutoModelForMultipleChoice`
- Computed supervised loss tensors
- Applied LoRA adapters to the BERT multiple-choice model
- Calculated trainable LoRA parameters
- Prepared Hugging Face datasets for training
- Fine-tuned the model using Hugging Face Trainer
- Performed inference on fine-tuned models
- Converted logits into probabilities using Softmax
- Predicted answer probabilities for multiple-choice options

**Status:** Completed

---

## Technologies Used

### Programming Language

- Python

### Libraries and Frameworks

- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Datasets
- Hugging Face Transformers
- Sentence Transformers
- PEFT (LoRA)
- Accelerate
- Datasets
- Evaluate

### Deep Learning Models

- BERT Base Uncased
- MiniLM
- FLAN-T5 Small
- BART Large MNLI
- LoRA Fine-Tuned Multiple Choice Models

### Development Environment

- Kaggle Notebooks
- GitHub
- Jupyter Notebook

---

## Evaluation Metric

The competition uses **Mean Average Precision at 3 (MAP@3)**.

For each question, models predict the top three most probable answer options. Higher scores are awarded when the correct answer appears earlier in the ranked predictions.

---

## Kaggle Competition

**Competition:** Smart MCQ Solver Challenge

The project is developed and evaluated using the Kaggle competition environment provided as part of the IIT Madras BS Degree program.

---

## Future Work

Future enhancements of this project may include:

- Retrieval-Augmented Generation (RAG)
- Larger Transformer Models
- Ensemble-Based Ranking Models
- Advanced Parameter-Efficient Fine-Tuning Techniques
- Improved Multiple-Choice Reasoning Models
- Hybrid Retrieval and Generation Pipelines
- Competition Score Optimization Strategies

---

## Author

**Ayush Sur**

- IIT Madras BS Degree in Data Science and Applications
- Roll Number: 23f3004113
