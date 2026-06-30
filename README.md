# Smart MCQ Solver Challenge

## Student Details

* **Name:** Ayush Sur
* **Roll Number:** 23f3004113
* **Program:** IIT Madras BS Degree in Data Science and Applications
* **Course:** Introduction to Deep Learning and Generative AI Project (May Term 2026(T2 Term))

---

## Project Overview

This repository contains the implementation of the **Smart MCQ Solver Challenge**, a Deep Learning and Generative AI project conducted as part of the IIT Madras BS Degree program.

The objective of the project is to build intelligent systems capable of predicting and ranking the most probable answers for multiple-choice questions using Natural Language Processing (NLP), Information Retrieval, and Deep Learning techniques.

The evaluation metric used in the competition is **Mean Average Precision at 3 (MAP@3)**.

---

## Repository Structure

```text
Smart_MCQ_Solver_Challenge/
│
├── data/
│
├── notebooks/
│   └── milestone1.ipynb
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

Topics Covered:

* Frequency Distribution Analysis
* Text Cleaning and Preprocessing
* Stop Word Removal
* Vocabulary Size Computation
* TF-IDF Vectorization
* Cosine Similarity
* MAP@3 Evaluation Metric
* Majority Class Baseline
* TF-IDF Similarity Ranking Pipeline

Tasks Completed:

* Calculated answer distribution statistics
* Computed vocabulary size after preprocessing
* Generated TF-IDF feature space
* Measured cosine similarity between prompts and answer options
* Implemented MAP@3 calculations
* Developed Majority Class Baseline
* Built a TF-IDF based ranking pipeline
* Submitted Milestone 1 responses

Status: **Completed**

---

### Milestone 2 (Completed)

Topics Covered:

* Hugging Face Datasets
* Hugging Face Transformers
* Sentence Transformers (MiniLM)
* Dense Vector Retrieval
* Zero-shot Classification
* FLAN-T5 Small Language Model
* Prompt Engineering
* Modular Python Package Design

Tasks Completed:

* Loaded datasets using the Hugging Face Datasets library
* Created combined text features using the `.map()` function
* Computed dataset statistics and text lengths
* Implemented semantic retrieval using MiniLM sentence embeddings
* Ranked answer options using cosine similarity
* Compared TF-IDF and MiniLM retrieval performance
* Implemented Zero-shot Classification using `facebook/bart-large-mnli`
* Compared Softmax and Independent Sigmoid (`multi_label=True`) scoring
* Loaded and used `google/flan-t5-small` for answer generation
* Generated answers through prompt engineering
* Refactored reusable code into the `src/` package
* Organized the project into modular components (`tfidf.py`, `minilm.py`, `zero_shot.py`, `flan_t5.py`)
* Submitted Milestone 2 responses

Status: **Completed**

---

## Technologies Used

### Programming Language
* Python

### Libraries and Frameworks
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Datasets
* Hugging Face Transformers
* Sentence Transformers

### Development Environment
* Kaggle Notebooks
* GitHub

---

## Evaluation Metric

The competition uses **Mean Average Precision at 3 (MAP@3)**.

For each question, models predict the top three most probable answer options. Higher scores are awarded when the correct answer appears earlier in the ranked predictions.

---

## Kaggle Competition

**Competition:** Smart MCQ Solver Challenge

The project is developed and evaluated using the Kaggle competition environment provided as part of the course.

---

## Future Work

Upcoming milestones will focus on:

* Transformer Architectures
* Hugging Face Models
* Fine-Tuning Pretrained Models
* Retrieval-Augmented Approaches
* Advanced Ranking Techniques
* Deep Learning Based MCQ Solvers

---

## Author

**Ayush Sur**
IIT Madras BS Degree in Data Science and Applications
Roll Number: 23f3004113
