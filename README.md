# [Project Title: SMS spam Text Classification with Naive Bayes]

## Overview
This project implements a Natural Language Processing (NLP) pipeline to classify text data. By leveraging natural language toolkits and machine learning algorithms, the model processes raw text, extracts meaningful features, and categorizes the data into predefined labels. 

The core classification engine uses a **Multinomial Naive Bayes** model, which is particularly effective for text-based features like word counts or term frequencies.

## Features
* **Text Preprocessing:** Cleans and prepares raw text data using `nltk` and standard string operations (tokenization, punctuation removal).
* **Feature Extraction:** Converts text into a matrix of token counts using `CountVectorizer`.
* **Predictive Modeling:** Trains a Multinomial Naive Bayes classifier (`MultinomialNB`) to categorize text.
* **Model Evaluation:** Analyzes model performance comprehensively using Accuracy, Precision, Recall, F1-Score, and Cross-Validation.
* **Data Visualization:** Generates visual reports including Confusion Matrices, Precision-Recall curves, and ROC curves using `matplotlib` and `seaborn`.
* **Model Export:** Saves the trained model using `joblib` for future deployment or inference without retraining.

## Technologies Used
* **Language:** Python 3.x
* **Data Manipulation:** `numpy`, `pandas`
* **Natural Language Processing:** `nltk`, `string`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`

## Getting Started

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment. You will need to install the required libraries:

```bash
pip install numpy pandas nltk matplotlib seaborn scikit-learn joblib
