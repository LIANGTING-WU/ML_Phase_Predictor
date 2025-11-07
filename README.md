# ML Phase Predictor

This repository contains a Google Colab notebook for predicting material phases (P2 or O3) based on ionic parameters computed from input compositions. The prediction is done using a pre-trained Deep Neural Network (DNN) model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Phase_Predictor.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Model_Training.ipynb)

**Try it online on Hugging Face Spaces!**  
[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/LIANGTING-WU/Phase_Predictor)

---

## Features

- Input custom compositions in a specified format
- Compute ionic parameters automatically from composition data
- Normalize input features using saved scaler
- Predict phase label using trained DNN model
- Fully reproducible and easy to use in Google Colab

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- shap

All dependencies are installed automatically in the provided Colab notebook.

---

## Usage Instructions

### 1. Clone or download this repository

You can open the provided Google Colab notebook directly or download it and upload to your Colab.

### 2. Prepare your input data

Input your composition data in the following format inside the notebook:

```python
data = [
    [36992, 'P2', 0.69, {'Mn': 0.77, 'Fe': 0.08, 'Mg': 0.15, 'O': 2.0}]
]
