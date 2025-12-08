If you use this classifier to help your research, please cite the following article:

Liang-Ting Wu, Milica Zdravković, Dijana Milosavljević, Konstantin Köster, Olivier Guillon, Jyh-Chiang Jiang, and Payam Kaghazchi*. Prediction of Structural Stability of Layered Oxide Cathode Materials: Combination of Machine Learning and Ab Initio Thermodynamics. _Adv. Energy Mater_., **2025**, Early View. https://doi.org/10.1002/aenm.202505470

# ML Phase Predictor

This repository provides a complete workflow for predicting material phases (P2 or O3) using machine learning, based on ionic parameters computed from input compositions.

It includes tools for feature calculation, training multiple ML models (including a DNN), and a final predictor that uses Monte Carlo (MC) dropout to assess prediction uncertainty.

**Try the live prediction tool on Hugging Face Spaces!** This GUI provides an easy-to-use interface for the trained model, including uncertainty estimation.  
[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/LIANGTING-WU/Phase_Predictor)

---

## Repository Contents & Workflow

This project is broken down into several notebooks that represent a full machine learning pipeline, from data processing to final prediction.

### 1. Feature Calculator
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Feature_Calculator.ipynb)

* **`Feature_Calculator.ipynb`**: This notebook contains the Python code used to process raw chemical formulas and compute the specific ionic-based features required for model training.

### 2. Model Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Model_Training.ipynb)

* **`Model_Training.ipynb`**: This notebook reads the features from `DNN-270-Training.csv` and trains several machine learning models for comparison:
    * Logistic Regression (LR)
    * Support Vector Machine (SVM)
    * Naive Bayes (NB)
    * Random Forest (RF)
    * _k_-Nearest Neighbors (_k_-NN)
    * Deep Neural Network (DNN)
* It also includes feature interpretability analysis using **SHAP**.

### 3. Data Analysis & Visualization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Raincloud_Plot_and_PCA.ipynb)

* **`Raincloud_Plot_and_PCA.ipynb`**: This notebook visualizes the feature distributions of the 270-sample training set and the 80-sample independent test set using **Raincloud plots**. It also performs **Principal Component Analysis (PCA)** to visualize the dataset.

### 4. Phase Predictor (with Uncertainty)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIANGTING-WU/ML_Phase_Predictor/blob/main/Phase_Predictor.ipynb)

* **`Phase_Predictor.ipynb`**: This is the final prediction tool that uses the pre-trained DNN model.
* It performs **Monte Carlo (MC) dropout** with **1,000 forward passes** to calculate not only the predicted probability but also the **uncertainty** (standard deviation) of the prediction.

---

## Datasets

* **`DNN-270-Training.csv`**: Contains the 270 compositions used for training and validating the models.
* **`Full-Dataset-350.csv`**: The complete dataset, which includes the 270 training samples plus an 80-sample independent testing set.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- shap

All dependencies are installed automatically in the provided Colab notebooks.

---

## Usage Example

To use the notebooks (e.g., `Feature_Calculator.ipynb` or `Phase_Predictor.ipynb`), you can prepare your input data in the following format:

```python
# Example format for new compositions
data = [
    [36992, 'P2', 0.69, {'Mn': 0.77, 'Fe': 0.08, 'Mg': 0.15, 'O': 2.0}]
]
