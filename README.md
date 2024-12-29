# Delhi House Prediction

This repository contains a Jupyter Notebook designed to predict housing-related outcomes for properties in Delhi. The notebook includes data preprocessing, exploratory data analysis, and machine learning model evaluation, with a focus on using the **XGBoost Classifier** for optimal predictions.

---

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical features, and normalizing numerical features.
- **Exploratory Data Analysis (EDA)**: Visualization of key trends and insights in the dataset.
- **Machine Learning Models**:
  - Multiple algorithms tested (e.g., Logistic Regression, Random Forest, etc.)
  - XGBoost Classifier identified as the best-performing model.
- **Evaluation Metrics**: Performance assessed using metrics such as accuracy, precision, recall, F1 score, and R².

---

## Files

- `delhi-house-prediction.ipynb`: Main notebook containing the code and outputs for data analysis and model predictions.

---

## Prerequisites

- Python 3.10 or later
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

To install the necessary libraries, run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Dataset

The notebook uses a dataset sourced from Kaggle. Ensure you download the required dataset from the linked Kaggle page before running the notebook.

---

## Usage

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Open the notebook in Jupyter:
   ```bash
   jupyter notebook delhi-house-prediction.ipynb
   ```
3. Follow the instructions in the notebook to execute each cell sequentially.

---

## Results

- The XGBoost Classifier achieved the highest accuracy on the test data, providing a 99% R² score.
- Detailed evaluation metrics and confusion matrices are included in the notebook.

---

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## Acknowledgments

- The dataset is sourced from Kaggle.
- Special thanks to the authors of the libraries used in this project.

