# Credit Default Prediction with Numerical Machine Learning

This project compares three approaches for predicting credit default:

- Regularized Logistic Regression
- Nyström-approximated kernel SVM
- Multilayer Perceptron implemented with JAX

The analysis focuses on the numerical effects of feature scaling, regularization, kernel approximation, optimization, probability calibration, and cost-sensitive decision thresholds.

## Dataset

The project uses the **Give Me Some Credit** dataset. Download `cs-training.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit) and place it inside:

```text
GiveMeSomeCredit/cs-training.csv
```

## Running the project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open `credit_default_analysis.ipynb` and run all cells.

## Evaluation

Models are evaluated using stratified train, validation, and test splits. The main metrics include average precision, ROC-AUC, Brier score, recall, F1 score, and cost-sensitive classification error.

The complete methodology and discussion are available in `report.pdf`.

## Author

Developed by **Leonardo Pelorosso** for the Numerical Analysis for Machine Learning course.
