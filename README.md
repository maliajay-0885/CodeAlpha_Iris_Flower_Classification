# CodeAlpha Internship — Iris Flower Classification

## Project Overview

This project uses machine learning to classify Iris flowers into three species based on their sepal and petal measurements.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Visual Studio Code

## Dataset

The project uses `Iris.csv`, which contains flower measurements and species labels.

The four features used for classification are:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Algorithm

Random Forest Classifier.

## Steps Performed

1. Loaded the Iris dataset using Pandas.
2. Checked the dataset structure and missing values.
3. Selected the four measurement features.
4. Split the dataset into training and testing sets.
5. Trained the Random Forest classification model.
6. Evaluated the model using accuracy and a classification report.
7. Generated a confusion matrix and feature-importance graph.
8. Predicted the species of a new flower using its measurements.

## Results

The model's accuracy and classification report are displayed when the Python program runs.

The program also generates and saves:
- `confusion_matrix.png` — shows the model's classification results.
- `feature_importance.png` — shows the relative importance of the input features.

## Installation

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

Make sure `Iris.csv` and `iris_classification.py` are in the same folder.

Run the program using:

```bash
python iris_classification.py
```

## Conclusion

This project demonstrates supervised machine learning, classification, model evaluation, and data visualization using Python and Scikit-learn.