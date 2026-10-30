# Iris Flower Classification using K-Nearest Neighbors (KNN)

## Description

This project demonstrates a machine learning classification workflow using the Iris Flower Dataset. The program trains a K-Nearest Neighbors (KNN) model to classify iris flowers into different species based on their measurements.

The application performs data preprocessing, model training, evaluation, visualisation, and prediction of new flower samples.

## Features

- Loads the Iris Dataset from Scikit-learn
- Creates a Pandas DataFrame for analysis
- Splits data into training and testing sets
- Standardises features using StandardScaler
- Determines the optimal K value using the Elbow Method
- Trains a K-Nearest Neighbors (KNN) classifier
- Evaluates model performance using:
  - Accuracy Score
  - F1 Score
  - Confusion Matrix
  - Classification Report
- Generates visualisations
- Predicts the species of a new flower sample
- Saves outputs to files

## Technologies Used

- Python 3.15.0
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Install the required libraries:

# bash
pip install pandas matplotlib seaborn scikit-learn


## How to Run

1. Open a terminal or command prompt.
2. Navigate to the project folder.

# bash
cd project_folder


3. Run the script:

# bash
python Data_classification.py


## Dataset

The project uses the built-in Iris dataset provided by Scikit-learn.

The dataset contains:

- 150 flower samples
- 3 species:
  - Setosa
  - Versicolor
  - Virginica
- 4 features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

## Output Files

The program automatically generates the following files:

| File | Description |
|--------|------------|
| iris_dataset.csv | Complete Iris dataset |
| elbow_plot.png | K-value optimisation graph |
| confusion_matrix.csv | Confusion matrix data |
| confusion_matrix.png | Confusion matrix heatmap |
| predictions.csv | Actual vs predicted results |
| classification_report.txt | Detailed classification report |
| metrics.txt | Accuracy, F1 Score and best K |

## Example Output

#text
Shape: (150, 6)

Best K: 5

Accuracy: 1.000

F1 Score: 1.000

Predicted species: setosa


## Machine Learning Workflow

1. Load Dataset
2. Create DataFrame
3. Split Dataset
4. Scale Features
5. Find Best K Value
6. Train KNN Model
7. Evaluate Performance
8. Generate Visualisations
9. Predict New Samples

## Project Structure

## text
Project Folder/
│
├── Data_classification.py
├── iris_dataset.csv
├── elbow_plot.png
├── confusion_matrix.csv
├── confusion_matrix.png
├── predictions.csv
├── classification_report.txt
├── metrics.txt
└── README.md


## Learning Objectives

This project demonstrates:

- Data preprocessing
- Feature scaling
- Supervised machine learning
- K-Nearest Neighbors (KNN)
- Model evaluation
- Data visualisation
- Classification problems
- Working with Scikit-learn

## Author

Emmanuel Nkaka

## Future Improvements

- Add additional classification algorithms
- Compare KNN with Decision Trees and Random Forests
- Implement cross-validation
- Create a graphical user interface (GUI)
- Allow users to input their own flower measurements
- Export results to PDF reports

