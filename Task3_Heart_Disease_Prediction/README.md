# Heart Disease Prediction

## Task Objective
- Build a binary classification model to predict whether a patient is at risk of heart disease from clinical health measurements.
- Clean the data, explore patterns with EDA, train classification models, evaluate performance, and explain the main predictors.

## Dataset Used
- Dataset: UCI Heart Disease (processed Cleveland file)
- Source: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- Target definition: the original `num` column is converted to a binary label where `0` means no heart disease and values greater than `0` mean heart disease is present.

## Files In This Task
- `Heart_Disease_Prediction.ipynb`: complete notebook for preprocessing, EDA, model training, evaluation, and interpretation
- `data/`: local storage location for the downloaded dataset when the notebook is run

## Models Applied
- Logistic Regression
- Decision Tree Classifier

## Notebook Coverage
- Clear problem statement and objective
- Dataset loading with local-file fallback and public URL download
- Missing value handling using median imputation
- Exploratory visualizations for class balance, numeric distributions, and correlations
- Model comparison using accuracy and ROC-AUC
- ROC curve plotting and confusion matrix visualization
- Feature importance analysis using logistic regression coefficients and decision tree importances

## Key Results And Findings
- The notebook compares Logistic Regression and Decision Tree on the same train/test split.
- Performance is summarized with accuracy, ROC-AUC, ROC curve, and confusion matrix.
- Important predictors are ranked from both models to highlight the clinical variables that most influence prediction.
- Missing values from the raw UCI file are handled explicitly before training.

## How To Run
1. Open `Heart_Disease_Prediction.ipynb` in Jupyter Notebook or VS Code.
2. Run the cells from top to bottom.
3. The notebook installs any missing Python packages in the first code cell if needed.
4. On first run, the dataset is downloaded and saved into the `data/` folder automatically.

## Suggested GitHub Repository Structure
- `README.md`
- `Heart_Disease_Prediction.ipynb`
- `data/` (optional to keep local only)

## Skills Demonstrated
- Binary classification
- Medical data preprocessing and interpretation
- ROC-AUC and confusion matrix evaluation
- Feature importance analysis
