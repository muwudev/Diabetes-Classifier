
# Random Forests and Ensemble Methods for Diabetes Prediction

This repository contains two Google Colab notebooks that explore the prediction of diabetes on the PIMA Indians Diabetes Dataset using machine learning models. The notebooks focus on data preprocessing, feature engineering, and model evaluation using various algorithms such as SVM, Random Forests and ensemble methods

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


## Notebooks

### 1. **Exploratory Data Analysis**
   - **File Name**: `Diabetes_Prediction_EDA.ipynb`
   - **Description**: This notebook focuses on exploring the PIMA dataset with visualisations. It includes data preprocessing, feature engineering, and model evaluation

### 2. **Diabetes Prediction with Ensemble Methods**
   - **File Name**: `Diabetes_Prediction_Ensemble_Methods.ipynb`
   - **Description**: This notebook explores the use of ensemble methods (e.g., Bagging, Voting, Stacking) for diabetes prediction. It also includes advanced feature engineering and model comparison


## Project Structure

```
diabetes-prediction/
├── Diabetes_Prediction_SVM_RandomForests.ipynb       # Notebook for SVM and Random Forests
├── Diabetes_Prediction_Ensemble_Methods.ipynb        # Notebook for Ensemble Methods
├── diabetes.csv                                      # Dataset used for analysis
├── README.md                                         # Project documentation
```


## Key Features

### Data Preprocessing
- Handling missing values by replacing zeros with median/mean
- Feature engineering to create new categorical features (e.g., BMI categories, glucose levels)
- One-hot encoding for categorical variables

### Model Training and Evaluation
- **Baseline Models**: SVM, KNN, Decision Tree, Random Forests
- **Ensemble Methods**: Bagging, Voting, Stacking
- Evaluation metrics: Accuracy, Confusion Matrix, ROC Curve

### Visualizations
- Density plots for feature distributions
- Correlation heatmaps (Pearson, Spearman, Kendall)
- ROC curves for model comparison


## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   ```
2. **Open in Google Colab**:
   - Upload the notebooks to Google Colab.
   - Ensure the dataset (`diabetes.csv`) is uploaded to Google Drive and mounted in Colab

3. **Run the Notebooks**:
   - Execute the cells in each notebook to preprocess data, train models, and evaluate performance


## Results

### Baseline Models
- **Decision Tree**: Accuracy = 0.6277
- **Random Forest**: Accuracy = 0.7316
- **SVM**: Accuracy = 0.7143
- **KNN**: Accuracy = 0.6840

### Ensemble Methods
- **Bagging with Decision Trees**: Accuracy = 0.7359
- **Bagging with KNN**: Accuracy = 0.6797
- **Voting Classifier**: Accuracy = 0.6970
- **Stacking Classifier**: Accuracy = 0.7143

### ROC Curves
- Baseline models and ensemble methods are compared using ROC curves to evaluate their performance


## Future Improvements
- Experiment with hyperparameter tuning for better model performance
- Explore additional ensemble techniques (e.g., Boosting)
- Add more visualizations for better insights into the data and model performance


- Libraries: `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`


Feel free to explore the notebooks and adapt them for your own use! If you have any questions or suggestions, dm me on discord :3
