# Loan Approval Prediction

This notebook explores a loan approval prediction dataset, performs exploratory data analysis, preprocesses the data, and trains various machine learning models to predict loan approval status.

## Table of Contents

1.  [Dataset Overview](#dataset-overview)
2.  [Exploratory Data Analysis](#exploratory-data-analysis)
    *   [Data Information](#data-information)
    *   [Summary Statistics](#summary-statistics)
    *   [Categorical Feature Visualization](#categorical-feature-visualization)
    *   [Correlation Heatmap](#correlation-heatmap)
3.  [Data Preprocessing](#data-preprocessing)
    *   [Handling Categorical Features](#handling-categorical-features)
    *   [Handling Missing Values](#handling-missing-values)
    *   [Feature Scaling](#feature-scaling)
4.  [Model Training and Evaluation](#model-training-and-evaluation)
    *   [Model Comparison (Cross-Validation)](#model-comparison-cross-validation)
    *   [Model Performance on Test Set](#model-performance-on-test-set)
5.  [Conclusions and Recommendations](#conclusions-and-recommendations)

## Dataset Overview

The dataset contains information about loan applications, including demographic details, income, loan amount, credit history, and the final loan status (approved or rejected). The goal is to build a model that can predict the `Loan_Status` based on the other features.

The dataset contains **598 entries (rows)** and **13 columns (features)**.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is a crucial step to understand the characteristics of the data, identify patterns, and uncover potential issues before building a predictive model.

### Data Information

To get a concise summary of the dataset, we used the `.info()` method. This provided the following key information:

*   **Number of Entries:** 598 loan applications.
*   **Number of Columns:** 13 different pieces of information for each application.
*   **Column Names and Data Types:** We saw columns like `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and `Loan_Status`. The data types included `object` (for text/categorical data), `int64` (for whole numbers), and `float64` (for numbers with decimals).
*   **Non-Null Counts:** This showed us that some columns had missing values, specifically `Dependents` (586 non-null), `LoanAmount` (577 non-null), `Loan_Amount_Term` (584 non-null), and `Credit_History` (549 non-null). Handling these missing values is necessary for model training.

### Summary Statistics

The `.describe()` method was used to calculate and display summary statistics for the numerical columns. This gave us insights into the central tendency, dispersion, and shape of the distribution of numerical features:

*   **count:** The number of non-null values.
*   **mean:** The average value.
*   **std:** The standard deviation, which measures the spread of the data.
*   **min:** The minimum value.
*   **25%, 50% (median), 75%:** The quartiles, which divide the data into four equal parts.
*   **max:** The maximum value.

From this, we observed ranges and typical values for features like `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount`.

### Categorical Feature Visualization

To understand the distribution of categorical features, we generated bar plots for `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, and `Loan_Status`. These plots visually represented the frequency of each unique value within these columns.

Key insights from these plots included:

*   The majority of applicants were **Male** and **Married**.
*   Most applicants had a **Graduate** education.
*   A large proportion of applicants were **Not Self-Employed**.
*   The `Property_Area` was relatively evenly distributed between **Semiurban**, **Urban**, and **Rural**.
*   The `Loan_Status` showed an imbalance, with more loans being **Approved (Y)** than **Rejected (N)**.

We also created a categorical plot to visualize the relationship between `Gender`, `Married`, and `Loan_Status`. This plot helped us see how the loan approval rate varied depending on the combination of gender and marital status. It visually suggested potential differences in approval rates across these groups.

### Correlation Heatmap

A correlation heatmap was generated to visualize the pairwise correlations between the numerical features. The heatmap uses color intensity to represent the strength and direction of the correlation coefficient (ranging from -1 to +1).

*   Values close to +1 indicate a strong positive linear correlation (as one feature increases, the other tends to increase).
*   Values close to -1 indicate a strong negative linear correlation (as one feature increases, the other tends to decrease).
*   Values close to 0 indicate a weak or no linear correlation.

The heatmap revealed the relationships between numerical features such as `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`. For example, we could observe the correlation between `ApplicantIncome` and `LoanAmount`.

## Data Preprocessing

Before training machine learning models, the data needs to be preprocessed to handle non-numerical data, missing values, and ensure features are on a similar scale.

### Handling Categorical Features

Machine learning models typically require numerical input. We used **Label Encoding** to convert the categorical features (`Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`) into numerical representations. Each unique category within a feature was assigned a unique integer value.

### Handling Missing Values

As identified during the EDA, several columns had missing values. We addressed this by **imputing** the missing values with the **mean** of the respective columns. This is a simple but effective strategy to fill in missing data points.

### Feature Scaling

Some machine learning algorithms are sensitive to the scale of the input features. To ensure that no single feature dominates the learning process due to its larger magnitude, we applied **Standard Scaling** to the numerical features. Standard Scaling transforms the data to have a mean of 0 and a standard deviation of 1.

## Model Training and Evaluation

After preprocessing, we trained and evaluated several different machine learning models to find the best one for predicting loan approval.

### Model Comparison (Cross-Validation)

We used **5-fold cross-validation** on the training data to get a more robust estimate of each model's performance before evaluating on the unseen test set. Cross-validation splits the training data into 5 subsets (folds), trains the model on 4 folds, and evaluates it on the remaining fold. This process is repeated 5 times, with each fold used as the evaluation set once. The mean accuracy across these 5 runs gives a better indication of the model's generalization ability.

A bar plot was generated to visualize the mean accuracy of each model during cross-validation. This provided an initial ranking of the models based on their performance on the training data.

### Model Performance on Test Set

Finally, each model was trained on the entire training dataset and evaluated on the completely unseen test dataset. The following metrics were calculated to provide a comprehensive understanding of each model's performance:

*   **Accuracy:** The proportion of correct predictions (both approved and rejected) out of the total number of predictions.
*   **Confusion Matrix:** A 2x2 table summarizing the prediction results:
    *   **True Positives (TP):** Actual Approved loans correctly predicted as Approved.
    *   **False Positives (FP):** Actual Rejected loans incorrectly predicted as Approved. (Type I error)
    *   **False Negatives (FN):** Actual Approved loans incorrectly predicted as Rejected. (Type II error)
    *   **True Negatives (TN):** Actual Rejected loans correctly predicted as Rejected.

*   **Classification Report:** Provides detailed metrics for each class (Approved and Rejected):
    *   **Precision:** Out of all instances predicted as Approved, what percentage were actually Approved? (TP / (TP + FP))
    *   **Recall (Sensitivity):** Out of all actual Approved instances, what percentage were correctly predicted as Approved? (TP / (TP + FN))
    *   **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure.
    *   **Support:** The number of actual instances in each class in the test set.

The evaluation results on the test set showed that:

*   **Logistic Regression** achieved an accuracy of **84.17%**. Its confusion matrix `[[19 19], [ 0 82]]` showed **0 False Negatives**, meaning it correctly identified all 82 actual approved loans in the test set. However, it had 19 False Positives. The classification report confirmed a high recall for the approved class (1.00).
*   **Naive Bayes** achieved an accuracy of **83.33%**. Similar to Logistic Regression, its confusion matrix `[[18 20], [ 0 82]]` also showed **0 False Negatives**, demonstrating excellent recall for approved loans. It had 20 False Positives.
*   Other models like K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and Gradient Boosting had slightly lower accuracies and varying numbers of False Positives and False Negatives.

## Conclusions and Recommendations

Based on the detailed evaluation metrics, **Logistic Regression** and **Naive Bayes** emerged as the top-performing models on this dataset, both achieving high accuracies and, notably, correctly identifying all actual approved loans in the test set (0 False Negatives). This makes them strong candidates if the priority is to minimize missing potential loan approvals. However, it's important to consider their False Positives, as these represent loans incorrectly predicted as approved.

For future work, the following steps are recommended:

*   **Hyperparameter Tuning:** Fine-tuning the parameters of Logistic Regression and Naive Bayes could potentially improve their performance further.
*   **Feature Engineering:** Creating new features or transforming existing ones might capture more complex relationships and enhance model accuracy.
*   **Exploring Other Models:** Investigating other classification algorithms that might be suitable for this dataset.
*   **Addressing Class Imbalance:** Given the imbalance in the `Loan_Status` (more Approved than Rejected loans), techniques like oversampling the minority class or using evaluation metrics less sensitive to imbalance (e.g., F1-score) could be further explored.
*   **Model Interpretability:** For practical application, understanding *why* a loan is predicted to be approved or rejected is valuable. Techniques for model interpretability can provide insights into the factors driving the predictions.
