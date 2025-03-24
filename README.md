# Loan Prediction Model

This project uses machine learning to predict loan eligibility based on applicant data.

## Project Description

This project utilizes a dataset from Kaggle containing information about loan applicants, including their demographics, financial history, and loan status.  The goal is to build a model that can accurately predict whether a new applicant will be eligible for a loan.

## Dataset

The dataset used in this project can be found on Kaggle: [Loan Predication Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)

The dataset contains the following features:

* **Loan_ID:** Unique identifier for each loan application.
* **Gender:** Applicant's gender.
* **Married:** Applicant's marital status.
* **Dependents:** Number of people dependent on the applicant.
* **Education:** Applicant's educational qualification.
* **Self_Employed:** Whether the applicant is self-employed.
* **ApplicantIncome:** Applicant's income.
* **CoapplicantIncome:** Co-applicant's income.
* **LoanAmount:** Loan amount in thousands.
* **Loan_Amount_Term:** Term of loan in months.
* **Credit_History:** Credit history meets guidelines.
* **Property_Area:** Urban, Semi Urban, or Rural.
* **Loan_Status:** Loan approved (Y/N).


## Methodology

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using the Pandas library.
   - Missing values are handled by dropping rows with missing data.
   - Categorical features are encoded into numerical values.

2. **Data Visualization:**
   - Seaborn and Matplotlib are used to visualize the relationship between features and loan status.

3. **Model Training and Evaluation:**
   - The dataset is split into training and testing sets.
   - A Support Vector Machine (SVM) classifier with a linear kernel is used for prediction.
   - The model is trained on the training data.
   - The model's accuracy is evaluated on the testing data using `accuracy_score`.

## Results

The model achieved an accuracy of approximately 80%. This indicates that the model can predict loan eligibility with reasonable accuracy, but there is room for improvement.

## Dependencies

The following libraries are required to run this project:

- pandas
- seaborn
- matplotlib
- scikit-learn (sklearn)
- IPython

You can install them using pip:
