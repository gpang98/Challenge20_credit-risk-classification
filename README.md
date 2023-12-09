# Challenge20_credit-risk-classification

## Overview

This project aim to build a model that can identify the creditworthiness of borrowers using the dataset of historical lending activity from a peer-to-peer lending services company using supervised machine learning predictive method and make recommendations based on the result of this prediction.

## Objective

The primary goal of this project is:
- To find out whether we can predict whether a loan is 'high-risk' or 'healthy' using logistic regression method.


## Dataset

The CSV file `lending_data.csv` is provided and contain the following field:

	- loan_size
	- interest_rate
	- borrower_income
	- debt_to_income ratio
	- num_of_accounts
	- derogatory_marks
	- total_debts
	- loan_status
    
The `loan_status` column will be used as `target` while the the rest of the columns as `features` in the supervised machine learning prediction.

The target column has 77526 total rows with 75036 label as 0 ('healthy loan') and 2500 label as 1 ('high-risk loan'). So, this show imbalance towards 'healthy loan'. This might influence the prediction result.

## Tools and Libraries
- `Python`: Used for data preprocessing, initial analysis, and visualization.
- `Pandas`: Utilized for data manipulation and analysis.
- `Jupyter Notebook`: Employed as the development environment.
- `sklearn`: module used to do supervised prediction.
- `train_test_split()`: Function to split the original dataset to Train and Test datasets
- `LogisticRegression()`: is the chosen function to do supervised prediction method.
- `confusion_matrix()`: Function to confusion matrix of the prediction.
- `classification_report()`: Function to generate the classification final results.


## Workflow
The following is the workflow employed in performing the logistic regression:

1. Define what is the target prediction and what is the features used to predict. In this case, the target is `loan_status` while the rest of the dataset are used as `features`.
2. Next step is splitting the dataset to test and training sets for both target and features using `train_test_split`.
3. The `logistic Regression Model` is initiated with a selected random_state for repeatability with solver=`lbfgs` used.
4. The training dataset (target and features) are fit into this model using `classifier.fit()` function.
5. Then prediction is made with the test dataset based on this training usign `classifier.predict()` function.
6. Finally, the assessment of the results of the prediction is carried out by running the `confusion_matrix()` function and the resulting `classification_report()` function to view the `precision`, `recall` and `f1-score` for both loan.

## Usage

1. **Setup Environment:**
   - Download Jupyter Notebook so that you can download the uploaded files and view within your local machine.

2. **Educational Purposes:**
   - Feel free to download the uploaded pages so that you can also explore the dataset and gain insights.

## Main Results and Findings.
The results of the prediction is as follows:

| Class           | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| healthy loan    | 1.00      | 0.99   | 1.00     | 18765   |
| high-risk loan  | 0.85      | 0.91   | 0.88     | 619     |
| accuracy        |           |        | 0.99     | 19384   |
| macro avg       | 0.92      | 0.95   | 0.94     | 19384   |
| weighted avg    | 0.99      | 0.99   | 0.99     | 19384   |

Based on the precision, recall and F1-Score, we can conclude that logistic regressions model performs very well for both classes. It predicts 'healthy loan" (label 1) with extremely high precision and recall (99-100%). For 'high-risk loan' (label 0), the model's performance is slightly lower but still very good, with precision around 85% and recall around 91%.

Overall, the logistic regression model demonstrates high accuracy in predicting both 'healthy loans' and 'high-risk loans', and especially exceptional in identifying 'healthy loans'.

We have noted earlier that there are imbalance in the dataset as it has a lot more data for healthy loan than high-risk loan.  Two possible approaches are given below:

1. Resampling Techniques.
- Oversampling.  Increase the number of the samples in minority class by duplicatin samples or generating syntheic examples (eg. using SMOTE - Synthetic Minority Over-sampling Technique).
- Undersampling.  Reduce the number of the majority class by randonly removing samples to balance the classes.

2. Algorithm Approaches.
- Choose a more robust algortihms to imbalanced data such as Random Forest or Gradient Boosting or algorithms desinged specifically for imbalanced datasets (e.g. Balanced Random Forest, SMOTEBoost).

## References

1. Inspired by lectures notes and ChatGPT.
