# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
- The purpose of this analysis is to find out whether we can predict whether a loan is 'high-risk" or 'healthy' using logistic regression method.

* Explain what financial information the data was on, and what you needed to predict.
- The main input as features are loan_size, interest_rate, borrower_income, debt_to_income ratio, num_of_accounts, derogatory_marks and total_debts while the loan_status is used as target.   

* Provide basic information about the variables you were trying to predict (e.g., `va
lue_counts`).
- The target has 77526 total rows with 75036 label as 0 ('healthy loan') and 2500 label as 1 ('high-risk loan').  So, the show inbalance towards 'healthy loan'.  This might influence the prediction result.

* Describe the stages of the machine learning process you went through as part of this analysis.
- The following is the workflow in performing the logistic regression:
- Define what is the target prediction and what is the features used to predict.  In this case, the target is 'loan_status' while the rest of the dataset are used as features.
- Next step is splitting the dataset to test and training sets for both target and features.
- The logistic regression is initiated with a selected random_state for repeatatbility with solver='lbfgs' used.
- The training dataset (target and features) are git into this model.
- Then prediction is made with the test dataset based on this training.
- Finally, the results of the prediction is carried ou by running confusion_matrix and the resulting classificaiton_report to view the precision, recall and f1-score for both loan.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
