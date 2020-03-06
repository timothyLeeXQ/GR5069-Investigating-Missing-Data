# Assignment 2: Analysing missing data in F1 Dataset

## Objectives

* Uncover patterns of missing data in datasets
* See how missing data can affect the distribution of variables
* Use and evaluate imputation techniques

**Notes:**
1. The code for this assignment was written using a mix of PySpark and Pandas. As far as I know, it needs to run off Databricks.
2. KNN Imputation was used as recommended, but the long model training time means that running the code is problematic. I have commented out the code for KNN imputation after getting the model to give me the imputed DF. This Df was saved to my S3 bucket's `interim` folder, which was then read in for the rest of the assignment.
