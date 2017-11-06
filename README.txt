LogisticRegression_py:
Generate beta coefficients for a logistic regression model and determine the accuracy of the model on training data set.

Getting started:
Run the script to import Breast Cancer Wisconsin (diagnostic) Data Set from UCI Machine Learning repository as a pandas DataFrame.  Subsequent steps will use a logistic regression to predict whether each observation is derived from a patient having a malignant or benign lession.  Following several iterations through the data set (epoch = 10) the beta variable will contain 30 coeffiecients for the predictive model based on the 30 predictor variables contained in data set.

The Logistic Regression Model section can be run on any data set to generate a predictive model given the data is prepared correctly.  Create a numpy array that consists of a desired number of numerical predictor variables (substitute for normBC array).  Additionally create a 'response' numpy array that contains binary response variables (substitute for label array). Set the alpha value used to adjust beta coefficient after each iteration.  Set epoch variable equal to the number of times you wish to iterate through the data set.