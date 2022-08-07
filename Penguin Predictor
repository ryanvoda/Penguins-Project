import pandas as pd
import urllib
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression # picked a MODEL CLASS
from sklearn.model_selection import cross_val_score

class penguinData():
    """ 
    Reads in csv file of penguin data and cleans up for Species Prediction 
    by functions logisticRegression() or randomForestClassifier()
    """
    
    def __init__(self, csv): #, X_train = pd.DataFrame(), X_test = pd.DataFrame(), y_train = pd.DataFrame(), y_test = pd.DataFrame()): # Takes in and reads csv file
        """
        Initialize class with user-supplied csv file
        Args:
            csv: a .csv file
        Returns: 
            None
        """ 
        
        self.csv = csv # instantiate the csv
        self.df = pd.read_csv(csv) # instantiate the df which reads in the csv file
        
    def readData(self): # Method that reads the data; pass in self, which contains the read csv file from the init method
        """
        Prints a readout of the data
        """
        
        print(self.df) # instantiate using self.___
    
    def dropNAN(self):
        """
        Drops any data that will not be used (i.e. drops unused columns or NAN values)
        """
        
        try: # only for training data, like palmer_penguins.csv, which contain Species data
            self.variables = self.df[['Culmen Length (mm)', 'Body Mass (g)', 'Island', 'Species']] # data on penguins with Species
        
        except KeyError: # for the actual dataset which does not contain Species data
            self.variables = self.df[['Culmen Length (mm)', 'Body Mass (g)', 'Island']] # data on penguins without Species
            
#         if 'Species' in self.df: # for "complete" data that contains true Species (e.g. palmer_penguins.csv)
#             self.variables = self.df[['Culmen Length (mm)', 'Body Mass (g)', 'Island', 'Species']] # for training data (palmer_penguins.csv)
        
#         else: # for data without Species
#             self.variables = self.df[['Culmen Length (mm)', 'Body Mass (g)', 'Island']] # for actual user data (data on penguins without Species)
            
        self.df = self.variables.dropna(axis = 0)

    def islandToNum(self): 
        """
        Changes categorical island data to numbers.
        """
        
        self.le = preprocessing.LabelEncoder() # makes an instance of labelencoder
        self.X = self.le.fit_transform(self.df['Island']) 
        self.df['Island_num'] = self.le.fit_transform(self.df["Island"]) # actually sets the #s to each island
    
    def splitTrainTest(self):
        """
        Splits the penguin data into X (predictor variables) and y (target variables). 
        If the data is training data (i.e. palmer_penguins.csv), generates X_train, X_test, y_train, and y_test groups;
        if the data is actual data (i.e. penguin data without Species), only creates dataframe of X (predictor variables)
        """
        
        self.dropNAN() # member function call to drop NAN values
        self.islandToNum() # member function call to turn island categories into numbers
    
        self.X = self.df[['Culmen Length (mm)', 'Body Mass (g)', 'Island_num']] # sets predictor variables
        
        if 'Species' in self.df: # for "complete" data that contains true Species (e.g. palmer_penguins.csv)
            self.y = self.df['Species'] # this sets the target variables
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 2022) # creates instance variables of X/y, train/test
#         print(self.X_train, self.X_test, self.y_train, self.y_test)

def logisticRegression(csv): 
    """
    Uses existing, complete penguin data (palmer_penguins.csv) as training dataset to predict the species of penguin on
    actual penguin dataset without species information using Logistic Regression model.
    Args: 
        csv: a .csv file
    Returns:
        Prints the train and test scores using Logistic Regression on existing, complete penguin data.
        Returns a new .csv file with species predictions appended to data from original csv .
    """
    
    training = penguinData('palmer_penguins.csv') # instantiate an object of class penguinData (class we just made)
    training.dropNAN()
    training.islandToNum()
    training.splitTrainTest() # call method splitTrainTest() on the object penguins
    
    actual = penguinData(csv) # csv file without Species in it (don't know the species but collected penguin data)
    actual.dropNAN()
    actual.islandToNum()
    actual.splitTrainTest()
    
    LR = LogisticRegression(max_iter = 500) # max_iter default usually 100, but our data is "too big"
    LR.fit(training.X_train, training.y_train) # for every review, LR will output a "probability" of the review being positive
    
    LR_train_score = LR.score(training.X_train, training.y_train) # score the training set
    LR_test_score = LR.score(training.X_test, training.y_test) # score the test set
#     # visually compare to see if there is overfitting
    
    speciesPrediction = LR.predict(actual.X) # makes species predictions using logistic regression
    
    actual.df['Species Prediction'] = speciesPrediction # adds column to dataframe with species predictions

    print("Trained Score:", LR_train_score, "\nTest Score:", LR_test_score) # prints trained vs test score
    
    return actual.df.to_csv('Predicted_Species_Logistic_Regression.csv') # returns a csv of all data plus the species predictions
        
        
def randomForestClassifier(csv):
    """
    Uses existing, complete penguin data (palmer_penguins.csv) as training dataset to predict the species of penguin on
    actual penguin dataset without species information using Random Forest Classifier model.
    Args: 
        csv: a .csv file
    Returns:
        Prints the train and test scores using Random Forest Classifier on existing, complete penguin data.
        Returns a new .csv file with species predictions appended to data from original csv .
    """
    
    training = penguinData('palmer_penguins.csv') # instantiate an object of class penguinData (class we just made)
    training.dropNAN()
    training.islandToNum()
    training.splitTrainTest() # call method splitTrainTest() on the object penguins
    
    actual = penguinData(csv)
    actual.dropNAN()
    actual.islandToNum()
    actual.splitTrainTest()
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(training.X_train, training.y_train)
    
    rfc_train_score = clf.score(training.X_train, training.y_train)
    rfc_test_score = clf.score(training.X_test, training.y_test)
    
    speciesPrediction = clf.predict(actual.X)
    actual.df['Species Prediction'] = speciesPrediction # adds column to dataframe with species predictions
    
    print("Trained Score:", rfc_train_score, "\nTest Score:", rfc_test_score) # prints trained vs test score
    
    return actual.df.to_csv('Predicted_Species_Random_Forest_Classifier.csv')
