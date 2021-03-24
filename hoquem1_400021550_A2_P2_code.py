# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:11:30 2020

@author: Mahimul
"""

"""
Assignment 2 Part 2: Ensemble Methods

Mahimul Hoque - hoquem1 - 400021550
"""

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import time    

def main():
    #attaining data from spambase database
    dataset = pd.read_csv('spambase.data')
    X = dataset.iloc[:, :-1].values
    t = dataset.iloc[:, -1].values
    
    #split data into training and test sets
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/3, random_state = 1550)
    
    #setting KFold cross validation properties
    # K = 5
    kF = KFold(n_splits = 5, shuffle = True, random_state = 1550)   
    
    max_leaves = []
    cross_val_err = []
    
    #looping through the range of max leaves for decision tree
    for n in range(2,401):
        
        max_leaves.append(n)
        
        #setting decision tree classifier properties as max leaves changes
        clf = DecisionTreeClassifier(random_state = 1550, max_leaf_nodes = n)
        
        err_val = 0
        
        #splitting X_train into 5 folds
        for train_index, test_index in kF.split(X_train):
                
            XK_train, XK_test = X_train[train_index], X_train[test_index] #the folds are here
            TK_train, TK_test = t_train[train_index], t_train[test_index]
            
            #iftting according to the splitted sets
            clf.fit(XK_train, TK_train)
            y_pred = clf.predict(XK_test)
            
            #obtaining validation error
            diff_t = np.subtract(TK_test, y_pred)
            err_val += np.sum(np.dot(diff_t, diff_t.T))/len(diff_t) #summing errors to average later
        
        #storing all cross val errors
        cross_val_err.append(err_val/5)
        
        #choosing the best max leaves based on the smallest cross val error
        if(n == 2):
            amount_leaves = n
            best_err = err_val/5
        if(best_err > err_val/5):
            amount_leaves = n
            best_err = err_val/5
    
    #setting clf with the best max leaves and finding the test error
    clf = DecisionTreeClassifier(random_state = 1550, max_leaf_nodes = amount_leaves)
    clf.fit(X_train, t_train)
    y_pred = clf.predict(X_test)
    diff_t = np.subtract(t_test, y_pred)
    err_test = np.sum(np.dot(diff_t, diff_t.T))/len(diff_t)
        
    predictors = []
    dt_error = []
    bg_err_test = []
    rf_err_test = []
    ab1_err_test = []
    ab2_err_test = []
    ab3_err_test = []
    
    #looping through predictor range for ensemble methods
    for n in range(100, 1100, 100):
        predictors.append(n)
        dt_error.append(err_test)
        
        #bagging classifier setting properties and obtaining test error
        bg = BaggingClassifier(n_estimators = n, random_state = 1550)
        bg.fit(X_train, t_train)
        y_pred = bg.predict(X_test)
        diff_t = np.subtract(t_test, y_pred)
        bg_err_test.append(np.sum(np.dot(diff_t, diff_t.T))/len(diff_t))     
        
        #random forest classifier setting properties and obtaining test error
        rf = RandomForestClassifier(n_estimators = n, random_state = 1550)
        rf.fit(X_train, t_train)
        y_pred = rf.predict(X_test)
        diff_t = np.subtract(t_test, y_pred)
        rf_err_test.append(np.sum(np.dot(diff_t, diff_t.T))/len(diff_t))      
        
        #adaboost classifier setting properties and obtaining test error        
        ab1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, random_state = 1550),n_estimators = n, random_state = 1550)
        ab1.fit(X_train, t_train)
        y_pred = ab1.predict(X_test)
        diff_t = np.subtract(t_test, y_pred)
        ab1_err_test.append(np.sum(np.dot(diff_t, diff_t.T))/len(diff_t))

        #adaboost classifier setting properties and obtaining test error         
        ab2 = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 1550),n_estimators = n, random_state = 1550)
        ab2.fit(X_train, t_train)
        y_pred = ab2.predict(X_test)
        diff_t = np.subtract(t_test, y_pred)
        ab2_err_test.append(np.sum(np.dot(diff_t, diff_t.T))/len(diff_t))

        #adaboost classifier setting properties and obtaining test error         
        ab3 = AdaBoostClassifier(n_estimators = n, random_state = 1550)
        ab3.fit(X_train, t_train)
        y_pred = ab3.predict(X_test)
        diff_t = np.subtract(t_test, y_pred)
        ab3_err_test.append(np.sum(np.dot(diff_t, diff_t.T))/len(diff_t))

    print("Decision Tree Errors " + str(dt_error))
    print("Bagging Classfier Errors " + str(bg_err_test))
    print("Random Forest Classifier Errors " + str(rf_err_test))
    print("Adaboost Classifier 1 Errors " + str(ab1_err_test))
    print("Adaboost Classifier 2 Errors " + str(ab2_err_test))
    print("Adaboost Classifier 3 Errors " + str(ab3_err_test))
    
    #Plot of test errors
    plt.scatter(predictors, dt_error, label = 'Decision Tree', color = 'black')
    plt.scatter(predictors, bg_err_test, label = 'Bagging',  color = 'red')
    plt.scatter(predictors, rf_err_test, label = 'Random Forest',  color = 'cyan')
    plt.scatter(predictors, ab1_err_test, label = 'Adaboost 1',  color = 'blue')
    plt.scatter(predictors, ab2_err_test, label = 'Adaboost 2',  color = 'purple')
    plt.plot(predictors, ab3_err_test, label = 'Adaboost 3', color = 'green')
    plt.title("Test Errors vs. Predictors Plot")
    plt.legend()
    plt.show()
    
    #Plot of max leaves vs cross val errors
    plt.scatter(max_leaves, cross_val_err)
    plt.title("Cross Validation Error vs. Max Leaves")
    plt.show()

#tracking time to run with set amount of predictors
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
