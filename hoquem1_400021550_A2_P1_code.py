# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:21:53 2020

@author: Mahimul
"""
"""
Assignment 2 Part 1: Binary Classification with Logistic
Regression and k-Nearest Neighbours

Mahimul Hoque - hoquem1 - 400021550
"""

'''imports'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score   
from sklearn.neighbors import KNeighborsClassifier

def calc_w(X_train, t_train, N):
    new_col=np.ones(N)
    X1_train = np.insert(X_train, 0, new_col, axis=1) # dummy feature was included
    
    A = np.dot(X1_train.transpose(), X1_train)
    Ainv = np.linalg.inv(A)
    
    XT = np.dot(X1_train.transpose(), t_train)
    w = np.dot(Ainv, XT)
    
    return w

def calc_gradient(w, iteration, X_train, t_train, N):
    new_col = np.ones(N)
    X1_train = np.insert(X_train, 0, new_col, axis = 1)
    alpha = 1 #for regularization (no reg = 1)
    
    iteration = 100
    
    gr_norms = np.zeros(iteration)
    
    for n in range(iteration):
        z = np.dot(X1_train,w) #changes each iteration due to w changing
        y = 1/(1+np.exp(-z))
        diff = y-t_train
        gr = np.dot(X1_train.T, np.transpose(diff.T))/N #this is the gradient 
        #computer squared norm of the gradient
        gr_norm_sq = np.dot(gr,gr)
        gr_norms[n] = gr_norm_sq
        #update the vector of parameters
        w = w - np.dot(alpha,gr)
    
    return w

def calc_partA(X_test, t_test, M, w):
    '''Need to get P and R but for each threshold. Sort z first'''
    new_col=np.ones(M)
    X1_test = np.insert(X_test, 0, new_col, axis=1) # dummy feature was included
    z = np.dot(X1_test,w)
    
    y_pred = np.zeros(M)
    FP = 0
    TP = 0
    FN = 0
    u = np.zeros(M)
    for i in range(M):
        if(z[i] >= 0):
            y_pred[i] = 1 
        u[i] = y_pred[i] - t_test[i]
        
        if(u[i] == 1):
            FP += 1
        if(y_pred[i] == 1 & t_test[i] == 1):
            TP += 1
        if(u[i] == -1):
            FN += 1

    P = TP / (TP+FP)
    R = TP / (TP+FN)
    F1_scr = 2*P*R/(P+R)
            
    u = y_pred - t_test
    
    err = np.count_nonzero(u)/M  #misclassification rate
    
    z = np.sort(z)

    d = (M,M)
    u = np.zeros(d)
    P = np.zeros(M)
    R = np.zeros(M)
    F1 = np.zeros(M)

    #determining if it is FP, TP, or FN
    for i in range(M):
        y = np.zeros(M)
        FP = 0
        TP = 0
        FN = 0
        for j in range(M):
            if(z[j]>=z[i]):
                y[j]=1 
            u[i][j] = y[j] - t_test[j]
            if(u[i][j] == 1):
                FP += 1
            if(y[j] == 1 & t_test[j] == 1):
                TP += 1
            if(u[i][j] == -1):
                FN += 1

        P[i] = TP / (TP+FP)
        R[i] = TP / (TP+FN)
        F1[i] = 2*P[i]*R[i]/(P[i]+R[i])
        #if gives NAN, assumes it is 0
        if(np.isnan(F1[i])):
            F1[i] = 0
    
    plt.scatter(R, P)
    plt.title("Precision vs Recall")
    plt.show()
    
    return err, F1_scr, y_pred

def calc_partB(X_train, t_train, X_test, t_test):
    #setting properties for logistic reg tool from sklearn
    log_reg = LogisticRegression(random_state = 1550)
    log_reg.fit(X_train, t_train)
    
    y_pred = log_reg.predict(X_test)
    
    average_precision = average_precision_score(t_test, y_pred)
    
    disp = plot_precision_recall_curve(log_reg, X_test, t_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    
    f1score = f1_score(t_test, y_pred)

    return 1-accuracy_score(t_test, y_pred), f1score, y_pred

def calc_kNN(X_train, X_test, t_train, t_test, k_value):
    M = len(X_test)
    N = len(X_train)
               
    #initialize arrays
    dist = np.zeros((M,N)) #2dim array to store distances from test points to trainig points
    ind = np.zeros((M,N))  #2dim array to store the order after sorting the distances
    u = np.arange(N)       # array of numbers from 0 to N-1
    
    for j in range(M):
        ind[j,:] = u

    #compute distances and sort
    for j in range(M): #each test point
        for i in range(N): #each training point
            z = X_train[i,:]-X_test[j,:]
            dist[j,i] = np.dot(z,z)

    ind = np.argsort(dist)
    
    y = np.zeros(M)
    #checks based on the the k value, which class outweights the other
    for j in range(M):
        temp_y = []
        for i in range(k_value):
            temp_y.append(t_train[ind[j,i]])
        if(np.count_nonzero(temp_y) > len(temp_y)-np.count_nonzero(temp_y)):
            y[j] = 1
        else:
            y[j] = 0
    
    z = y - t_test
    
    return z, y

def calc_partC(X_train, t_train, X_test, t_test):
    kF = KFold(n_splits = 5, shuffle = True, random_state = 1550)
    
    err_val = np.zeros(5)
    cross_val_err = []
    for k in range(0,5):
        
        #split for K Fold
        for train_index, test_index in kF.split(X_train):
            
            XK_train, XK_test = X_train[train_index], X_train[test_index] #the folds are here
            TK_train, TK_test = t_train[train_index], t_train[test_index]
            
            #obtaining z and y from k nearest neighbors function
            z, y = calc_kNN(XK_train, XK_test, TK_train, TK_test, k)
            err_val[k] += np.sum(np.dot(z, z.T))/len(z)
        
        cross_val_err.append(err_val[k]/5)
        
    #min error
    min_err = err_val[np.argmin(err_val)]/5
    k_value = 1+np.argmin(err_val)
    
    #using k with the min error from KFold to get z and y again and get the misclassifier error and f1 score
    z, y = calc_kNN(X_train, X_test, t_train, t_test, k_value)
    M = len(X_test)
    miss = np.count_nonzero(z)/M
    f1 = f1_score(t_test, y)
        
    return miss, f1, cross_val_err

#basically same as calc_partC but with sklearn kNN and predict
def calc_partD(X_train, t_train, X_test, t_test):
    kF = KFold(n_splits = 5, shuffle = True, random_state = 1550)

    err_val = np.zeros(5)
    cross_val_err = []
    for k in range(5):
        kNN = KNeighborsClassifier(n_neighbors = k+1)

        for train_index, test_index in kF.split(X_train):
            
            XK_train, XK_test = X_train[train_index], X_train[test_index] #the folds are here
            TK_train, TK_test = t_train[train_index], t_train[test_index]
        
            kNN.fit(XK_train, TK_train)
            y_pred = kNN.predict(XK_test)
            
            diff_t = np.subtract(TK_test, y_pred)
            err_val[k] += np.sum(np.dot(diff_t, diff_t.T))/len(diff_t)

        cross_val_err.append(err_val[k]/5)
        
    min_err = err_val[np.argmin(err_val)]/5
    k_value = 1+np.argmin(err_val)
    
    kNN = KNeighborsClassifier(n_neighbors = k_value)
    kNN.fit(X_train, t_train)
    
    y_pred = kNN.predict(X_test)

    miss = 1-accuracy_score(t_test, y_pred)
    f1 = f1_score(t_test, y_pred)

    return miss, f1, cross_val_err
    
def main():
    
    #splitting into X and t
    X, t = load_breast_cancer(return_X_y=True)
        
    #split data into training and test sets
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 1550)
    
    sc = StandardScaler()
    X_train[:, :] = sc.fit_transform(X_train[:, :])
    X_test[:, :] = sc.transform(X_test[:, :])
    
    N = len(X_train)
    M = len(X_test)
    
    #calculate initial w
    w = calc_w(X_train, t_train, N)
    #calculate w with gradient
    iteration = 100
    w = calc_gradient(w, iteration, X_train, t_train, N)
    #calc P, R, F1 score, missclassification rate and return best one
    err_partA, f1_partA, yA = calc_partA(X_test, t_test, M, w)
    print("\n")
    print("A Misclassification Error = " + str(err_partA) + " F1 Score = " + str(f1_partA) + "\n" + "Vector of Parameters \n" + str(yA) + "\n")
    #Use sklearn logistic reg and precision/recall curve
    err_partB, f1_partB, yB = calc_partB(X_train, t_train, X_test, t_test)
    print("B Misclassification Error B = " + str(err_partB) + " F1 Score = " + str(f1_partB) + "\n" + "Vector of Parameters \n" + str(yB) + "\n")    
    # calc kNN and K-Fold Cross Validation
    err_partC, f1_partC, cross_val_errC = calc_partC(X_train, t_train, X_test, t_test)
    print("C Misclassification Error = " + str(err_partC) + " F1 Score = " + str(f1_partC) + "\n" + " Cross Validation Error \n" + str(cross_val_errC) + "\n")    
    # use sklearn kNN and K-Fold Cross Validaion
    err_partD, f1_partD, cross_val_errD = calc_partD(X_train, t_train, X_test, t_test)
    print("D Misclassification Error = " + str(err_partD) + " F1 Score = " + str(f1_partD) + "\n" + " Cross Validation Error \n" + str(cross_val_errD) + "\n")
        
main()
