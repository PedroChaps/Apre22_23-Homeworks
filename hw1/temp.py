"""
Using sklearn, apply a stratified 70-30 training-testing split with a fixed seed
(random_state=1), and assess in a single plot the training and testing accuracies of a decision tree
with no depth limits (and remaining default behavior) for a varying number of selected features
in {5,10,40,100,250,700}. Feature selection should be performed before decision tree learning
considering the discriminative power of the input variables according to mutual information
criterion (mutual_info_classif).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

def main():
    # Read data
    data = pd.read_csv('data.csv')

    # Split data
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Select features
    n_features = [5,10,40,100,250,700]
    for n in n_features:
        # Select features
        mi = mutual_info_classif(X_train, y_train)
        mi = pd.Series(mi)
        mi.index = X_train.columns
        mi.sort_values(ascending=False, inplace=True)
        features = mi[:n].index

        # Train classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train[features], y_train)

        # Predict
        y_train_pred = clf.predict(X_train[features])
        y_test_pred = clf.predict(X_test[features])

        # Compute accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Plot
        plt.scatter(n, train_acc, c='b', label='train')
        plt.scatter(n, test_acc, c='r', label='test')
        plt.title('Accuracy')
        plt.xlabel('Number of features')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.show()
