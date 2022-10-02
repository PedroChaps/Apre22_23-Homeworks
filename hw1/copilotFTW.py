"""
Using sklearn, apply a stratified 70-30 training-testing split with a fixed seed
(random_state=1), and assess in a single plot the training and testing accuracies of a decision tree
with no depth limits (and remaining default behavior) for a varying number of selected features
in {5,10,40,100,250,700}. Feature selection should be performed before decision tree learning
considering the discriminative power of the input variables according to mutual information
criterion (mutual_info_classif).
Use the pd_speech.arff dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy.io.arff import loadarff

def main():
    
    # Loads the data from the available .arff file
    data = loadarff('pd_speech.arff')
    df = pd.DataFrame(data[0])
    df = df.dropna()
    df['class'] = df['class'].str.decode('utf-8')
    
    X=df.drop('class', axis=1)
    y = df['class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Feature selection
    selected_features = [5, 10, 40, 100, 250, 700]
    train_accs = []
    test_accs = []
    for f in selected_features:
        # Select features
        mi = mutual_info_classif(X_train, y_train)
        mi = mi / np.max(mi)
        idx = np.argsort(mi)[::-1][:f]
        X_train_sel = X_train.iloc[:, idx]
        X_test_sel = X_test.iloc[:, idx]

        # Train decision tree
        clf = DecisionTreeClassifier()
        clf.fit(X_train_sel, y_train)

        # Predict
        y_train_pred = clf.predict(X_train_sel)
        y_test_pred = clf.predict(X_test_sel)

        # Compute accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # Plot
    plt.plot(selected_features, train_accs, label="Training accuracy")
    plt.plot(selected_features, test_accs, label="Testing accuracy")
    plt.xlabel("Number of selected features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()