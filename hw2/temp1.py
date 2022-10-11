"""
Use the the pd_speech.arff file.
Using sklearn, considering a 10-fold stratified cross validation (random=0), plot the cumulative testing confusion matrices of kNN (uniform weights, k = 5, Euclidean distance) and Naive Bayes (Gaussian assumption). Use all remaining classifier parameters as default.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from scipy.io.arff import loadarff

def main():
    data = loadarff('hw2/pd_speech.arff')
    df = pd.DataFrame(data[0])
    df = df.dropna()
    df['class'] = df['class'].str.decode('utf-8')

    X = df.drop('class', axis=1)
    y = df['class']

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    nb = GaussianNB()

    knn_cms = []
    nb_cms = []

    for train_index, test_index in skf.split(X, y):
        knn.fit(X[train_index], y[train_index])
        nb.fit(X[train_index], y[train_index])
        knn_cms.append(confusion_matrix(y[test_index], knn.predict(X[test_index])))
        nb_cms.append(confusion_matrix(y[test_index], nb.predict(X[test_index])))

    knn_cms = np.sum(np.array(knn_cms), axis=0)
    nb_cms = np.sum(np.array(nb_cms), axis=0)

    plt.figure()
    plt.imshow(knn_cms)
    plt.title('kNN Confusion Matrix')
    plt.show()

    plt.figure()
    plt.imshow(nb_cms)
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()
    
if __name__ == '__main__':
    main()