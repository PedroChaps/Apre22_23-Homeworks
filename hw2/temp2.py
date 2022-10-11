
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

data = loadarff('hw2/pd_speech.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
df['class'] = df['class'].str.decode('utf-8')

X = df.drop('class', axis=1)
Y = df['class']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 
knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', metric='euclidean')
nb = GaussianNB()

for train, test in skf.split(X, Y):
    knn.fit(X.iloc[train], Y.iloc[train])
    nb.fit(X.iloc[train], Y.iloc[train])
    
    knn_y_predict = knn.predict(X.iloc[test])
    nb_y_predict = nb.predict(X.iloc[test])

    knn_y_predict = np.round(knn_y_predict)
    nb_y_predict = np.round(nb_y_predict)

    knn_y_predict = np.array(knn_y_predict, dtype='int')
    nb_y_predict = np.array(nb_y_predict, dtype='int')

    knn_y_predict = knn_y_predict.astype('str')
    nb_y_predict = nb_y_predict.astype('str')

    knn_y_predict[knn_y_predict == '0'] = 'non-PD'