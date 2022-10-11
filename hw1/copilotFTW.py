"""
Using sklearn, apply a stratified 70-30 training-testing split with a fixed seed
(random_state=1), and assess in a single plot the training and testing accuracies of a decision tree
with no depth limits (and remaining default behavior) for a varying number of selected features
in {5,10,40,100,250,700}. Feature selection should be performed before decision tree learning
considering the discriminative power of the input variables according to mutual information
criterion (mutual_info_classif).
Use the pd_speech.arff dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

