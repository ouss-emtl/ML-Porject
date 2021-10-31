#Useful libraries
import numpy as np
from collections import Counter
# creating the k nearest neighborrs class
class KNN:
class KNN:
    def __init__(self,k=3,qMinkowski=2):
        self.k=k
        self.qMinkowski=qMinkowski
        # this method fit the model to the data
        # for the knn the data only need to be stored for the training  
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    # prediction for all the datas
    def predict(self,X):
        return np.array([self._predict(x) for x in X])
    # prediction for an instance
    def _predict(self,x):
        # compute the distances to the instance
        distances_to_x=[np.linalg.norm(x-x_,self.qMinkowski) for x_ in self.X_train]
        # get the k nearest neighbors labels
        k_indices=np.argsort(distances_to_x[:self.k])
        kn_labels=[self.y_train[i] for i in k_indices]
        # using the majority vote to obtain the label
        x_label=Counter(kn_labels).most_common(1)[0][0]
        return x_label
    # The accuracy 
    def accuracy(self,y_pred,y_real):
        return np.sum(y_pred==y_real)/len(y_real)