#import libraries
import pandas as pd
import numpy as np
import math

#This naive bayes is implemented for binary classifications
class NAIVE_BAYES():

    def __init__(self,k = 0.5):
        self.k = k
        self.__estimator_type = "classifier"
    
    def fit(self,X,y):

        #store the real classes
        self.y = y

        #Store the probability of being of the class s
        self.classes_,_ = np.unique(y, return_inverse=True)
        self.p_s = y.value_counts()[self.classes_[1]]/len(y)

        #Store the X into two dataframe
        self.df_s = X.iloc[list(np.where(np.array(y) == self.classes_[1])).pop(),:]
        self.df_not_s = X.iloc[list(np.where(np.array(y) != self.classes_[1])).pop(),:]
        
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    #It's possible to storaged the probabilities when we fit the model, 
    #but we don't do this because of the size of the df that we are working with
    def predict_proba(self,X):

        #Prepare the data
        X = X.replace('?',2.0).copy()
        X = X.replace('y',1.0).copy()
        X = X.replace('n',0.0).copy()

        probabilities = []
        for _,line in X.iterrows():
            probability_xi_s = 0
            probability_xi_not_s = 0

            for position in range(len(line)):
                value = line.iloc[position]

                #Calculate the probability to have the value if it's from the class s
                probability_xi_s += math.log((1 +self.df_s.iloc[:,position].value_counts()[value])/(2+len(self.df_s)))

                #Calculate the probability to don't have the value if it's from the class s
                probability_xi_not_s += math.log((1 +self.df_not_s.iloc[:,position].value_counts()[value])/(2+len(self.df_not_s)))

            probability = ((math.exp(probability_xi_s)*self.p_s)\
                /(math.exp(probability_xi_s)*self.p_s + math.exp(probability_xi_not_s)*(1-self.p_s)))

            probabilities.append([1-probability,probability])

        return np.array(probabilities)

    def decision_function(self,X):
        
        probability = self.predict_proba(X)
        return list(map(lambda x: 1 if x[1]>self.k else 0,probability))

    def predict(self,X):
        decisions = self.decision_function(X)
        return list(map(lambda x: self.classes_[x],decisions))
    
    def get_params(self, deep = True):
        return {"k": self.k}
    








