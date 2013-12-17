'''
Data Scientist for  Day - Codelab source code - classification.py

Copyright (C) 2013  Ferrari Alessandro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.o
'''

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np


def dataset_scaling(X):
    
    scaler=preprocessing.Scaler().fit(X)
    X=scaler.transform(X)
    
    return X , scaler


def misclassification_errors(classifier, X_tr, y_tr, X_cv, y_cv):
    
    """
	TODO: Exercise 1
	Given an already trained classifier, the training set features and labels, and the
	cross-validation features and labels, compute the misclassification error measure
	on the training set and on the cross validation set, as explained in the lab track.
	Try to do it without for loops. That does not mean that you can use while loop instead.
    """

    raise Exception("Implement your own misclassification error measure!")
        
    return tr_err, cv_err
            

class ModelSelection(object):
    
    def __init__(self, C_list = None, gamma_list = None):
        
        assert C_list is None or isinstance(C_list, list)
        assert gamma_list is None or isinstance(gamma_list,list)
        
        if C_list is None:
            #regularization parameters
            self.C_list=[0.0000001,0.000001,0.00001,
                    0.0001,0.001,0.01,0.1,1,10,
                    100,1000,10000,100000,1000000]
        else:
            self.C_list = C_list
            
        if gamma_list is None:
            self.gamma_list = [0.0000001,0.000001,0.00001,
                          0.0001,0.001,0.01,0.1,1,10,
                          100,1000,10000,100000,1000000]
        else:
            self.gamma_list = gamma_list
    
    
    def C_gamma_selection(self,X, y, C_list = None, gamma_list = None,
                          classifier_by_C_function_params = None, 
                          test_size = 0.3, n_iterations = 20 ):
        
        if C_list is not None:
            self.C_list = C_list
            
        if gamma_list is not None:
            self.gamma_list = gamma_list
        
        tr_err_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        cv_err_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        
        acc_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)),dtype=np.float)
        prec_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        recall_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        f1_by_C_and_gamma=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        
        set_ripartitions = StratifiedShuffleSplit(y, n_iterations = n_iterations, 
                                                  test_size = test_size, indices=False)
    
        n_iter=len(set_ripartitions)
        
        for train,test in set_ripartitions:
            X_tr,X_cv,y_tr,y_cv =X[train],X[test],y[train],y[test]
            
            index_C=0
            for C in self.C_list:
                
                idx_gamma=0
                for gamma in self.gamma_list:
                
		    """
			TODO: Exercise 2
			For each combination of C and gamma, compute the training error, cross-validation error, 
			accuracy, precision, recall and f1 score obtained with the relative SVM rbf classifier,
			obtained averaging the results obtained by the different dataset partitions re-arrangements.
			The results will be stored relatively in the numpy arrays tr_err_by_C_and_gamma, 
			cv_err_by_C_and_gamma, acc_by_C_and_gamma, prec_by_C_and_gamma, recall_by_C_and_gamma,
			f1_score_by_C_and_gamma created previously. Columns contain the C index, while rows contain
			the gamma index. While doing the exercise, you may find useful the SVC class in sklearn.svm 
			module, the misclassification_errors that you implemented in the previous exercise and the
			score functions that are implemented in sklearn metrics.
		    """

		    raise Exception("Wake up! You are supposed to implement this part of code!")

                    
                    idx_gamma=idx_gamma + 1
                
                index_C=index_C + 1
                
        result=dict()
        result["C_list"]=self.C_list
        result["gamma_list"]=self.gamma_list
        result["tr_err_by_C_and_gamma"]=tr_err_by_C_and_gamma
        result["cv_err_by_C_and_gamma"]=cv_err_by_C_and_gamma
        result["acc_by_C_and_gamma"]=acc_by_C_and_gamma
        result["prec_by_C_and_gamma"]=prec_by_C_and_gamma
        result["recall_by_C_and_gamma"]=recall_by_C_and_gamma
        result["f1_by_C_and_gamma"]=f1_by_C_and_gamma
        
        return result
             
             
class LearningCurves(object):
        
    def stratifiedShuffleMask(self, y,m):
        pos_m=np.ceil(m/2)
        neg_m=m-pos_m
        
        max_pos=np.sum(y)
        max_neg=len(y)-max_pos
        
        if(pos_m>max_pos):
            pos_m=max_pos
            neg_m=m-pos_m
            
        if(neg_m>max_neg):
            neg_m=max_neg
            pos_m=m-neg_m
                
        mask=np.zeros(len(y),dtype=np.float)
        
        idx=0
        while pos_m>0:
            if y[idx]==1:
                mask[idx]=1
                pos_m-=1
            idx+=1
    
        idx=0   
        while neg_m>0:
            if y[idx]==0:
                mask[idx]=1
                neg_m-=1
            idx+=1
            
        return mask
    
    def compute(self,X,y,C,gamma, test_size=0.3, n_iterations = 5, training_set_minsize = 10, learning_curves_step = 20):   
        
        assert len(X)==len(y)
        assert len(y)>training_set_minsize
        
        assert isinstance(C, (int, float))
        assert isinstance(gamma, (int, float))
        
        train_size=int( round( (1-test_size) * len(y) ))   
        set_ripartitions = StratifiedShuffleSplit(y, n_iterations = n_iterations, 
                                                  train_size=train_size, 
                                                  indices=False)
        n_iter=len(set_ripartitions)
        
        n_samples=X.shape[0]
        n_features=X.shape[1]
        
        m_list=range(training_set_minsize,train_size,learning_curves_step)
            
        tr_errors=np.zeros((len(m_list),1),dtype=np.float)
        cv_errors=np.zeros((len(m_list),1),dtype=np.float)
        
        for train,test in set_ripartitions:
            X_tr,X_cv,y_tr,y_cv =X[train],X[test],y[train],y[test]
        
            idx=0
            for m in m_list:
                
                y_mask = self.stratifiedShuffleMask(y_tr,m)
                x_mask = np.kron(np.ones((n_features,1)),y_mask).T
            
                reduced_X = X_tr[x_mask!=0].reshape(m,n_features)
                reduced_y = y_tr[y_mask!=0]


		"""
			TODO: Exercise 3
			Read the code of the current method "compute" and understand what is
			happening. Once you have understood the code, try to understand the
			meaning of the stratifiedShuffleMask method. What is that method suppose 
			to do? What do reduced_X and reduced_y contain?
			Then, compute for each m, the training error and the cross-validation error
			averaged by the different re-arranged dataset ripartitions, and store them 
			relatively in the tr_errors and cv_errors numpy vectors, order by the idx index.
		"""
            
                raise Exception("One last effort! It is the last exercise.")
                
		idx+=1
                
                result=dict()
                result["m_list"]=m_list
                result["tr_errors"]=tr_errors
                result["cv_errors"]=cv_errors
                
        return result
             
    
class SVM(object):
       
    
    def model_selection(self,X,y, C_list = None, gamma_list = None, test_size = 0.3, n_iterations = 20):
        
        assert X!=None and y!=None     
        assert len(X)==len(y)
            
        model_selection = ModelSelection(C_list=C_list,gamma_list=gamma_list)
        parameters_result = model_selection.C_gamma_selection(X, y, 
                                                              test_size = 0.3, n_iterations = n_iterations)
            
        return parameters_result
    
    
    def training(self,X,y,C=1,gamma=None):
        
        assert isinstance(C,(int,float))
        assert isinstance(gamma,(int,float))
        
        self.classifier = SVC(kernel="rbf",C=C,gamma=gamma)
        self.classifier.fit(X,y)

        return self.classifier
        
    
    def classify(self,X):
        
        assert hasattr(self,"classifier")
        
        return self.classifier.predict(X)
