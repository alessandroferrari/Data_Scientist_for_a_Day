'''
Data Scientist for  Day - Codelab source code - main.py

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
import os
import classification
import numpy as np
from classification import SVM, LearningCurves, dataset_scaling
import matplotlib.pyplot as plt
from utilities import plot_3d, save_csv_submitted_labels, plot_learning_curves

testdir = os.getcwd()

print "Reading the dataset from file..."
# Read data
train = np.genfromtxt(open(os.path.join(testdir, 'train.csv'),'rb'), delimiter=',')
target = np.genfromtxt(open(os.path.join(testdir, 'trainLabels.csv'),'rb'), delimiter=',')
test = np.genfromtxt(open(os.path.join(testdir, 'test.csv'),'rb'), delimiter=',')
print "Dataset loaded!"
#features scaling
print "Starting features preprocessing ..."

dataset_scaled, scaler = dataset_scaling(np.vstack((train,test)))
train_scaled = dataset_scaled[:1000]
test_scaled = dataset_scaled[1000:]

print "Features preprocessing done!"

classification_obj=SVM()

print "Starting model selection ..."

#performing model selection

C_list = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
gamma_list = [0.0001,0.001,0.01,0.1,1,10, 100,10000]

ms_result = classification_obj.model_selection(train_scaled,target,n_iterations=3, 
                                               C_list=C_list, 
                                               gamma_list=gamma_list)

#displaying model selection
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["acc_by_C_and_gamma"], zlabel="accuracy", title="Accuracy by C and gamma")
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["recall_by_C_and_gamma"], zlabel="recall", title="Recall by C and gamma")
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["prec_by_C_and_gamma"], zlabel="precision", title="Precision by C and gamma")
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["f1_by_C_and_gamma"], zlabel="accuracy", title="f1 score by C and gamma")
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["tr_err_by_C_and_gamma"], zlabel="training error", title="Training error score by C and gamma")
plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["cv_err_by_C_and_gamma"], zlabel="cross-validation error", title="Cross-validation error score by C and gamma")
plt.show()

#entering the C and gamma chosen
print "Plotted graphics for model selection. Choose the best C and gamma ..."
            
while True:
    C_str = raw_input("Enter the C value suggested by model selection:")
    try:
        C = float(C_str)
    except Exception as e:
        print "Invalid C inserted. C has to be numeric. Exception: {0}".format(e)
        continue
    break

while True:
    gamma_str = raw_input("Enter the gamma value suggested by model selection:")
    try:
        gamma = float(gamma_str)
    except Exception as e:
        print "Invalid gamma inserted. gamma has to be numeric. Exception: {0}".format(e)
        continue
    break

print "Parameters selection performed! C = {0}, gamma = {1}".format(C, gamma)    

#training
print "Performing training..."

classifier = classification_obj.training(train_scaled, target, C=C, gamma=gamma)

print "Training performed!"

#prediction on kaggle test set
print "Performing classification on the test set..."

predicted = classification_obj.classify(test_scaled)

print "Classification performed on the test set!"

#plot learning curves
print "Plotting learning curves..."
learning_curves = LearningCurves()
learning_curves_result = learning_curves.compute(X=train_scaled,y=target,C=C,gamma=gamma)
    
plot_learning_curves(x1=learning_curves_result["m_list"], 
                     y1=learning_curves_result["tr_errors"], 
                     x2=learning_curves_result["m_list"], 
                     y2=learning_curves_result["cv_errors"])
    
plt.show()

#save data in the submission format
print "Saving classification results..."
save_csv_submitted_labels(predicted, os.path.join(testdir,"predicted_y.csv"))

print "Congratulations! You have finished this codelab, you rock!"
