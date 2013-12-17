'''
Data Scientist for  Day - Codelab source code - utilities.py

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def plot_3d( x, y, z, xlabel = "log10(gamma)", ylabel ="log10(C)" , zlabel = "accuracy", title = "Accuracy by C and gamma"):
        
        fig = plt.figure()
        fig.canvas.set_window_title('{0}'.format(title)) 
        ax = fig.add_subplot(111, projection='3d')
        
        ax_xlabel = ax.set_xlabel(xlabel)
        ax_ylabel = ax.set_ylabel(ylabel)
        ax_zlabel = ax.set_zlabel(title)
        
        x1=np.log10(x)
        y1=np.log10(y)
        z1=z
        X1,Y1=np.meshgrid(np.array(x1), np.array(y1))
        surf = ax.plot_surface(X1, Y1, z1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)
    
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
        fig.colorbar(surf, shrink=0.5, aspect=5)
        

def plot_learning_curves(x1, y1, x2, y2):
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('training and cross-validation error')
        ax.plot(x1,y1,"ro-")
        ax.plot(x2,y2,"go-")


def save_csv_submitted_labels(predicted_labels, filename):
       
        f = open(filename, "w")
        len_labels = predicted_labels.shape[0]
        f.write("Id,Solution\n")
        for i in xrange(len_labels):
            f.write("{0},{1}\n".format(i+1,int(predicted_labels[i])))
        f.close()
