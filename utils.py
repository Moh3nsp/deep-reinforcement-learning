# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:21:42 2021

@author: mohsen
"""

import matplotlib.pyplot as plt 
import numpy as np


def plot_learning_curve(x,scores,epsilons,filename):
    fig=plt.figure()
    ax=fig.add_subplot(111,label="1")
    ax2=fig.add_subplot(111,label="2" , frame_on = False)
    
    
    ax.plot(x,epsilons,color="C0")
    ax.set_xlabel("training Steps", color="C0")
    ax.set_ylabel("Epsilon" , color="C0")
    ax.tick_params(axis='x'  , colors="C0")
    ax.tick_params(axis='y'  , colors="C0")
    
    
    N=len(scores)
    running_Avg=np.empty(N)
    for t in range(N):
        running_Avg[t] = np.mean(scores[max(0,t-100):(t+1)])
    
    ax2.scatter(x,running_Avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('score',color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y' , colors="C1")

    plt.savefig(filename)    
    
    
    
    
    
    
    
    
    
    