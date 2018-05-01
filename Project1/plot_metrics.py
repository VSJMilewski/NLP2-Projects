# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:09:32 2018

@author: Diana
"""

from matplotlib import pyplot as plt
import pickle

def plot(metrics, filename, title, folder='plots'):
    # metrics is a vector where the index is the epoch and the value is the metric
    
    plt.plot(range(len(metrics)), metrics, 'bo')
    plt.ylabel('AER')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.savefig('{}/{}.jpg'.format(folder,filename))

metrics_dict = pickle.load(open( "data/pickles/modelI_report_metrics.p", "rb" ) )
val_aers = metrics_dict['val_aers']
#print(val_aers[10])
plot(val_aers, 'val_aers_modelI_EM', 'Evolution of AER on validation data per training epoch')