#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:58:38 2021

@author: filtsem

This script caluclates the explain variance of acoustic features by subtstracting the phoneme features+freq model from the full model for both low and high entropy word models. 
Then plot the language vs word entropy graphs and their interaction. Color settings of graphs needs to be changed manually according to gragh. 
"""

from pathlib import Path
import numpy as np
import eelbrain
from matplotlib import pyplot
import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

Results_path = '/project/3027007.02/Scripts_for_publication/TRFs/Accuracy_analysis/Accuracies/'

# Configure the matplotlib figure style
FONT = "Times New Roman"
colors = cycler(color=[ 'lightgrey','dimgrey','yellowgreen' , 'olivedrab','coral','orangered' ]) #Dutch
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    # Font
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (4,3),
    'axes.prop_cycle':colors}



pyplot.rcParams.update(RC)

language = 'French'
region = 'Whole_brain'


DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
SUBJECTS.sort()  

models_low = ['equal_signal_acoustic_low',
          'equal_signal_acoustic+phonemes+surprisal+entropy_low',]
          
models_high = ['equal_signal_acoustic_high',
          'equal_signal_acoustic+phonemes+surprisal+entropy_high',]


#%%
rows = []


for subject in SUBJECTS:
    trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[0]}_lh.pickle')
    trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[0]}_rh.pickle')
    trf_lh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[1]}_lh.pickle')
    trf_rh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[1]}_rh.pickle')
    
    trf_rh.proportion_explained.x  = trf_rh.proportion_explained.x - trf_rh0.proportion_explained.x
    trf_lh.proportion_explained.x  = trf_lh.proportion_explained.x - trf_lh0.proportion_explained.x
    
    
    rows.append(['French', 'Low', 'Right', subject, models_low[1][:-4], trf_rh.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
    rows.append(['French', 'Low', 'Left', subject, models_low[1][:-4], trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    print(np.shape(rows))

for subject in SUBJECTS:
    trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[0]}_lh.pickle')
    trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[0]}_rh.pickle')
    trf_lh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[1]}_lh.pickle')
    trf_rh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[1]}_rh.pickle')
    
    trf_rh.proportion_explained.x  = trf_rh.proportion_explained.x - trf_rh0.proportion_explained.x
    trf_lh.proportion_explained.x  = trf_lh.proportion_explained.x - trf_lh0.proportion_explained.x
    
    rows.append(['French', 'High', 'Right', subject, models_high[1][:-5], trf_rh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    rows.append(['French', 'High', 'Left', subject, models_high[1][:-5], trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    print(np.shape(rows))
    
        
language = 'Dutch'
region = 'Whole_brain'


DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
SUBJECTS.sort()  

 
for subject in SUBJECTS:
    trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[0]}_lh.pickle')
    trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[0]}_rh.pickle')
    trf_lh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[1]}_lh.pickle')
    trf_rh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_low[1]}_rh.pickle')
    
    trf_rh.proportion_explained.x  = trf_rh.proportion_explained.x - trf_rh0.proportion_explained.x
    trf_lh.proportion_explained.x  = trf_lh.proportion_explained.x - trf_lh0.proportion_explained.x
    
    
    rows.append(['Dutch', 'Low', 'Right', subject, models_low[1][:-4], trf_rh.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
    rows.append(['Dutch', 'Low', 'Left', subject, models_low[1][:-4], trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    print(np.shape(rows))

for subject in SUBJECTS:
    trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[0]}_lh.pickle')
    trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[0]}_rh.pickle')
    trf_lh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[1]}_lh.pickle')
    trf_rh  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {models_high[1]}_rh.pickle')
    
    trf_rh.proportion_explained.x  = trf_rh.proportion_explained.x - trf_rh0.proportion_explained.x
    trf_lh.proportion_explained.x  = trf_lh.proportion_explained.x - trf_lh0.proportion_explained.x
    
    rows.append(['Dutch', 'High', 'Right', subject, models_high[1][:-5], trf_rh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    rows.append(['Dutch', 'High', 'Left', subject, models_high[1][:-5], trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    print(np.shape(rows))
 
        
df = pd.DataFrame(data = rows, columns = ['Language','Word Entropy' ,'hemisphere',  'subject','model', 'accuracy'])



df.to_csv(os.path.join(Results_path,'Accuracies_both_hemispheres_word_entropy_French_Dutch_part1_acoustic_diff.csv'))


plotting_parameters = {
    'data':    df,
    'x':       'Language',
    'y':       'accuracy',
    'hue':     'Word Entropy',
    'width':    0.75,
    'fliersize': 0,
    'linewidth': 0.5}



with sns.plotting_context(RC):

    ax = sns.pointplot(**plotting_parameters)
    sns.despine()
    
    x1, x2 = -0.2, 0.2   # columns 
    y, h, col = 0.00022, 0.000005, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    plt.show()
    
    x1, x2 = 0.8, 1.2   # columns 
    y, h, col = 0.00022, 0.000005, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    plt.show()
    
    
    ax.yaxis.set_major_locator(MaxNLocator(4))

    plt.tick_params(axis='x', which='major', labelsize=8)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    
    # plt.tight_layout()

    plt.show()
    
    #interaction plot
    # sns.pointplot(**plotting_parameters)
    # plt.legend(loc='lower right')  # Default is upper right, which overlaps the data here.
    # plt.show()