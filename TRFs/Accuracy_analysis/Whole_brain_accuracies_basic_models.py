#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:58:38 2021

@author: filtsem

This script reads accuracies from Eelbrain output files, takes the average overall all brain sources after smoothing, combined them in pd sturcture and writes to a csv file
"""

from pathlib import Path
import numpy as np
import eelbrain
import re
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
import scipy

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
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.figsize': (4,3),
    'axes.prop_cycle':colors}


models = ['acoustic',
          'acoustic+phonemes',
          'acoustic+phonemes+surprisal',
          'acoustic+phonemes+surprisal+entropy',
          'acoustic+phonemes+surprisal+entropy+freq',]



#%%
language = 'French'
region = 'Whole_brain'

DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
         
rows = []
for model in models:
    for subject in SUBJECTS:
        print(subject)
        trf_lh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_lh.pickle')
        trf_rh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_rh.pickle')
        rows.append(['French', subject, model, np.mean([trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source'), trf_rh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])])

language = 'Dutch'

DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language
TRF_DIR = DATA_ROOT / TRF_folder_name

SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]


for model in models:
    for subject in SUBJECTS:
        print(subject)
        trf_lh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_lh.pickle')
        trf_rh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_rh.pickle')
        rows.append(['Dutch', subject, model, np.mean([trf_lh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source'), trf_rh.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])])

                
               
df = pd.DataFrame(data = rows, columns = ['Language',  'subject','model', 'accuracy'])



df.to_csv(os.path.join(Results_path,"Accuracies_Whole_brain_basic_part1.csv"))


df = pd.read_csv(os.path.join(Results_path,"Accuracies_Whole_brain_basic_part1.csv"))

Index = df['Language']=='Dutch'
df_language = df[Index]
rows =[]
subjects=list(set(df_language['subject'].to_list()))
subjects.sort()

a=(np.vstack((subjects,['Phoneme onset']*24,(df_language[df_language['model']=='acoustic+phonemes']['accuracy'].reset_index(drop=True)-df_language[df_language['model']=='acoustic']['accuracy'].reset_index(drop=True)).to_numpy())).T)
b=(np.vstack((subjects,['Phoneme surprisal']*24,(df_language[df_language['model']=='acoustic+phonemes+surprisal']['accuracy'].reset_index(drop=True)-df_language[df_language['model']=='acoustic+phonemes']['accuracy'].reset_index(drop=True)).to_numpy())).T)
c=(np.vstack((subjects,['Phoneme entropy']*24,(df_language[df_language['model']=='acoustic+phonemes+surprisal+entropy']['accuracy'].reset_index(drop=True)-df_language[df_language['model']=='acoustic+phonemes+surprisal']['accuracy'].reset_index(drop=True)).to_numpy())).T)
d=(np.vstack((subjects,['Word freq']*24,(df_language[df_language['model']=='acoustic+phonemes+surprisal+entropy+freq']['accuracy'].reset_index(drop=True)-df_language[df_language['model']=='acoustic+phonemes+surprisal+entropy']['accuracy'].reset_index(drop=True)).to_numpy())).T)

rows = np.vstack((a,b,c,d))

df_diff = pd.DataFrame(data = rows, columns = [ 'subject','model', 'accuracy'])
df_diff["accuracy"] = pd.to_numeric(df_diff["accuracy"])

plotting_parameters = {
    'data':    df_diff,
    'x':       'model',
    'y':       'accuracy',
    'width':    0.75,
    'fliersize': 0,
    'linewidth': 0.5,
    'color': 'grey'}

    
with sns.plotting_context(RC):

    ax = sns.boxplot(**plotting_parameters)
    sns.despine()

    x1, x2 = -0.1, 0.1   # columns 
    y, h, col = 0.00015, 0.000002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    plt.show()
    
    x1, x2 = 0.9, 1.1   # columns 
    y, h, col = 0.00015, 0.000002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    plt.show()
    
    x1, x2 = 1.9, 2.1   # columns 
    y, h, col = 0.00015, 0.000002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    plt.show()
    
    x1, x2 = 2.9, 3.1   # columns 
    y, h, col = 0.00015, 0.000002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    plt.show()
    
    plt.axhline(-0.0000001, linestyle='--')
        
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.ylim(-0.000025,0.00020)

    plt.tick_params(axis='x', which='major', labelsize=12)


    plt.show()

