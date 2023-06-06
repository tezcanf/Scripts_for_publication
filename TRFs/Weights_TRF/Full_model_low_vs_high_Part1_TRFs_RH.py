#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:58:38 2021

@author: filtsem
"""

from pathlib import Path
import numpy as np
import eelbrain
import re
import utils
import os

ANOVA_results_path =  '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Full_model_high_vs_low/'


x_names =   ['gammatone_low',
             'gammatone_high',
             'gammatone_on_low',
             'gammatone_on_high',
             'phonemes_low',
             'phonemes_high',
             'cohort_surprisal_low',
             'cohort_surprisal_high',
             'cohort_entropy_low',
             'cohort_entropy_high',
             'word_freq_log_low',
             'word_freq_log_high']

language = 'French'
region = 'Whole_brain'

DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
SUBJECTS.sort() 

models = 'equal_signal_acoustic+phonemes+surprisal+entropy_all_rh'


rows = []

for subject in SUBJECTS:
    print(subject)
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])

Data_French_low = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)


rows = []

for subject in SUBJECTS:
    print(subject)
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])

Data_French_high = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)



language = 'Dutch'

DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language
TRF_DIR = DATA_ROOT / TRF_folder_name

SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]

rows = []

for subject in SUBJECTS:
    print(subject)
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])

Data_Dutch_low = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)



rows = []

for subject in SUBJECTS:
    print(subject)
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])

Data_Dutch_high = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)



entropy = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_interaction_source.npy'), allow_pickle=True)  

        
utils.line_graph1((Data_French_low["phonemes_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_French_low["cohort_surprisal_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_French_low["cohort_entropy_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3, (Data_French_high["phonemes_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+ Data_French_high["cohort_surprisal_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+ Data_French_high["cohort_entropy_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,(Data_Dutch_low["phonemes_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_low["cohort_surprisal_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_low["cohort_entropy_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,(Data_Dutch_high["phonemes_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_high["cohort_surprisal_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_high["cohort_entropy_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Phoneme Features RH', entropy, langauge,interaction)  
        
        
entropy = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_interaction_source.npy'), allow_pickle=True)           
utils.line_graph1(Data_French_low["gammatone_on_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'), Data_French_high["gammatone_on_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),Data_Dutch_low["gammatone_on_low"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),Data_Dutch_high["gammatone_on_high"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Acoustic Edges RH', entropy, langauge,interaction)                             

