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


ANOVA_results_path = '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Dutch_part2/'


language = 'French'
region = 'Whole_brain'

DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
SUBJECTS.sort() 

models_low = 'equal_signal_acoustic+phonemes+surprisal+entropy_low_lh'
          
models_high = 'equal_signal_acoustic+phonemes+surprisal+entropy_high_lh'


models_low2 = 'part2_equal_signal_acoustic+phonemes+surprisal+entropy_low_lh'
          
models_high2 = 'part2_equal_signal_acoustic+phonemes+surprisal+entropy_high_lh'


rows = []
x_names = None
for subject in SUBJECTS:
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models_low}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])
    x_names = trf.x
Data_French_low = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)


rows = []
x_names = None
for subject in SUBJECTS:
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models_high}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])
    x_names = trf.x
Data_French_high = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)


language = 'Dutch'
DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language
TRF_DIR = DATA_ROOT / TRF_folder_name

SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]


rows = []
x_names = None
for subject in SUBJECTS:
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models_low2}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])
    x_names = trf.x
Data_Dutch_low = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)



rows = []
x_names = None
for subject in SUBJECTS:
    trf = eelbrain.load.unpickle(
        TRF_DIR / subject / f'{subject} {models_high2}.pickle')
    rows.append([subject, trf.proportion_explained, *trf.h])
    x_names = trf.x
Data_Dutch_high = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)

        
entropy = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'all_phonemes_RH_interaction_source.npy'), allow_pickle=True)          
utils.line_graph1((Data_French_low["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_French_low["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_French_low["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3, (Data_French_high["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+ Data_French_high["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+ Data_French_high["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,(Data_Dutch_low["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_low["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_low["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,(Data_Dutch_high["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_high["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source')+Data_Dutch_high["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'))/3,['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Phoneme Features RH', entropy, langauge,interaction)  
        
        
entropy = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'acoustic_edge_RH_interaction_source.npy'), allow_pickle=True)           
utils.line_graph1(Data_French_low["gammatone_on"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'), Data_French_high["gammatone_on"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),Data_Dutch_low["gammatone_on"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),Data_Dutch_high["gammatone_on"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean('frequency').mean(
    'source'),['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Acoustic Edges RH', entropy, langauge,interaction)                             



entropy = np.load(os.path.join(ANOVA_results_path,'phoneme_onset_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'phoneme_onset_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'phoneme_onset_RH_interaction_source.npy'), allow_pickle=True)           
utils.line_graph1(Data_French_low["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'), Data_French_high["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_low["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_high["phonemes"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Phoneme Onset RH', entropy, langauge,interaction)                             

        
entropy = np.load(os.path.join(ANOVA_results_path,'surprisal_RH_entropy_source.npy'), allow_pickle=True)       
langauge = np.load(os.path.join(ANOVA_results_path,'surprisal_RH_language_source.npy'), allow_pickle=True) 
interaction = np.load(os.path.join(ANOVA_results_path,'surprisal_RH_interaction_source.npy'), allow_pickle=True)     
utils.line_graph2(Data_French_low["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'), Data_French_high["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_low["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_high["cohort_surprisal"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Cohort Surprisal RH', entropy, langauge,interaction)                          

     
entropy = np.load(os.path.join(ANOVA_results_path,'entropy_RH_entropy_source.npy'), allow_pickle=True)   
langauge = np.load(os.path.join(ANOVA_results_path,'entropy_RH_language_source.npy'), allow_pickle=True)       
interaction = np.load(os.path.join(ANOVA_results_path,'entropy_RH_interaction_source.npy'), allow_pickle=True)      
utils.line_graph2(Data_French_low["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'), Data_French_high["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_low["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),Data_Dutch_high["cohort_entropy"].square().sqrt().smooth('source', window_size=0.014, window='gaussian').mean(
    'source'),['French Low ', 'French High ','Dutch Low ','Dutch High '], 'TRF Cohort Entropy RH', entropy, langauge,interaction)     
        
