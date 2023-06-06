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
import os
import mne
import pandas as pd
from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,  f_mway_rm)
from mne import spatial_src_adjacency
import pickle 
import matplotlib.pyplot as plt


result_folder = '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Time_interval/'



#%%


TRF_nos = [0,1,2]
TRF_names = ['Phoneme_onset', 'Surprisal', 'Entropy']


ANOVA_tests = ['A', 'B', 'A:B']
ANOVA_test_names = ['language','entropy', 'interaction']

Results_path = '/project/3027007.02/Scripts_for_publication/TRFs/Accuracy_analysis/Accuracies/Revision/'

df = pd.read_csv(os.path.join(Results_path,'Accuracies_both_hemispheres_word_entropy_French_Dutch_revision_suggestion_time_interval_with_acoustic.csv'))

language = 'Dutch'
region = 'Whole_brain' #'Whole_brain'
# Data locations
DATA_ROOT = Path(
    "/project/3027007.02/Scripts_for_publication/").expanduser()
TRF_folder_name = 'TRFs_' + region+'_'+language

TRF_DIR = DATA_ROOT / TRF_folder_name
SUBJECTS = [path.name for path in TRF_DIR.iterdir()
            if re.match(r'sub\d*', path.name)]
SUBJECTS.sort()
    
for i in TRF_nos: 
    feature = TRF_names[i]
    rows_all = []
    
    languages = ['French', 'Dutch']
    models = ['Low', 'High']
   
    for language in languages:  
        for model in models: 
            subject_rows = []
            for subject in SUBJECTS:
                rows = []
                for t in range(17):                   
                    x = np.concatenate((df.loc[(df['model'] == feature) & (df['Language'] == language)  & (df['Word Entropy'] == model)  & (df['hemisphere'] == 'Left')  & (df['subject'] == subject) & (df['time'] == t)]['accuracy'].to_numpy(), df.loc[(df['model'] == feature) & (df['Language'] == language)  & (df['Word Entropy'] == model)  & (df['hemisphere'] == 'Right') & (df['subject'] == subject)& (df['time'] == t)]['accuracy'].to_numpy()))
                    rows.append(x) 
                subject_rows.append(rows)

            rows_all.append(subject_rows)
    print(np.shape(rows_all))
    
    Y = np.array(rows_all,dtype = float)    
    # print(dur)
    with open(os.path.join(result_folder,'Y_'+TRF_names[i]+'_time_interval.pickle'), 'wb') as f:
        pickle.dump(Y, f)
        
    for  (ANOVA_test, ANOVA_test_name) in zip(ANOVA_tests,ANOVA_test_names):
        print(ANOVA_test, ANOVA_test_name)
    
        ### Interaction cluster statistics     
        factor_levels = [2, 2]
        
        effects = ANOVA_test
        return_pvals = False
                
        n_conditions = 4
        n_subjects = 24
                
        def stat_fun(*args):
            # get f-values only.
            return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                              effects=effects, return_pvals=return_pvals)[0]
        
    
        pthresh = 0.025
        threshold_tfce = dict(start=0, step=0.1)
        f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)
        
        n_permutations = 8000  
        
        print('Clustering.')
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_test(Y, adjacency=None, n_jobs=-1, buffer_size=10000,
                                          threshold=threshold_tfce, stat_fun=stat_fun,
                                          n_permutations=n_permutations,
                                          out_type= 'mask' )
            
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        
        with open(os.path.join(result_folder,'clu_'+TRF_names[i]+'_'+ANOVA_test_name+'_time_interval.pickle'), 'wb') as f:
            pickle.dump(clu, f)


