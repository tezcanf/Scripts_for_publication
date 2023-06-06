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

subjects_dir = '/project/3027007.01/processed/'
fname_fsaverage_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                   'fsaverage-ico-4-src.fif')
src_to = mne.read_source_spaces(fname_fsaverage_src)
src_to = mne.add_source_space_distances(src_to)       
    
MEG_source = '/project/3027007.02/Scripts_for_publication/raw_data/Source/' 

story = 'Anderson_S01_P01'

subject = 'sub-003'    
    
stc_filename = subject + '_' + story
stc_filepath = os.path.join(MEG_source,stc_filename)
stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
        
adjacency = mne.spatial_src_adjacency(src_to)
fsave_vertices = stc.vertices

result_folder = '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Dutch_part2/'

models = [ 'equal_signal_acoustic+phonemes+surprisal+entropy_low',
          'equal_signal_acoustic+phonemes+surprisal+entropy_high']

models2 = ['part2_equal_signal_acoustic+phonemes+surprisal+entropy_low',
          'part2_equal_signal_acoustic+phonemes+surprisal+entropy_high',]

#%%
TRF_nos = [0,1,2]
TRF_names = ['phoneme_onset', 'surprisal', 'entropy']
ANOVA_tests = ['A', 'B', 'A:B']
ANOVA_test_names = ['language','entropy', 'interaction']
All_Y = []
for i in TRF_nos: 
    print(TRF_names[i])
    
    language = 'French'
    region = 'Whole_brain'   
    DATA_ROOT = Path(
        "/project/3027007.02/Scripts_for_publication/").expanduser()
    TRF_folder_name = 'TRFs_' + region+'_'+language
    
    TRF_DIR = DATA_ROOT / TRF_folder_name
    SUBJECTS = [path.name for path in TRF_DIR.iterdir()
                if re.match(r'sub\d*', path.name)]
    SUBJECTS.sort() 
    
    # Where to save the figure
    DST = DATA_ROOT / 'figures'
    DST.mkdir(exist_ok=True)
    
    rows_all = []
    for model in models:
        rows = []
        for subject in SUBJECTS:
            trf_lh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_lh.pickle')
            trf_rh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_rh.pickle')
            print(np.shape(trf_lh.h[2].x.T))
            trf_lh.proportion_explained.source._subjects_dir = subjects_dir
            trf_rh.proportion_explained.source._subjects_dir = subjects_dir
            rows.append(np.concatenate((trf_lh.h[i].square().sqrt().smooth('source', window_size=0.014, window='gaussian').x.T,trf_rh.h[i].square().sqrt().smooth('source', window_size=0.014, window='gaussian').x.T),1))

        rows_all.append(np.array(rows,dtype = float))
    
    language = 'Dutch'
    region = 'Whole_brain'   
    DATA_ROOT = Path(
        "/project/3027007.02/Scripts_for_publication/").expanduser()
    TRF_folder_name = 'TRFs_' + region+'_'+language
    
    TRF_DIR = DATA_ROOT / TRF_folder_name
    SUBJECTS = [path.name for path in TRF_DIR.iterdir()
                if re.match(r'sub\d*', path.name)]
    SUBJECTS.sort() 
    
    
    for model in models2:
        rows = []
        for subject in SUBJECTS:
            trf_lh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_lh.pickle')
            trf_rh = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {model}_rh.pickle')
            print(np.shape(trf_lh.h[2].x.T))
            trf_lh.proportion_explained.source._subjects_dir = subjects_dir
            trf_rh.proportion_explained.source._subjects_dir = subjects_dir
            rows.append(np.concatenate((trf_lh.h[i].square().sqrt().smooth('source', window_size=0.014, window='gaussian').x.T,trf_rh.h[i].square().sqrt().smooth('source', window_size=0.014, window='gaussian').x.T),1))

            
        rows_all.append(np.array(rows,dtype = float))
        
    
    print('Computing adjacency.')

    X = np.array(rows_all,dtype = float)      
    print(np.shape(X))    
    Y = X[:,:,2:-2,:]
    print(np.shape(Y))  
    All_Y.append(Y)
    

    
Y =  All_Y[0] + All_Y[1] + All_Y[2]
 
with open(os.path.join(result_folder,'Y_all_phonemes_2_voxel_smoothed_2022_equal_signal_freq_part2_4_stories.pickle'), 'wb') as f:
    pickle.dump(Y, f)   

for  (ANOVA_test, ANOVA_test_name) in zip(ANOVA_tests,ANOVA_test_names):
    print(ANOVA_test, ANOVA_test_name)

    ### Interaction cluster statistics     
    factor_levels = [2, 2]
    
    effects = ANOVA_test
    # Tell the ANOVA not to compute p-values which we don't need for clustering
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
        spatio_temporal_cluster_test(Y, adjacency=adjacency, n_jobs=-1, buffer_size=10000,
                                      threshold=threshold_tfce, stat_fun=stat_fun,
                                      n_permutations=n_permutations,
                                      out_type= 'indices' )

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    
    with open(os.path.join(result_folder,'clu_all_phonemes_'+ANOVA_test_name+'_2_voxel_smoothed_2022_equal_signal_freq_part2_4_stories.pickle'), 'wb') as f:
        pickle.dump(clu, f)



