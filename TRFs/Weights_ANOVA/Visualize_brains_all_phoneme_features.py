#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:58:38 2021

@author: filtsem
"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot
import os
import mne
import pandas as pd
import pickle 
import matplotlib


subjects_dir = '/Volumes/project/3027007.01/processed/'
fname_fsaverage_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                   'fsaverage-ico-4-src.fif')
src_to = mne.read_source_spaces(fname_fsaverage_src)
src_to = mne.add_source_space_distances(src_to)       

output_path = '/Volumes/project/3027007.02/Scripts_for_publication/raw_data/Source' 

story = 'Anderson_S01_P01'

subject = 'sub-003'    
    
stc_filename = subject + '_' + story
stc_filepath = os.path.join(output_path,stc_filename)
stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
        
fsave_vertices = stc.vertices

# Configure the matplotlib figure style
FONT = 'Helvetica Neue'
FONT_SIZE = 8
RC = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
}
pyplot.rcParams.update(RC)
# -

# 
result_folder = '/Volumes/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Dutch_part1/'

# result_folder = '/Volumes/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Full_model_high_vs_low/'


  
#%%

final_report =[]
TRF_names = ['acoustic_edge','phoneme_onset', 'surprisal', 'all_phonemes', 'entropy']


# final_report =[]
# TRF_names = ['all_phonemes', 'acoustic_edge']

for feature in TRF_names: 
    
    file_name = os.path.join(result_folder,'Y_' + feature + '_2_voxel_smoothed_2022_equal_signal_freq_part1_4_stories.pickle')
    with open(file_name, 'rb') as f:
         Y = pickle.load(f)
    
    ANOVA_test_names = ['language','entropy', 'interaction']
    for effect in ANOVA_test_names:
        file_name = os.path.join(result_folder,'clu_' + feature +'_' + effect + '_2_voxel_smoothed_2022_equal_signal_freq_part1_4_stories.pickle')
        image_save_left = os.path.join(result_folder, 'ANOVA_brains','Brain_' + feature + '_' + effect + '_LH.png')
        image_save_right = os.path.join(result_folder, 'ANOVA_brains','Brain_' + feature + '_' + effect + '_RH.png')
        with open(file_name, 'rb') as f:
             clu = pickle.load(f)
              
             
        T_obs, clusters, cluster_p_values, H0 = clu     
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        
        
        right_hemi = []
        left_hemi = []
        right_hemi_pi_values = []
        left_hemi_pi_values = []
        
        for i in good_cluster_inds:
            if np.where(clusters[i]==True)[1][0] < 2562:
                left_hemi.append(i)
                left_hemi_pi_values.append(cluster_p_values[i])
            else:
                right_hemi.append(i)
                right_hemi_pi_values.append(cluster_p_values[i])
        
        if effect == 'interaction':
            #Interaction
            Dutch_contrast = Y[3,:,:,:]-Y[2,:,:,:] #High-Low
            French_contrast = Y[1,:,:,:]-Y[0,:,:,:] #High-Low
            contrast = Dutch_contrast - French_contrast
            contrast_ava = contrast.mean(0) #Averaged over subjects
        
        if effect == 'language':
            #Language main effect
            Dutch_contrast = Y[3:5,:,:,:].mean(0)
            French_contrast = Y[0:2,:,:,:].mean(0)
            contrast = Dutch_contrast - French_contrast
            contrast_ava = contrast.mean(0) #Averaged over subjects
        
        if effect == 'entropy':
            #Entropy main effect
            High_contrast = (Y[3,:,:,:]+Y[1,:,:,:])/2 #Dutch-French avarage
            Low_contrast = (Y[2,:,:,:]+Y[0,:,:,:])/2 #Dutch-French avarage
            contrast = High_contrast -  Low_contrast 
            contrast_ava = contrast.mean(0) #Averaged over subjects
            
                
        
        def brain_graph(Contrast_masked,tmin,tmax, hemi,views):
        
        
            t_init = -0.1
              
            Contrast_masked_mean = Contrast_masked[tmin:tmax,:].mean(0).T #Mean over time interval
            
            max_val = np.abs(Contrast_masked_mean).max() 
            
            Contrast_masked_scaled = Contrast_masked_mean / max_val * 100
        
            colors = [(0, 0, 0)] + [(matplotlib.cm.coolwarm(i)) for i in range(1, 256)]
            new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N = 256)
        
            tmin_corr = (tmin*0.01+t_init)
            stc_mean = mne.source_estimate.SourceEstimate(Contrast_masked_scaled,vertices=fsave_vertices, tmin=tmin_corr, tstep=0.01, subject='fsaverage')
        
            brain = mne.viz.plot_source_estimates(stc_mean,subject='fsaverage',hemi=hemi,smoothing_steps=10,subjects_dir=subjects_dir,spacing='ico4',
                                          background='white',colormap=new_cmap,views=views, surface='white', size=(1711,1044),
                                          clim=dict(kind='value', pos_lims=[0, 50, 100]))
            
            
            return brain, Contrast_masked_mean
        
        # Visualize the clusters
        # ----------------------
        
        tstep = 0.01
        tmin = -0.1
        print('Visualizing clusters.')
        
        
        left_clusters = np.zeros_like(clusters[0])
        right_clusters = np.zeros_like(clusters[0])
        
        
        for i in left_hemi:
            left_clusters = left_clusters + clusters[i]
        
        for i in right_hemi:
            right_clusters = right_clusters + clusters[i]
        
        Contrast_masked_left = np.multiply(contrast_ava,(left_clusters)) 
        Contrast_masked_right = np.multiply(contrast_ava,(right_clusters)) 
        
        
        
        times_left = np.where((left_clusters) ==True)[0]  #Put good cluster index 
        times_right = np.where((right_clusters) ==True)[0]  #Put good cluster index 
        
        n_times = np.arange(-0.1,0.8,0.01)
        cluster_size_threshold = 0
        
        sig_time_left = []
        sig_time_right = []
        for i in range(len(n_times)):
            if len(np.where(times_left==i)[0]) > cluster_size_threshold:
                sig_time_left.append(i)
            if len(np.where(times_right==i)[0]) > cluster_size_threshold:
                sig_time_right.append(i)
                
        if len(sig_time_left)>0:
            Brain_left, Contrast_masked_mean_left = brain_graph(Contrast_masked_left,sig_time_left[0],sig_time_left[-1]+1, 'lh', 'lateral') #Check from np.where(clusters[0]==True)[0] if it's RH or LH
            Brain_left.save_image(image_save_left)
        if len(sig_time_right)>0:
            Brain_right, Contrast_masked_mean_right = brain_graph(Contrast_masked_right,sig_time_right[0],sig_time_right[-1]+1, 'rh', 'lateral') #Check from np.where(clusters[0]==True)[0] if it's RH or LH
            Brain_right.save_image(image_save_right)
        
        
        
        save_name_LH = feature + '_LH_' + effect +'_source.npy'
        save_name_RH = feature + '_RH_' + effect +'_source.npy'
        
        save = np.ones(len(n_times))
        
        if len(sig_time_left) > 0:
            for i in range(len(n_times)):
                if not i in sig_time_left:              
                    save[i] = np.NaN
                    
            np.save(os.path.join(result_folder,save_name_LH), save, allow_pickle=True)        
              
        else:
            # no cluster
            for i in range(len(n_times)): 
                save[i] = np.NaN     
            np.save(os.path.join(result_folder,save_name_LH), save, allow_pickle=True)     
                
        save = np.ones(len(n_times))
        
        if len(sig_time_right) > 0:
            for i in range(len(n_times)):
                if not i in sig_time_right:             
                    save[i] = np.NaN
                    
            np.save(os.path.join(result_folder,save_name_RH), save, allow_pickle=True) 
            
        else:
            # no cluster
            for i in range(len(n_times)): 
                save[i] = np.NaN     
            np.save(os.path.join(result_folder,save_name_RH), save, allow_pickle=True)     
        
        
        final_report_line = [feature, effect, sig_time_left, sig_time_right, left_hemi_pi_values, right_hemi_pi_values]
        final_report.append(final_report_line)

df = pd.DataFrame(data = final_report, columns = ['feature','effect',  'sig_time_left','sig_time_right', 'left_hemi_pi_values', 'right_hemi_pi_values'])
df.to_csv(os.path.join(result_folder,'Report.csv'))
