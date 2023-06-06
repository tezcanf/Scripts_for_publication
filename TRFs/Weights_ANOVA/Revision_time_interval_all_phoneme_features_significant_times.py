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

result_folder = '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Time_interval/'

#%%

final_report =[]
TRF_names = ['Acoustic','Phoneme_onset', 'Surprisal', 'Entropy']

for feature in TRF_names: 
    
    file_name = os.path.join(result_folder,'Y_' + feature + '_time_interval.pickle')
    with open(file_name, 'rb') as f:
         Y = pickle.load(f)
    
    ANOVA_test_names = ['language','entropy', 'interaction']
    for effect in ANOVA_test_names:
        file_name = os.path.join(result_folder,'clu_' + feature +'_' + effect + '_time_interval.pickle')
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
            if np.where(clusters[i]==True)[1][0] < 1:
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
        
        n_times = np.arange(-0.05,0.8,0.05)
        cluster_size_threshold = 0
        
        sig_time_left = []
        sig_time_right = []
        for i in range(len(n_times)):
            if len(np.where(times_left==i)[0]) > cluster_size_threshold:
                sig_time_left.append(i)
            if len(np.where(times_right==i)[0]) > cluster_size_threshold:
                sig_time_right.append(i)

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
