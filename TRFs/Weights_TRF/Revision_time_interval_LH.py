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
import pickle 


ANOVA_results_path = '/project/3027007.02/Scripts_for_publication/TRFs/Weights_ANOVA/Time_interval/'


feature = 'Phoneme_onset'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      P = pickle.load(f)

feature = 'Surprisal'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      S = pickle.load(f)
     
feature = 'Entropy'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      E = pickle.load(f)
     

utils.stacked_graph_time_interval(P[0,:,:,:].mean(axis=2),S[0,:,:,:].mean(axis=2),E[0,:,:,:].mean(axis=2),['Acoustic ', 'Phoneme Onset ','Phoneme Surprisal ','Phoneme Entropy '], 'French Low Word Entropy')                             
utils.stacked_graph_time_interval(P[1,:,:,:].mean(axis=2),S[1,:,:,:].mean(axis=2),E[1,:,:,:].mean(axis=2),['Acoustic ', 'Phoneme Onset ','Phoneme Surprisal ','Phoneme Entropy '], 'French High Word Entropy')                             
utils.stacked_graph_time_interval(P[2,:,:,:].mean(axis=2),S[2,:,:,:].mean(axis=2),E[2,:,:,:].mean(axis=2),['Acoustic ', 'Phoneme Onset ','Phoneme Surprisal ','Phoneme Entropy '], 'Dutch Low Word Entropy')                             
utils.stacked_graph_time_interval(P[3,:,:,:].mean(axis=2),S[3,:,:,:].mean(axis=2),E[3,:,:,:].mean(axis=2),['Acoustic ', 'Phoneme Onset ','Phoneme Surprisal ','Phoneme Entropy '], 'Dutch High Word Entropy')                             

feature = 'Phoneme_onset'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      Y = pickle.load(f)
       
entropy = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_LH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_LH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_LH_interaction_source.npy'), allow_pickle=True)           
utils.line_graph_time_interval(Y[0,:,:,0], Y[1,:,:,0],Y[2,:,:,0],Y[3,:,:,0],['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Phoneme Onset LH', entropy, langauge,interaction)                             

entropy = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_RH_entropy_source.npy'), allow_pickle=True)    
langauge = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_RH_language_source.npy'), allow_pickle=True)     
interaction = np.load(os.path.join(ANOVA_results_path,'Phoneme_onset_RH_interaction_source.npy'), allow_pickle=True)           
utils.line_graph_time_interval(Y[0,:,:,1], Y[1,:,:,1],Y[2,:,:,1],Y[3,:,:,1],['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Phoneme Onset RH', entropy, langauge,interaction)                             


feature = 'Surprisal'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      Y = pickle.load(f)
     
entropy = np.load(os.path.join(ANOVA_results_path,'Surprisal_LH_entropy_source.npy'), allow_pickle=True)       
langauge = np.load(os.path.join(ANOVA_results_path,'Surprisal_LH_language_source.npy'), allow_pickle=True) 
interaction = np.load(os.path.join(ANOVA_results_path,'Surprisal_LH_interaction_source.npy'), allow_pickle=True)     
utils.line_graph_time_interval(Y[0,:,:,0], Y[1,:,:,0],Y[2,:,:,0],Y[3,:,:,0],['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Cohort Surprisal LH', entropy, langauge,interaction)                          

entropy = np.load(os.path.join(ANOVA_results_path,'Surprisal_RH_entropy_source.npy'), allow_pickle=True)       
langauge = np.load(os.path.join(ANOVA_results_path,'Surprisal_RH_language_source.npy'), allow_pickle=True) 
interaction = np.load(os.path.join(ANOVA_results_path,'Surprisal_RH_interaction_source.npy'), allow_pickle=True)     
utils.line_graph_time_interval(Y[0,:,:,1], Y[1,:,:,1],Y[2,:,:,1],Y[3,:,:,1],['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Cohort Surprisal RH', entropy, langauge,interaction)                          



feature = 'Entropy'
file_name = os.path.join(ANOVA_results_path,'Y_' + feature + '_time_interval.pickle')
with open(file_name, 'rb') as f:
      Y = pickle.load(f)
     
entropy = np.load(os.path.join(ANOVA_results_path,'Entropy_LH_entropy_source.npy'), allow_pickle=True)   
langauge = np.load(os.path.join(ANOVA_results_path,'Entropy_LH_language_source.npy'), allow_pickle=True)       
interaction = np.load(os.path.join(ANOVA_results_path,'Entropy_LH_interaction_source.npy'), allow_pickle=True)      
utils.line_graph_time_interval(Y[0,:,:,0], Y[1,:,:,0],Y[2,:,:,0],Y[3,:,:,0],['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Cohort Entropy LH', entropy, langauge,interaction)     
        
entropy = np.load(os.path.join(ANOVA_results_path,'Entropy_RH_entropy_source.npy'), allow_pickle=True)   
langauge = np.load(os.path.join(ANOVA_results_path,'Entropy_RH_language_source.npy'), allow_pickle=True)       
interaction = np.load(os.path.join(ANOVA_results_path,'Entropy_RH_interaction_source.npy'), allow_pickle=True)      
utils.line_graph_time_interval(Y[0,:,:,1], Y[1,:,:,1],Y[2,:,:,1],Y[3,:,:,1] , ['French Low ', 'French High ','Dutch Low ','Dutch High '], 'Cohort Entropy RH', entropy, langauge,interaction)     
        
