#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:51:41 2021

@author: filtsem
"""
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.minimum_norm import read_inverse_operator, apply_inverse, apply_inverse_raw, make_inverse_operator,apply_inverse, write_inverse_operator
import pandas as pd 
from mne.io import read_raw_ctf
from mne.datasets import fetch_fsaverage
from functools import reduce 
from mne import pick_types

def make_source_space(subject, language, region, story, DATA_ROOT, tmax):
    
    
    stim_path = DATA_ROOT / 'stimuli' / language    
    processed_MEG_folder = DATA_ROOT / 'raw_data' / 'MEG' / 'raw' / 'ICA_600Hz' 
    subjects_dir = '/project/3027007.01/processed/'
    output_path_Dutch = DATA_ROOT / 'raw_data' / 'Source'
    output_path_French = DATA_ROOT / 'raw_data' / 'Source'
    fname_fsaverage_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                   'fsaverage-ico-4-src.fif')
    Sources_saved = True # If the source recontruction is already done and files are saved. 
    
    def story_to_triggers(story,stim_path):
        "Returns trigger onset and offset values"
        stim_map = pd.read_csv(os.path.join(stim_path,'story_ids_filiz.csv'), delim_whitespace=1, names=['id', 'file', 'lang'])
        
        sid = str(stim_map.loc[stim_map.file == story  +'_normalized.wav', 'id'].to_list()[0])
        return int('1' + sid), int('2' + sid)
    
    def load_data_per_story(raw, events, story, stim_path,new_fs=100):
        start, end = story_to_triggers(story,stim_path)
        print('raw fs: '+ str(raw.info['sfreq']))
        tstart = events[np.argwhere(events[:, 2]==start), 0]/raw.info['sfreq']
        tend = events[np.argwhere(events[:, 2]==end), 0]/raw.info['sfreq']
        print(tstart)
        print(tend)
        raw_cropped = raw.copy().crop(tmin=tstart.squeeze()-0.1, tmax=tend.squeeze()+tmax,)  # avoid inplace modification
        raw_cropped.load_data()
        
        return raw_cropped
    
    labels = mne.read_labels_from_annot(subject='fsaverage', parc = 'aparc.a2009s', subjects_dir=subjects_dir) # 'aparc' 
    labels_name = [l.name for l in labels]
    

    if region == 'STG':
        search_lh = np.array(['G_temp_sup-G_T_transv-lh', 'G_temp_sup-Lateral-lh', 'G_temp_sup-Plan_polar-lh', 'G_temp_sup-Plan_tempo-lh'])
        label_index_lh = [labels_name.index(l) for l in search_lh]
                
        search_rh = np.array(['G_temp_sup-G_T_transv-rh', 'G_temp_sup-Lateral-rh', 'G_temp_sup-Plan_polar-rh', 'G_temp_sup-Plan_tempo-rh'])

        label_index_rh = [labels_name.index(l) for l in search_rh]
        
        lh_label_list = [labels[i] for i in label_index_lh]
        rh_label_list = [labels[i] for i in label_index_rh]
        
        stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
        stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
                
    if region == 'IFG':
        search_lh = np.array(['G_front_inf-Opercular-lh', 'G_front_inf-Orbital-lh', 'G_front_inf-Triangul-lh'])
        
        label_index_lh = [labels_name.index(l) for l in search_lh]
                
        search_rh = np.array(['G_front_inf-Opercular-rh', 'G_front_inf-Orbital-rh', 'G_front_inf-Triangul-rh'])
        
        label_index_rh = [labels_name.index(l) for l in search_rh]
        
        lh_label_list = [labels[i] for i in label_index_lh]
        rh_label_list = [labels[i] for i in label_index_rh]
        
        stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
        stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
                
    if region == 'Whole_brain':

        search_lh = [f for f in labels_name if f.endswith('-lh')]
        
        label_index_lh = [labels_name.index(l) for l in search_lh]
                   
        search_rh = [f for f in labels_name if f.endswith('-rh')]   
                
        label_index_rh = [labels_name.index(l) for l in search_rh]
        
        lh_label_list = [labels[i] for i in label_index_lh]
        rh_label_list = [labels[i] for i in label_index_rh]
        
        stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
        stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
    
    if Sources_saved == False:
           
        # The transformation file obtained by coregistration
        trans = os.path.join(subjects_dir, subject, 'meg', subject + '-trans.fif')
        
    
        #Read raw file
        raw_file_name = os.path.join(processed_MEG_folder, '%s_resampled_600Hz-ICA-raw.fif' %(subject))
        raw = mne.io.read_raw_fif(raw_file_name, preload=True)
        info = raw.info 
        
        src = mne.setup_source_space(subject, spacing='ico4', #add_dist='patch',
                                      subjects_dir=subjects_dir) 
        src = mne.add_source_space_distances(src)  
    
        
        conductivity = (0.3,)  # for single layer
        model = mne.make_bem_model(subject=subject, 
                                    conductivity=conductivity,
                                    subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        
        fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                        meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                        verbose=True)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, copy=True, use_cps=True, verbose=None)
    
    
            # Find events and pick the data for Dutch stories
        ch_types = {
                    # EOG
                    'EEG057-4302':'eog', 
                    'EEG058-4302':'eog',
                    # EKG
                    'EEG059-4302':'ecg',
                    # STIM - audio
                    #'UADC003':'stim',
                    #'UADC004':'stim',
                    # Triggers
                    'UPPT001':'stim',
                    #'UPPT002':'resp', # response
                        }
        raw = raw.set_channel_types(ch_types)    
        events = mne.find_events(raw,shortest_event=1)
        
        event_id = [111, 121, 131, 141, 112, 122, 132, 113, 123, 114, 124, 115, 116]
            
        baseline = None
        reject = dict(mag=4e-12)
        picks = pick_types(raw.info, meg=True,ref_meg=False, eeg=False, eog=False,
                                    stim=False)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-10, tmax=0, preload=True,baseline=baseline, picks=picks)     #, event_repeated='drop'
        
        # Computing covariance matrix.
        noise_cov = mne.compute_covariance(epochs, tmax=0., method=  'empirical', rank=None, verbose=True)
    
        inverse_operator = make_inverse_operator(info, fwd, noise_cov, fixed = True ) 
        
        
        
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
        
        raw = raw.pick_types(meg=True, ref_meg=False)

        src_to = mne.read_source_spaces(fname_fsaverage_src)
        src_to = mne.add_source_space_distances(src_to)   
        
        if language == 'Dutch':
            s = story
    
            # Source reconstruction of Dutch Stories
    
            print(s)    
            raw_cropped = load_data_per_story(raw, events, s, stim_path) 
            raw_cropped = raw_cropped.filter(l_freq=None, h_freq=8)
            
            stc = apply_inverse_raw(raw_cropped, inverse_operator, lambda2, method)
            
    
            morph = mne.compute_source_morph(stc, subject_from=subject,
                                      subject_to='fsaverage', src_to=src_to,
                                      subjects_dir=subjects_dir)
            
            
            del raw_cropped
            
            stc =  morph.apply(stc)
            
            stc = stc.resample(100) 
            
            stc_filename = subject + '_' + story
            stc_filepath = os.path.join(output_path_Dutch,stc_filename)
            stc.save(stc_filepath)
            
            stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
            
            stc_lh = stc.in_label(stc_lh_merged_label) 
            print(stc_lh)
    
            stc_rh = stc.in_label(stc_rh_merged_label) 
            print(stc_rh)

            del stc
    
            
            if s=='Anderson_S01_P01':
                crop=len(stc_lh._data.T)-29889
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P02':
                crop=len(stc_lh._data.T)-31859
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P03':
                crop=len(stc_lh._data.T)-28989
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P04':
                crop=len(stc_lh._data.T)-35169
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='grimm_20_1':
                crop=len(stc_lh._data.T)-36748
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                                    
            if s=='grimm_20_2':
                crop=len(stc_lh._data.T)-40144
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
    
                
            if s=='grimm_23_1':
                crop=len(stc_lh._data.T)-30398
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
    
            if s=='grimm_23_2':
                crop=len(stc_lh._data.T)-33376
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                    
            if s=='grimm_23_3':
                crop=len(stc_lh._data.T)-30381
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
           
        
        if language == 'French':
            s = story
    
            # Source reconstruction of Dutch Stories
    
            print(s)    
            raw_cropped = load_data_per_story(raw, events, s, stim_path) 
            raw_cropped = raw_cropped.filter(l_freq=None, h_freq=8)
            
            stc = apply_inverse_raw(raw_cropped, inverse_operator, lambda2, method)
            
    
            morph = mne.compute_source_morph(stc, subject_from=subject,
                                      subject_to='fsaverage', src_to=src_to,
                                      subjects_dir=subjects_dir)
            
            
            del raw_cropped
            
            stc =  morph.apply(stc)
            
            stc = stc.resample(100) 
            
            stc_filename = subject + '_' + story
            stc_filepath = os.path.join(output_path_French,stc_filename)
            stc.save(stc_filepath)
            
            stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
            
            stc_lh = stc.in_label(stc_lh_merged_label) 
            print(stc_lh)
    
            stc_rh = stc.in_label(stc_rh_merged_label) 
            print(stc_rh)
            
    
            del stc
    
                    
            if s=='ANGE_part1':
                crop=len(stc_lh._data.T)-27491
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='BALL_part1':
                crop=len(stc_lh._data.T)-29890
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='EAUV_part1':
                crop=len(stc_lh._data.T)-34471
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='EAUV_part2':
                crop=len(stc_lh._data.T)-36242
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                    
    else:
        if language == 'Dutch':
            s = story
            
            stc_filename = subject + '_' + story
            stc_filepath = os.path.join(output_path_Dutch,stc_filename)
            
            stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
            
            stc_lh = stc.in_label(stc_lh_merged_label) 
            print(stc_lh)
    
            stc_rh = stc.in_label(stc_rh_merged_label) 
            print(stc_rh)

            del stc
    
            
            if s=='Anderson_S01_P01':
                crop=len(stc_lh._data.T)-29889 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P02':
                crop=len(stc_lh._data.T)-31859 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P03':
                crop=len(stc_lh._data.T)-28989 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='Anderson_S01_P04':
                crop=len(stc_lh._data.T)-35169#+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='grimm_20_1':
                crop=len(stc_lh._data.T)-36748 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                                    
            if s=='grimm_20_2':
                crop=len(stc_lh._data.T)-40144 # +int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
    
                
            if s=='grimm_23_1':
                crop=len(stc_lh._data.T)-30398 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
    
            if s=='grimm_23_2':
                crop=len(stc_lh._data.T)-33376 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                    
            if s=='grimm_23_3':
                crop=len(stc_lh._data.T)-30381 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
           
        
        if language == 'French':
            
            s = story

            stc_filename = subject + '_' + story
            stc_filepath = os.path.join(output_path_French,stc_filename)
            
            stc = mne.read_source_estimate(stc_filepath, subject='fsaverage')
            
            stc_lh = stc.in_label(stc_lh_merged_label) 
            print(stc_lh)
    
            stc_rh = stc.in_label(stc_rh_merged_label) 
            print(stc_rh)
            
    
            del stc
    
                    
            if s=='ANGE_part1':
                crop=len(stc_lh._data.T)-27491 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='BALL_part1':
                crop=len(stc_lh._data.T)-29890 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='EAUV_part1':
                crop=len(stc_lh._data.T)-34471 #+ int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)
                
            if s=='EAUV_part2':
                crop=len(stc_lh._data.T)-36242 #+int((0.8-tmax)*100)
                if crop!=0:
                    print('Deleting the last bits')
                for delete in range(crop):
                    stc_lh._data=np.delete(stc_lh._data,0,-1)
                    stc_lh._times=np.delete(stc_lh.times,-1)
                    stc_rh._data=np.delete(stc_rh._data,0,-1)
                    stc_rh._times=np.delete(stc_rh.times,-1)        
                    

    return stc_lh, stc_rh
