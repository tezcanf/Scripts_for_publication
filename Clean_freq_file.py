#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:33:32 2021

@author: filtsem

This script filters the word freq file to take out the words 
which occured less then 2 times and the words which has non alphanumeric characters.
"""

import os
from os import path, mkdir
import json, pickle
import pandas as pd
import numpy as np


# Set the root folder
dir_path = os.path.dirname(os.path.realpath(__file__))
language='Dutch' # or 'Dutch'
    
dict_folder = os.path.join(dir_path,  'raw_data', language) 

if language=='Dutch':
    '''Read the Subtlex file for word frequencies. '''
    
    ## First filter the file 
    freq_filename='SUBTLEX-NL.csv' 
    freq_file_path = os.path.join(dict_folder, freq_filename)
    df_freq =pd.read_csv(freq_file_path,sep=';')

    number_of_words=len(df_freq['Word'])

    for i in range(number_of_words):
        if min(sum(not c.isalpha() and not c=='\''for c in str(df_freq['Word'][i])), 1) :
            df_freq=df_freq.drop(i)   

    df_freq = df_freq.reset_index(drop=True)    
    output_filename = 'SUBTLEX-NL_filtered_2022.csv'
    
    with open(os.path.join(dict_folder, output_filename), 'w') as outfile:
        df_freq.to_csv(outfile,index=False, encoding='utf8', sep=';', line_terminator='\n')   
            
    ### Then cut it at the same length with French file
    freq_filename='SUBTLEX-NL_filtered_2022.csv'  
    freq_file_path = os.path.join(dict_folder, freq_filename)
    df_freq =pd.read_csv(freq_file_path,sep=';')
    number_of_words=len(df_freq['Word'])     
    for i in range(number_of_words):
        if i > 92108: #length of French frequency file
            df_freq=df_freq.drop(i)
            
    df_freq = df_freq.reset_index(drop=True)    
    output_filename = 'SUBTLEX-NL_filtered_2022_cut.csv'
    
    with open(os.path.join(dict_folder, output_filename), 'w') as outfile:
        df_freq.to_csv(outfile,index=False, encoding='utf8', sep=';', line_terminator='\n')       

if language=='French':
    import locale
    locale.setlocale(locale.LC_ALL, 'French')
    '''Read the Subtlex file for word frequencies. '''
    freq_filename='Lexique383.csv' 
    freq_file_path = os.path.join(dict_folder, freq_filename)
    df_freq =pd.read_csv(freq_file_path,sep=';', encoding='latin1') 
    
    number_of_words=len(df_freq['1_ortho'])
    for i in range(number_of_words):
        if min(sum(not c.isalpha() and not c=='\''for c in str(df_freq['1_ortho'][i])), 1) or (float(df_freq['9_freqfilms2'][i])==0) : 
            #print(df_freq['Word'][i])
            df_freq=df_freq.drop(i)   


    df_freq = df_freq.reset_index(drop=True)  
    
    #Delete NaN values
    NaNs = df_freq['1_ortho'].isnull()
    Nans_index = np.where(NaNs==True)[0]
    for i in Nans_index:
        df_freq=df_freq.drop(i)  
    df_freq = df_freq.reset_index(drop=True) 
    
    list_of_words=df_freq['1_ortho'].tolist()
    unique_list_of_words=list(set(list_of_words))
    
    i=0            
    while i<(len(df_freq['1_ortho'])-1):
        curr_word=df_freq['1_ortho'][i]

        i=i+1

        ''' Merge the same words in one row'''
        if df_freq['1_ortho'][i]==curr_word:
            df_freq.at[(i-1),'8_freqlemlivres']=float(df_freq['8_freqlemlivres'][i-1])+float(df_freq['8_freqlemlivres'][i])
            df_freq.at[(i-1),'7_freqlemfilms2']=float(df_freq['7_freqlemfilms2'][i-1])+float(df_freq['7_freqlemfilms2'][i])
            df_freq.at[(i-1),'9_freqfilms2']=float(df_freq['9_freqfilms2'][i-1])+float(df_freq['9_freqfilms2'][i])
            df_freq.at[(i-1),'10_freqlivres']=float(df_freq['10_freqlivres'][i-1])+float(df_freq['10_freqlivres'][i]) 
            df_freq=df_freq.drop(i)
            df_freq=df_freq.reset_index(drop=True) 
            i=i-1
            
 
    output_filename = 'Lexique383_filtered_2022.csv'            

        
    with open(os.path.join(dict_folder, output_filename), 'w') as outfile:
        df_freq.to_csv(outfile,index=False, encoding='latin1', sep=';', line_terminator='\n')              
                
    

    
    
    