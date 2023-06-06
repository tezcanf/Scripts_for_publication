#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:34:04 2021

@author: filtsem
"""

import pandas as pd
import os
import numpy as np


# Set the root folder
dir_path = os.path.dirname(os.path.realpath(__file__))
language= 'Dutch' #'Dutch' # or 'French'
       

# Set the folder with the dataset

data_folder = os.path.join(dir_path,  'raw_data', language, 'Word_entropy_GPT')
ôutput_folder = os.path.join(dir_path,  'raw_data', language, 'Equal_signal_low_vs_high_entropy')

phoneme_files = [f for f in os.listdir(data_folder) if f.endswith('_freq_corrected.csv')]

total_low = 0
total_high = 0

low_entropy_duration = 0 
high_entropy_duration = 0

phoneme_file = phoneme_files[8]

filename=os.path.join(data_folder, phoneme_file)
df_phonemes =pd.read_table(filename, encoding = "utf-8", sep=';' )
    
word_entropy_GPT = df_phonemes[['word_entropy_GPT','word_number']]
#print(word_entropy_LSTM)
word_entropy_GPT_grouped=word_entropy_GPT.groupby('word_number', as_index=False).mean()
median=word_entropy_GPT_grouped.word_entropy_GPT.median()
entropy_class=np.zeros(len(word_entropy_GPT))


scale =  1.0541

for i in range(len(entropy_class)):
    if word_entropy_GPT.word_entropy_GPT[i]<median*scale:
        entropy_class[i]=-1.0
    elif  word_entropy_GPT.word_entropy_GPT[i]>median*scale:
        entropy_class[i]=1.0
    else:
        entropy_class[i]=0.0

#Count duration
   
Data={'entropy_class':entropy_class}
df_word=pd.DataFrame(data=Data)
df_all=pd.concat([df_phonemes, df_word], axis=1)  

low_entopy_phonemes = len(df_all.loc[df_all.entropy_class==-1.0,'phonemes'])   
high_entopy_phonemes = len(df_all.loc[df_all.entropy_class==1.0,'phonemes'] )   

for i in range(len(df_all['phonemes'])):
    if df_all['entropy_class'][i] ==1:
        high_entropy_duration = high_entropy_duration + (df_all['phoneme_offset'][i] - df_all['phoneme_onset'][i]   )
    if df_all['entropy_class'][i] ==-1:
        low_entropy_duration = low_entropy_duration + (df_all['phoneme_offset'][i]  - df_all['phoneme_onset'][i]   )

print(phoneme_file)        
print('len low duration')
print(low_entropy_duration)  
print('len high duration')
print(high_entropy_duration) 


print('len low entropy')
print(low_entopy_phonemes)  
print('len high entropy')
print(high_entopy_phonemes) 

filename = phoneme_file[0:-4]+'_equal_signal.csv' 
file_path = os.path.join(ôutput_folder,filename)
with open(file_path, 'w') as outfile:
    df_all.to_csv(outfile,index=False, sep=';', line_terminator='\n') 

  
    