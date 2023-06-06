# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:13:20 2021

@author: filtez

This script is to concatanate the phoneme and word transcription files. 
It will align phonemes and words and create new files with a structure like 
(Phoneneme, Phoneme Onset, Phoneme Offset, Word, Word Onset, Word Offset)
"""

import os
import pandas as pd

# Set the root folder
dir_path = os.path.dirname(os.path.realpath(__file__))
language='Dutch' # or 'French'
dict_folder = os.path.join(dir_path,  'raw_data', language) 

#Setting the directories where phoneme and word onset, offset alignments are located.
phoneme_folder_name='Phoneme_transcription_of_stories'
phoneme_folder = os.path.join(dict_folder, phoneme_folder_name)
phoneme_files = [f for f in os.listdir(phoneme_folder) if f.endswith('.csv')]

word_folder_name='Word_transcription_of_stories'
word_folder = os.path.join(dict_folder, word_folder_name)
word_files = [f for f in os.listdir(word_folder) if f.endswith('.csv')]

word_phoneme_folder_name='Word_phoneme_transcription_of_stories'
word_phoneme_folder = os.path.join(dict_folder, word_phoneme_folder_name)

if not os.path.exists(word_phoneme_folder): #Create the folder if not exists
    os.mkdir(word_phoneme_folder)
        
       
for i in range(len(word_files)): #
    final_data=[] #Array for new data structure 
    
    df_phonemes =pd.read_table(os.path.join(phoneme_folder, phoneme_files[i]), encoding = "utf-8", sep=',' , names=('onset', 'offset','phones'))
    df_words =pd.read_table(os.path.join(word_folder, word_files[i]), encoding = "utf-8", sep=',' , names=('onset', 'offset','words'))

    p=0 #initilize phoneme iteration
    for w in range(len(df_words['words'])-1): #
        if not pd.isna(df_words['words'][w]):
            if df_phonemes['onset'][p]==df_words['onset'][w]:
                new_data_row=(df_phonemes['phones'][p], df_phonemes['onset'][p], df_phonemes['offset'][p], df_words['words'][w], df_words['onset'][w], df_words['offset'][w])
                final_data.append(new_data_row)
                p=p+1
                while float(df_phonemes['onset'][p])<float(df_words['offset'][w]):
                    new_data_row=(df_phonemes['phones'][p], df_phonemes['onset'][p], df_phonemes['offset'][p], df_words['words'][w], df_words['onset'][w], df_words['offset'][w])
                    final_data.append(new_data_row)
                    p=p+1
        else:
            p=p+1
                
      
    output_filename = word_files[i][0:-4]+'_phonemes.csv'
    df_final = pd.DataFrame(final_data, columns=['phonemes', 'phoneme_onset','phoneme_offset', 'words', 'word_onset', 'word_offset'])
   
        
    with open(os.path.join(word_phoneme_folder, output_filename), 'w') as outfile:
         df_final.to_csv(outfile,index=False, sep=',', line_terminator='\n')  
    
