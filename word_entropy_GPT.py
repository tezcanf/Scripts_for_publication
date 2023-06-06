# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:57:32 2021

@author: filtez
"""

import os
import random as rnd
import numpy as np
import pickle
import time
import string
import numpy as np
import os, gc, json
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch
from torch.nn import functional as F
import json
import logging
transformers.logging.get_verbosity = lambda: logging.NOTSET

# Set the root folder
language= 'Dutch' #'Dutch' # or 'French'
dir_path = os.path.dirname(os.path.realpath(__file__))
output_folder=os.path.join(dir_path,  'raw_data', language, 'Word_entropy_GPT')

if language == 'Dutch':
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch", from_tf=True)
    model = GPT2LMHeadModel.from_pretrained("GroNLP/gpt2-small-dutch",from_tf=True)
else:
    tokenizer = AutoTokenizer.from_pretrained("antoiloui/belgpt2", from_tf=True)
    model = GPT2LMHeadModel.from_pretrained("antoiloui/belgpt2",from_tf=True)      

# Set the folder with the dataset
data_folder = os.path.join(dir_path,  'raw_data', language, 'Cohort_model')


def load_csv_data(filename):
    ''' Load the phonemes from csv files.
    '''
    
    df_phonemes =pd.read_table(filename, encoding = "utf-8", sep=';' )

    j=0   
    phonemes_words_test=[] #Array to list the phoneme transcriptions of words in the stories.
    grapheme_words_test=[]
    while j<(len(df_phonemes)): #len(df_phonemes) #It goes through the list of phonemes to capture the phoneme transcription of each word.
        phoneme_word=[]
        same=True
        word='start'
        while same==True:
            if word=='start' or word==df_phonemes['words'][j]:
                same=True
                word=df_phonemes['words'][j]
                ph=df_phonemes['phonemes'][j]
                phoneme_word.append(ph)
                j+=1
            else:
                same=False        
            if j==len(df_phonemes): #len(df_phonemes)
                break
        phonemes_words_test.append(phoneme_word)
        grapheme_words_test.append(word)
                  
    return grapheme_words_test,phonemes_words_test, df_phonemes


def load_text_data(filename):
       ''' Load the texts from the filename, splitting by lines and removing empty strings.
       '''
       all_words = []
       words=[]
       counter=0
       with open(filename, encoding="ISO-8859-1") as reader:
           #sentences = reader.readlines()
           for line in reader:
               counter=counter+1
                   # Append the line to the sentences, removing the end of line character
               line=line.split('  ') 
               words=(line[0], line[1][0:-1])
               if len(line)>0:
                  all_words.append(words)
                    
       return all_words
   
       
            

   
def convert_list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)  
        
phoneme_files = [f for f in os.listdir(data_folder) if f.endswith('cohort_model_2022.csv')]


for p in range(len(phoneme_files)): # phoneme transcription of stories #len(phoneme_files))

    shannon_all=[]
    word_number=[]
    j=0   
    phonemes_words_test=[]
    graph_words_test=[]
    
    graph_words_test,phonemes_words_test, df_phonemes=load_csv_data(os.path.join(data_folder, phoneme_files[p]))
    sequence= ['start']
    #print(sequence_test)
    
    for w in range(len(graph_words_test)):
        if len(sequence)<30:
            sequence.append(graph_words_test[w])
        else:
            sequence.pop(0)
            sequence.append(graph_words_test[w])
             
        
        sequence_test = convert_list_to_string(sequence)
        
        
        input_ids = tokenizer.encode(sequence_test, return_tensors="pt")
    
        # get logits of last hidden state
        next_token_logits = model(input_ids , return_dict=True).logits[:, -1, :]  #This is the vector of conditional probs of all words in dict.
        
        # sample
        probs = F.softmax(next_token_logits, dim=-1)
        
        for ph in range(len(phonemes_words_test[w])):
            shannon_all.append(-np.sum(probs.detach().numpy()*np.log2(probs.detach().numpy())))
            word_number.append(w+1)
            

    Data_word={'word_entropy_GPT':shannon_all,
               'word_number': word_number}
    df_word=pd.DataFrame(data=Data_word)
    df_all=pd.concat([df_phonemes, df_word], axis=1)

    print('Bitti2')


    base_filename = phoneme_files[p][0:-4]+'_word_number_cohort_word_entropy_GPT_2022.csv'

        
    with open(os.path.join(output_folder, base_filename), 'w') as outfile:
        df_all.to_csv(outfile,index=False, sep=';', line_terminator='\n')              
            
            
            
