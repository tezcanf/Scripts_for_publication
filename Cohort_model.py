# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:57:32 2021

@author: filtez

This scripts take the grapheme to phoneme dictinory file and freq count file 
to calculate the phoneme suprisal and entropy values of the each word in the stories 
based on cohort model.
"""

import os
from os import path, mkdir
import json, pickle
import pandas as pd
import numpy as np


def load_text_data(filename,language):

    """Reads the grapheme to phoneme dictinory file. In the file each line has one word. 
    Grapheme and phoneme transcription is seperated by a single space. """

    all_words = []
    words=[]
    if language== 'French':
        df =pd.read_csv(filename,sep=';', encoding='latin1') #French
        # print(df)
        for i in range(len(df)):
            words = [df['grapheme'][i],df['phoneme'][i]]
            all_words.append(words)
                   
    else:
        encoding = 'utf8'        
        with open(filename, encoding=encoding) as reader: 
            for line in reader:
                line=line.split('\t') #Seperating the grapheme and phoneme
                #words=line[0:-1]
                line[-1]=line[-1][0:-1]# Not reading the new line char.
                words=line
                #words=(line[0], line[1][0:-1]) # Not reading the new line char.
                if len(line)>0:
                    all_words.append(words)
                    
    return all_words
             
class PrononVocab: 


    def __init__(self): #Initialization of the type of vocabulary

        self.phone2int ={}
        self.int2phone ={}
        self.phoneme_word_list=[]
        self.all_phones=[]
        
    def __call__(self, text):       

        # Creating a dictionary that maps integers to the phonemes

        for ind  in range(0, len(text)):
            phonemes=text[ind][1] #Seperating each phoneme 
            phonemes=str(phonemes).split(' ')
            leng=len(phonemes)
            t=0
            while t<leng:
                if phonemes[t]=='':
                    phonemes.pop(t) # Removes the empty index
                    leng=len(phonemes)
                else:
                    t=t+1
            
            for ph in range(len(phonemes)):
                self.all_phones.append(phonemes[ph])
            phonemes_clean =(' ').join(phonemes)
            self.phoneme_word_list.append(phonemes_clean)
                      
   
# Set the root folder
dir_path = os.path.dirname(os.path.realpath(__file__))
language='Dutch' # or 'French'

""" Initial phoneme count in words in frequency count file """

phoneme_words_all=[]  #phoneme transcription of all the words in freq file 
graph_words_all=[] #grapheme transcription of all the words in freq file 

#This is for Dutch
no_phoneme_transcription_count = 0
total_freq_count = 0

if language=='Dutch':
    
    dict_filename='Dutch_dict_2022.txt' 
    dict_folder = os.path.join(dir_path,  'raw_data', language) 
    dict_path = os.path.join(dict_folder, dict_filename)
    
    
    freq_filename='SUBTLEX-NL_filtered_2022_cut.csv'      
    freq_file_path = os.path.join(dict_folder, freq_filename)
    
    
    '''Read the Subtlex file for word frequencies. This is different for French!!! '''
    df_freq =pd.read_csv(freq_file_path,sep=';') #Dutch


    all_words=load_text_data(dict_path,language) #Load all words in the dictionary file
    Dictionary=PrononVocab() #Create the dictionary
    Dictionary(all_words)
    
    #Setting the directories where phoneme and word onset, offset alignments are located.
    phoneme_folder_name='Word_phoneme_transcription_of_stories' #'Word_phoneme_transcription_of_stories_phonemes_revised'
    phoneme_folder = os.path.join(dict_folder, phoneme_folder_name)
    phoneme_files = [f for f in os.listdir(phoneme_folder) if f.endswith('.csv')]
    
    output_folder_name= 'Cohort_model'
    output_folder = os.path.join(dir_path,  'raw_data', language , output_folder_name)
       
    if not os.path.exists(output_folder): #Create the folder if not exists
        os.mkdir(output_folder)
        
    unique_phones=list(set(Dictionary.all_phones))
    for i in range(len(unique_phones)):
        Dictionary.phone2int[unique_phones[i]]=i
        Dictionary.int2phone[i]=unique_phones[i]
              
    # Save the dictionary to data path dir  
        
    with open(os.path.join(dict_folder, 'phone2int_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.phone2int, f)
        
    with open(os.path.join(dict_folder, 'int2phone_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.int2phone, f)
    
        
    num_phones=len(Dictionary.phone2int)
    
    num_words=len(df_freq['Word']) #Dutch
    num_words_all_cohort = 970843 #df_freq['SUBTLEXWF'].sum() #Number of words in the frequency count cohort Dutch 
    """ Frequncy of each first phoneme per word in the cohort. """
    
    Counter=np.ones(num_phones)*1/num_words_all_cohort
    
    df_word_grapheme = pd.DataFrame(all_words,columns=['grapheme', 'phoneme','delete'])
    df_word_grapheme = df_word_grapheme.drop(axis=1,columns='delete')
    
    for w in range(num_words):
    
        if df_freq['Word'][w] in df_word_grapheme['grapheme'].to_list():
            graph_words_all.append(df_freq['Word'][w])
            freq=float(df_freq['SUBTLEXWF'][w])
            index =df_word_grapheme.loc[df_word_grapheme['grapheme'] == df_freq['Word'][w]].index[0]
            phoneme_word_in_freq_file=df_word_grapheme['phoneme'][index]
            phoneme_words_all.append(phoneme_word_in_freq_file.split(' '))
    
            Counter[Dictionary.phone2int[phoneme_word_in_freq_file.split(' ')[0]]]+=freq/num_words_all_cohort
        else:
            no_phoneme_transcription_count = no_phoneme_transcription_count +1
            total_freq_count = total_freq_count + df_freq['SUBTLEXWF'][w]
    

    for p in range(len(phoneme_files)): # phoneme transcription of stories           
    
        df_phonemes =pd.read_table(os.path.join(phoneme_folder, phoneme_files[p]), encoding = "utf-8", sep=',' )
     
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
                       
        shannon_all=[] #Array to keep entropy values of each phoneme
        surprisal_all=[] #Array to keep surprisal values of each phoneme
        word_freq_in_story_all = []
        
        for i in range(len(phonemes_words_test)): #
            phoneme_words_all_temp=phoneme_words_all[:]
            Prob_matrix=np.ones((len(phonemes_words_test[i]), num_phones ))
            Prob_matrix=Prob_matrix*1/num_words_all_cohort #Initialize the probability matrix with min probabilities 
            Prob_matrix[0][:]=Counter # Probabilities of initial phonemes of each word was calculated based on the number of words in the cohort
            if phonemes_words_test[i]!='UNK':
                res=[] #Array to keep the remaining words in the cohort 
                for k in range(len(phonemes_words_test[i])-1): # -1 No need to count the remaining words in the cohort when it is on the last phoneme
                    for j in range(len(phoneme_words_all_temp)):
                        if not(len(phoneme_words_all_temp[j])<=k ):
                            if phoneme_words_all_temp[j][k]==phonemes_words_test[i][k]:
                                res.append(phoneme_words_all_temp[j])
    
                    """ Taking the words in cohort which start with the intial phoneme of test word.
                    Then it iterates to the next phoneme and only keeps the words in cohort 
                    which has the same phoneme with the test word at that position. """ 
                    
                    phoneme_words_all_temp=[]
                    phoneme_words_all_temp=res[:] 
                    res=[]
                    """Sumation of the remaining words in cohort starting with that phoneme weighted by their occurance freq"""
                    for w in range(len(phoneme_words_all_temp)):
                        if not(len(phoneme_words_all_temp[w])<=k+1 ):
                            if (' ').join(phoneme_words_all_temp[w]) in df_word_grapheme['phoneme'].to_list():
                                index =df_word_grapheme.loc[df_word_grapheme['phoneme'] == (' ').join(phoneme_words_all_temp[w])].index[0]
                                grapheme=df_word_grapheme['grapheme'][index]
                                a=df_freq.loc[df_freq['Word'] == grapheme]
                                
                            else:
                                a=[]
                                print('yok')
                                   
                            if len(a)==0:
                                freq=1/num_words_all_cohort #If the word is not in the freq count list, then freq is set to min value
                                
                            else:
                                freq=a.iloc[-1]['SUBTLEXWF'] 
                                freq=float(freq)
                            Prob_matrix[k+1][Dictionary.phone2int[phoneme_words_all_temp[w][k+1]]]+=1*freq/num_words_all_cohort 
    
                #Calculating the entropy and surprisal values of each phoneme in the test word
                shannon=np.zeros(len(phonemes_words_test[i]))
                surprisal=np.zeros(len(phonemes_words_test[i]))
                for k in range(len(phonemes_words_test[i])): 
                    shannon[k] = -np.sum(Prob_matrix[k]*np.log2(Prob_matrix[k]))
                    surprisal[k]=-np.log2(Prob_matrix[k][Dictionary.phone2int[phonemes_words_test[i][k]]]/Prob_matrix[k].sum())            
                    shannon_all.append(shannon[k])           
                    surprisal_all.append(surprisal[k])
                    
                    ff = df_freq.loc[df_freq['Word'] == grapheme_words_test[i].lower()]
                    if len(ff) > 0:
                        word_freq_in_story = -np.log2(ff.iloc[-1]['SUBTLEXWF']/num_words_all_cohort)
    
                    else:
                        word_freq_in_story = 'NAN' #1/num_words_all_cohort
                        print(grapheme_words_test[i])
                        print('This word does not have a freq')
                    word_freq_in_story_all.append(word_freq_in_story)
    
        Data_cohort={'cohort_entropy':shannon_all, 'cohort_surprisal': surprisal_all, 'word_freq': word_freq_in_story_all}
        df_cohort=pd.DataFrame(data=Data_cohort)
        df_all=pd.concat([df_phonemes, df_cohort], axis=1)
    
        output_filename = phoneme_files[p][0:-4]+'_cohort_model.csv'
            
        with open(os.path.join(output_folder, output_filename), 'w') as outfile:
            df_all.to_csv(outfile,index=False, sep=';', line_terminator='\n')              
    
if language=='French':
    dict_filename='French_dict_2022.csv' #'fr_ANSI.dict' 
    dict_folder = os.path.join(dir_path,  'raw_data', language) 
    dict_path = os.path.join(dict_folder, dict_filename)


    freq_filename='Lexique383_filtered_2022.csv'   
    freq_file_path = os.path.join(dict_folder, freq_filename)
    
    
    '''Read the Subtlex file for word frequencies. This is different for French!!! '''
    df_freq =pd.read_csv(freq_file_path,sep=';', encoding='latin1') #French

    all_words=load_text_data(dict_path,language) #Load all words in the dictionary file
    Dictionary=PrononVocab() #Create the dictionary
    Dictionary(all_words)
    
    #Setting the directories where phoneme and word onset, offset alignments are located.
    phoneme_folder_name='Word_phoneme_transcription_of_stories' #'Word_phoneme_transcription_of_stories_phonemes_revised'
    phoneme_folder = os.path.join(dict_folder, phoneme_folder_name)
    phoneme_files = [f for f in os.listdir(phoneme_folder) if f.endswith('.csv')]
    
    output_folder_name= 'Cohort_model'
    output_folder = os.path.join(dir_path,  'raw_data', language , output_folder_name)
       
    if not os.path.exists(output_folder): #Create the folder if not exists
        os.mkdir(output_folder)
        
    unique_phones=list(set(Dictionary.all_phones))
    for i in range(len(unique_phones)):
        Dictionary.phone2int[unique_phones[i]]=i
        Dictionary.int2phone[i]=unique_phones[i]
              
    # Save the dictionary to data path dir  
        
    with open(os.path.join(dict_folder, 'phone2int_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.phone2int, f)
        
    with open(os.path.join(dict_folder, 'int2phone_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.int2phone, f)
    
        
    num_phones=len(Dictionary.phone2int)
    
    num_words=len(df_freq['1_ortho']) #Dutch
    num_words_all_cohort = 902028 #df_freq['SUBTLEXWF'].sum() #Number of words in the frequency count cohort Dutch 
    """ Frequncy of each first phoneme per word in the cohort. """
    
    Counter=np.ones(num_phones)*1/num_words_all_cohort
    
    df_word_grapheme = pd.DataFrame(all_words,columns=['grapheme', 'phoneme'])
    
    for w in range(num_words):
    
        if df_freq['1_ortho'][w] in df_word_grapheme['grapheme'].to_list():
            graph_words_all.append(df_freq['1_ortho'][w])
            freq=float(df_freq['9_freqfilms2'][w])
            index =df_word_grapheme.loc[df_word_grapheme['grapheme'] == df_freq['1_ortho'][w]].index[0]
            phoneme_word_in_freq_file=df_word_grapheme['phoneme'][index]
            phoneme_words_all.append(phoneme_word_in_freq_file.split(' '))
    
            Counter[Dictionary.phone2int[phoneme_word_in_freq_file.split(' ')[0]]]+=freq/num_words_all_cohort
        else:
            no_phoneme_transcription_count = no_phoneme_transcription_count +1
            total_freq_count = total_freq_count + df_freq['9_freqfilms2'][w]

    
    
    for p in range( len(phoneme_files) ): # phoneme transcription of stories          
    
        df_phonemes =pd.read_table(os.path.join(phoneme_folder, phoneme_files[p]), encoding = "utf-8", sep=',' )
     
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
                       
        shannon_all=[] #Array to keep entropy values of each phoneme
        surprisal_all=[] #Array to keep surprisal values of each phoneme
        word_freq_in_story_all = []
        
        for i in range(len(phonemes_words_test)): #
            phoneme_words_all_temp=phoneme_words_all[:]
            Prob_matrix=np.ones((len(phonemes_words_test[i]), num_phones ))
            Prob_matrix=Prob_matrix*1/num_words_all_cohort #Initialize the probability matrix with min probabilities 
            Prob_matrix[0][:]=Counter # Probabilities of initial phonemes of each word was calculated based on the number of words in the cohort
            if phonemes_words_test[i]!='UNK':
                res=[] #Array to keep the remaining words in the cohort 
                for k in range(len(phonemes_words_test[i])-1): # -1 No need to count the remaining words in the cohort when it is on the last phoneme
                    for j in range(len(phoneme_words_all_temp)):
                        if not(len(phoneme_words_all_temp[j])<=k ):
                            if phoneme_words_all_temp[j][k]==phonemes_words_test[i][k]:
                                res.append(phoneme_words_all_temp[j])
    
                    """ Taking the words in cohort which start with the intial phoneme of test word.
                    Then it iterates to the next phoneme and only keeps the words in cohort 
                    which has the same phoneme with the test word at that position. """ 
                    
                    phoneme_words_all_temp=[]
                    phoneme_words_all_temp=res[:] 
                    res=[]
                    #print('burda')
                    """Sumation of the remaining words in cohort starting with that phoneme weighted by their occurance freq"""
                    for w in range(len(phoneme_words_all_temp)):
                        if not(len(phoneme_words_all_temp[w])<=k+1 ):
                            if (' ').join(phoneme_words_all_temp[w]) in df_word_grapheme['phoneme'].to_list():
                                index =df_word_grapheme.loc[df_word_grapheme['phoneme'] == (' ').join(phoneme_words_all_temp[w])].index[0]
                                grapheme=df_word_grapheme['grapheme'][index]
                                a=df_freq.loc[df_freq['1_ortho'] == grapheme]
                                
                            else:
                                a=[]
                                print('yok')
                                   
                            if len(a)==0:
                                freq=1/num_words_all_cohort #If the word is not in the freq count list, then freq is set to min value
                                
                            else:
                                freq=a.iloc[-1]['9_freqfilms2'] 
                                freq=float(freq)
                            Prob_matrix[k+1][Dictionary.phone2int[phoneme_words_all_temp[w][k+1]]]+=1*freq/num_words_all_cohort 
    
                #Calculating the entropy and surprisal values of each phoneme in the test word
                shannon=np.zeros(len(phonemes_words_test[i]))
                surprisal=np.zeros(len(phonemes_words_test[i]))
                for k in range(len(phonemes_words_test[i])): 
                    shannon[k] = -np.sum(Prob_matrix[k]*np.log2(Prob_matrix[k]))
                    surprisal[k]=-np.log2(Prob_matrix[k][Dictionary.phone2int[phonemes_words_test[i][k]]]/Prob_matrix[k].sum())            
                    shannon_all.append(shannon[k])           
                    surprisal_all.append(surprisal[k])
                    
                    ff = df_freq.loc[df_freq['1_ortho'] == grapheme_words_test[i].lower()]
                    if len(ff) > 0:
                        word_freq_in_story = -np.log2(ff.iloc[-1]['9_freqfilms2']/num_words_all_cohort)
    
                    else:
                        word_freq_in_story = 'NAN' #1/num_words_all_cohort
                        print(grapheme_words_test[i])
                        print('This word does not have a freq')
                    word_freq_in_story_all.append(word_freq_in_story)
    
        Data_cohort={'cohort_entropy':shannon_all, 'cohort_surprisal': surprisal_all, 'word_freq': word_freq_in_story_all}
        df_cohort=pd.DataFrame(data=Data_cohort)
        df_all=pd.concat([df_phonemes, df_cohort], axis=1)
    
        output_filename = phoneme_files[p][0:-4]+'_cohort_model.csv'
            
        with open(os.path.join(output_folder, output_filename), 'w') as outfile:
            df_all.to_csv(outfile,index=False, sep=';', line_terminator='\n')   