

from pathlib import Path
import numpy as np
import eelbrain
from matplotlib import pyplot
import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

Results_path = '/project/3027007.02/Scripts_for_publication/New_results/'

# Configure the matplotlib figure style
FONT = "Times New Roman"
colors = cycler(color=[ 'lightgrey','dimgrey','yellowgreen' , 'olivedrab','coral','orangered' ]) #Dutch
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    # Font
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (4,3),
    'axes.prop_cycle':colors}



pyplot.rcParams.update(RC)


models_low = ['equal_signal_acoustic_low',
              'equal_signal_acoustic+phonemes_low',
              'equal_signal_acoustic+phonemes+surprisal_low',
              'equal_signal_acoustic+phonemes+surprisal+entropy_low',]
          
models_high = ['equal_signal_acoustic_high',
              'equal_signal_acoustic+phonemes_high',
              'equal_signal_acoustic+phonemes+surprisal_high',
              'equal_signal_acoustic+phonemes+surprisal+entropy_high',]


# print (dur)
#%%

rows = []
for t in range(17):
    
    language = 'French'
    region = 'Whole_brain'
    
    
    DATA_ROOT = Path(
        "/project/3027007.02/Scripts_for_publication/").expanduser()
    TRF_folder_name = 'TRFs_' + region+'_'+language
    
    TRF_DIR = DATA_ROOT / TRF_folder_name
    SUBJECTS = [path.name for path in TRF_DIR.iterdir()
                if re.match(r'sub\d*', path.name)]
    SUBJECTS.sort()  

    time_interval = 'mod_revision_time_interval_'+str(t)+'_'
    
    for subject in SUBJECTS:
        trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[0]}_lh.pickle')
        trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[0]}_rh.pickle')
        trf_lh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_lh.pickle')
        trf_rh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_rh.pickle')
        trf_lh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_lh.pickle')
        trf_rh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_rh.pickle')
        trf_lh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_lh.pickle')
        trf_rh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_rh.pickle')
    
        #Create a copy of Boosting result before substracting the accuracy values of previous base model
        trf_lh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_lh.pickle')
        trf_rh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_rh.pickle')
        trf_lh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_lh.pickle')
        trf_rh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_rh.pickle')
        trf_lh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_lh.pickle')
        trf_rh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_rh.pickle')
        
        #Substract the accuracy values of previous base model from the model of interest
        
        trf_rh_phoneme_onset.proportion_explained.x  = trf_rh1.proportion_explained.x - trf_rh0.proportion_explained.x
        trf_lh_phoneme_onset.proportion_explained.x  = trf_lh1.proportion_explained.x - trf_lh0.proportion_explained.x
    
        trf_rh_surprisal.proportion_explained.x  = trf_rh2.proportion_explained.x - trf_rh1.proportion_explained.x
        trf_lh_surprisal.proportion_explained.x  = trf_lh2.proportion_explained.x - trf_lh1.proportion_explained.x
    
        trf_rh_entropy.proportion_explained.x  = trf_rh3.proportion_explained.x - trf_rh2.proportion_explained.x
        trf_lh_entropy.proportion_explained.x  = trf_lh3.proportion_explained.x - trf_lh2.proportion_explained.x
            
        rows.append(['French', 'Low', 'Right',  'Phoneme_onset', t, subject,  trf_rh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'Low', 'Left', 'Phoneme_onset', t,  subject, trf_lh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    
        rows.append(['French', 'Low', 'Right',  'Surprisal', t, subject,  trf_rh_surprisal.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'Low', 'Left', 'Surprisal', t,  subject,  trf_lh_surprisal.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
        
        rows.append(['French', 'Low', 'Right',  'Entropy', t, subject,  trf_rh_entropy.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'Low', 'Left', 'Entropy', t,  subject,  trf_lh_entropy.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
            
        print(np.shape(rows))
        # print(dur)
    
    for subject in SUBJECTS:
        trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[0]}_lh.pickle')
        trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[0]}_rh.pickle')
        trf_lh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_lh.pickle')
        trf_rh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_rh.pickle')
        trf_lh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_lh.pickle')
        trf_rh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_rh.pickle')
        trf_lh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_lh.pickle')
        trf_rh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_rh.pickle')
    
        #Create a copy of Boosting result before substracting the accuracy values of previous base model
        trf_lh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_lh.pickle')
        trf_rh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_rh.pickle')
        trf_lh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_lh.pickle')
        trf_rh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_rh.pickle')
        trf_lh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_lh.pickle')
        trf_rh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_rh.pickle')
        
        #Substract the accuracy values of previous base model from the model of interest
       
        trf_rh_phoneme_onset.proportion_explained.x  = trf_rh1.proportion_explained.x - trf_rh0.proportion_explained.x
        trf_lh_phoneme_onset.proportion_explained.x  = trf_lh1.proportion_explained.x - trf_lh0.proportion_explained.x
    
        trf_rh_surprisal.proportion_explained.x  = trf_rh2.proportion_explained.x - trf_rh1.proportion_explained.x
        trf_lh_surprisal.proportion_explained.x  = trf_lh2.proportion_explained.x - trf_lh1.proportion_explained.x
    
        trf_rh_entropy.proportion_explained.x  = trf_rh3.proportion_explained.x - trf_rh2.proportion_explained.x
        trf_lh_entropy.proportion_explained.x  = trf_lh3.proportion_explained.x - trf_lh2.proportion_explained.x

        rows.append(['French', 'High', 'Right',  'Phoneme_onset', t, subject,  trf_rh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'High', 'Left', 'Phoneme_onset', t,  subject, trf_lh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    
        rows.append(['French', 'High', 'Right',  'Surprisal', t, subject, trf_rh_surprisal.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'High', 'Left', 'Surprisal', t,  subject,trf_lh_surprisal.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
        
        rows.append(['French', 'High', 'Right',  'Entropy', t, subject,  trf_rh_entropy.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['French', 'High', 'Left', 'Entropy', t,  subject,  trf_lh_entropy.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
          
        
            
    language = 'Dutch'
    region = 'Whole_brain'
       
    DATA_ROOT = Path(
        "/project/3027007.02/Scripts_for_publication/").expanduser()
    TRF_folder_name = 'TRFs_' + region+'_'+language
    
    TRF_DIR = DATA_ROOT / TRF_folder_name
    SUBJECTS = [path.name for path in TRF_DIR.iterdir()
                if re.match(r'sub\d*', path.name)]
    SUBJECTS.sort()  
    
     
    for subject in SUBJECTS:
        trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[0]}_lh.pickle')
        trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[0]}_rh.pickle')
        trf_lh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_lh.pickle')
        trf_rh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_rh.pickle')
        trf_lh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_lh.pickle')
        trf_rh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_rh.pickle')
        trf_lh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_lh.pickle')
        trf_rh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_rh.pickle')
    
        #Create a copy of Boosting result before substracting the accuracy values of previous base model
        trf_lh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_lh.pickle')
        trf_rh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[1]}_rh.pickle')
        trf_lh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_lh.pickle')
        trf_rh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[2]}_rh.pickle')
        trf_lh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_lh.pickle')
        trf_rh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_low[3]}_rh.pickle')
        
        #Substract the accuracy values of previous base model from the model of interest
        
        trf_rh_phoneme_onset.proportion_explained.x  = trf_rh1.proportion_explained.x - trf_rh0.proportion_explained.x
        trf_lh_phoneme_onset.proportion_explained.x  = trf_lh1.proportion_explained.x - trf_lh0.proportion_explained.x
    
        trf_rh_surprisal.proportion_explained.x  = trf_rh2.proportion_explained.x - trf_rh1.proportion_explained.x
        trf_lh_surprisal.proportion_explained.x  = trf_lh2.proportion_explained.x - trf_lh1.proportion_explained.x
    
        trf_rh_entropy.proportion_explained.x  = trf_rh3.proportion_explained.x - trf_rh2.proportion_explained.x
        trf_lh_entropy.proportion_explained.x  = trf_lh3.proportion_explained.x - trf_lh2.proportion_explained.x
     
        rows.append(['Dutch', 'Low', 'Right',  'Phoneme_onset', t, subject, trf_rh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'Low', 'Left', 'Phoneme_onset', t,  subject,  trf_lh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    
        rows.append(['Dutch', 'Low', 'Right',  'Surprisal', t, subject,  trf_rh_surprisal.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'Low', 'Left', 'Surprisal', t,  subject,  trf_lh_surprisal.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
        
        rows.append(['Dutch', 'Low', 'Right',  'Entropy', t, subject,  trf_rh_entropy.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'Low', 'Left', 'Entropy', t,  subject, trf_lh_entropy.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
            
        print(np.shape(rows))
    
    for subject in SUBJECTS:
        trf_lh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[0]}_lh.pickle')
        trf_rh0 = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[0]}_rh.pickle')
        trf_lh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_lh.pickle')
        trf_rh1  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_rh.pickle')
        trf_lh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_lh.pickle')
        trf_rh2  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_rh.pickle')
        trf_lh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_lh.pickle')
        trf_rh3  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_rh.pickle')
    
        #Create a copy of Boosting result before substracting the accuracy values of previous base model
        trf_lh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_lh.pickle')
        trf_rh_phoneme_onset  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[1]}_rh.pickle')
        trf_lh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_lh.pickle')
        trf_rh_surprisal  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[2]}_rh.pickle')
        trf_lh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_lh.pickle')
        trf_rh_entropy  = eelbrain.load.unpickle(TRF_DIR / subject / f'{subject} {time_interval}{models_high[3]}_rh.pickle')
        
        #Substract the accuracy values of previous base model from the model of interest
        
        trf_rh_phoneme_onset.proportion_explained.x  = trf_rh1.proportion_explained.x - trf_rh0.proportion_explained.x
        trf_lh_phoneme_onset.proportion_explained.x  = trf_lh1.proportion_explained.x - trf_lh0.proportion_explained.x
    
        trf_rh_surprisal.proportion_explained.x  = trf_rh2.proportion_explained.x - trf_rh1.proportion_explained.x
        trf_lh_surprisal.proportion_explained.x  = trf_lh2.proportion_explained.x - trf_lh1.proportion_explained.x
    
        trf_rh_entropy.proportion_explained.x  = trf_rh3.proportion_explained.x - trf_rh2.proportion_explained.x
        trf_lh_entropy.proportion_explained.x  = trf_lh3.proportion_explained.x - trf_lh2.proportion_explained.x
        
        rows.append(['Dutch', 'High', 'Right',  'Phoneme_onset', t, subject, trf_rh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'High', 'Left', 'Phoneme_onset', t,  subject,  trf_lh_phoneme_onset.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
    
        rows.append(['Dutch', 'High', 'Right',  'Surprisal', t, subject,trf_rh_surprisal.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'High', 'Left', 'Surprisal', t,  subject,  trf_lh_surprisal.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
        
        rows.append(['Dutch', 'High', 'Right',  'Entropy', t, subject,  trf_rh_entropy.proportion_explained.smooth('source', window_size=0.014,window='gaussian').mean(axis='source')])
        rows.append(['Dutch', 'High', 'Left', 'Entropy', t,  subject,  trf_lh_entropy.proportion_explained.smooth('source', window_size=0.014, window='gaussian').mean(axis='source')])
          
 
        
df = pd.DataFrame(data = rows, columns = ['Language','Word Entropy' ,'hemisphere', 'model','time', 'subject','accuracy'])

df.to_csv(os.path.join(Results_path,'Accuracies_both_hemispheres_word_entropy_French_Dutch_revision_suggestion_time_interval.csv'))

