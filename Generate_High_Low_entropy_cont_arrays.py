"""This script creates a continous array to mark the boundries of phonemes in Low or High Entropy words"""
from pathlib import Path
import re
import os
import eelbrain
import mne
import trftools
import numpy as np

language = 'Dutch'
DATA_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).expanduser() 
STIMULUS_DIR = DATA_ROOT / 'stimuli' / language
PREDICTOR_DIR = DATA_ROOT / 'predictors' / language
STIMULI = [f.split('_word')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('_word_entropy_GPT_freq_corrected_equal_signal.csv')]
STIMULI.sort()


gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# # Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 0.8, name='gammatone') for x in gammatone]

phoneme_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~phoneme_GPT_equal_signal_freq_corrected.pickle') for stimulus in STIMULI] 
phoneme_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') for gt, ds in zip(gammatone, phoneme_tables)]

phoneme_tables2 = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~phoneme_offset_GPT_equal_signal_freq_corrected.pickle') for stimulus in STIMULI] 
phoneme_offset = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') for gt, ds in zip(gammatone, phoneme_tables2)]
word_entropy = [eelbrain.event_impulse_predictor(gt.time, value= 'entropy_class', ds=ds, name='entropy_class') for gt, ds in zip(gammatone, phoneme_tables)]

low_cont = []
high_cont = []
for s,story in enumerate(STIMULI):

    cont=word_entropy[s].x #entropy class feature
    offset=phoneme_offset[s].x.copy() #phoneme offset feature
    check=0

    i=0
    while i<len(cont): 
        print('i: '+str(i))
        if not (cont[i]==0):# Search for the start of phoneme
            check=cont[i]
            j=i+1 
            stop=-1
            while j<len(cont):
                if not (cont[j]==0) or  not (offset[j]==0): # When the phoneme onset is found, it starts to search for phoneme offset or the start of next phoneme
                    stop=j
                    break
                else:
                    j=j+1
                    
            while i<stop: # It fills between phoneme onset and offset with 1s for high entropy, -1s for low entropy, and keeps 0s for silent gaps.
                cont[i]=check
                i=i+1
        else:
            i=i+1

    low = cont.copy()
    low[low>0]=0
    low_cont.append(low)
    high=cont.copy()
    high[high<0]=0
    high_cont.append(high)
    
    
low_entropy = np.array(low_cont)
high_entropy = np.array(high_cont)    
low_entropy_name = language+' low_entropy_cont_100Hz_all_stories_equal_signal.npy'
high_entropy_name = language+' high_entropy_cont_100Hz_all_stories_equal_signal.npy'

np.save(STIMULUS_DIR/low_entropy_name,low_entropy)    
np.save(STIMULUS_DIR/high_entropy_name,high_entropy)     