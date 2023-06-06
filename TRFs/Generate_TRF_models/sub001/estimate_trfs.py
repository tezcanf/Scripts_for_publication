"""This script estimates TRFs for several models and saves them"""
from pathlib import Path
import re
import os
import eelbrain
import mne
import trftools
from utils_TRF import make_source_space
import numpy as np

DATA_ROOT = Path('/project/3027007.02/Scripts_for_publication').expanduser() 
language = 'Dutch'
region =  'Whole_brain' #'Whole_brain_resting_full_rank'
STIMULUS_DIR = DATA_ROOT / 'stimuli' / language
STIMULI = [f.split('_word')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('word_entropy_GPT_freq_corrected_equal_signal.csv')]
STIMULI.sort()
PREDICTOR_DIR = DATA_ROOT / 'predictors' / language
i=0

SUBJECTS = ['sub-001',
            'sub-003',
            'sub-004',
            'sub-008',
            'sub-009',
            'sub-010',
            'sub-011',
            'sub-013',
            'sub-014',
            'sub-015',
            'sub-017',
            'sub-018',
            'sub-019',
            'sub-020',
            'sub-021',
            'sub-023',
            'sub-025',
            'sub-026',
            'sub-027',
            'sub-028',
            'sub-029',
            'sub-030',
            'sub-032',
            'sub-033',           
            ]

subject = SUBJECTS[i]
# Define a target directory for TRF estimates and make sure the directory is created
TRF_folder_name = 'TRFs_'+region+'_'+language
TRF_DIR = DATA_ROOT / TRF_folder_name
TRF_DIR.mkdir(exist_ok=True)
subjects_dir = '/project/3027007.01/processed/'


#Phoneme table name, low_vs_high entropy file name change, stimuli end

# Load stimuli
# ------------
# Make sure to name the stimuli so that the TRFs can later be distinguished
# Load the gammatone-spectrograms; use the time axis of these as reference
gammatone = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
# # Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 0.8, name='gammatone') for x in gammatone]

gammatone_low = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone_low = [x.bin(0.01, dim='time', label='start') for x in gammatone_low]
# # Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone_low = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 0.8, name='gammatone') for x in gammatone_low]


gammatone_high = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-8.pickle') for stimulus in STIMULI]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone_high = [x.bin(0.01, dim='time', label='start') for x in gammatone_high]
# # Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone_high = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 0.8, name='gammatone') for x in gammatone_high]


# Load onset spectrograms and make sure the time dimension is equal to the gammatone spectrograms
gammatone_onsets = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-on-8.pickle') for stimulus in STIMULI]
gammatone_onsets = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets]
gammatone_onsets = [eelbrain.set_time(x, gt.time, name='gammatone_on') for x, gt in zip(gammatone_onsets, gammatone)]

gammatone_onsets_low = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-on-8.pickle') for stimulus in STIMULI]
gammatone_onsets_low = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets_low]
gammatone_onsets_low = [eelbrain.set_time(x, gt.time, name='gammatone_on') for x, gt in zip(gammatone_onsets_low, gammatone_low)]

gammatone_onsets_high = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~gammatone-on-8.pickle') for stimulus in STIMULI]
gammatone_onsets_high = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets_high]
gammatone_onsets_high = [eelbrain.set_time(x, gt.time, name='gammatone_on') for x, gt in zip(gammatone_onsets_high, gammatone_high)]


# Load word tables and convert tables into continuous time-series with matching time dimension
phoneme_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~phoneme_GPT_equal_signal_freq_corrected.pickle') for stimulus in STIMULI] 
# phoneme_tables = [eelbrain.load.unpickle(PREDICTOR_DIR / f'{stimulus}~phoneme_GPT_freq_corrected_median_word_entropy.pickle') for stimulus in STIMULI] 
phoneme_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_onsets_low = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_onsets_high = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') for gt, ds in zip(gammatone, phoneme_tables)]

phoneme_surprisal= [eelbrain.event_impulse_predictor(gt.time, value='cohort_surprisal', ds=ds, name='cohort_surprisal') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_entropy= [eelbrain.event_impulse_predictor(gt.time, value='cohort_entropy', ds=ds, name='cohort_entropy') for gt, ds in zip(gammatone, phoneme_tables)]

phoneme_surprisal_low= [eelbrain.event_impulse_predictor(gt.time, value='cohort_surprisal', ds=ds, name='cohort_surprisal') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_entropy_low= [eelbrain.event_impulse_predictor(gt.time, value='cohort_entropy', ds=ds, name='cohort_entropy') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_surprisal_high= [eelbrain.event_impulse_predictor(gt.time, value='cohort_surprisal', ds=ds, name='cohort_surprisal') for gt, ds in zip(gammatone, phoneme_tables)]
phoneme_entropy_high= [eelbrain.event_impulse_predictor(gt.time, value='cohort_entropy', ds=ds, name='cohort_entropy') for gt, ds in zip(gammatone, phoneme_tables)]

word_freq= [eelbrain.event_impulse_predictor(gt.time, value='word_freq_log', ds=ds, name='word_freq_log') for gt, ds in zip(gammatone, phoneme_tables)]
word_freq_low= [eelbrain.event_impulse_predictor(gt.time, value='word_freq_log', ds=ds, name='word_freq_log') for gt, ds in zip(gammatone, phoneme_tables)]
word_freq_high= [eelbrain.event_impulse_predictor(gt.time, value='word_freq_log', ds=ds, name='word_freq_log') for gt, ds in zip(gammatone, phoneme_tables)]


#Make the word freq cont
for i in range(len(word_freq)):    
    fill = 0
    for t in range(len(word_freq[i].x)):
        if word_freq[i].x[t]!=0:
            fill = word_freq[i].x[t]
        word_freq[i].x[t]=fill
        
for i in range(len(word_freq)):
    word_freq_low[i].x = word_freq[i].x.copy()
    word_freq_high[i].x = word_freq[i].x.copy()



High_word_entropy = [eelbrain.event_impulse_predictor(gt.time, value='High_word_entropy', ds=ds, name='entropy_class_high') for gt, ds in zip(gammatone, phoneme_tables)]
Low_word_entropy = [eelbrain.event_impulse_predictor(gt.time, value='Low_word_entropy', ds=ds, name='entropy_class_low') for gt, ds in zip(gammatone, phoneme_tables)]

High_word_entropy_cont = [eelbrain.event_impulse_predictor(gt.time, value='High_word_entropy', ds=ds, name='entropy_class_high') for gt, ds in zip(gammatone, phoneme_tables)]
Low_word_entropy_cont = [eelbrain.event_impulse_predictor(gt.time, value='Low_word_entropy', ds=ds, name='entropy_class_low') for gt, ds in zip(gammatone, phoneme_tables)]


low_entropy_name = language+ ' low_entropy_cont_100Hz_all_stories_equal_signal.npy'
high_entropy_name = language+ ' high_entropy_cont_100Hz_all_stories_equal_signal.npy'


low_entropy_cont = np.load(STIMULUS_DIR / low_entropy_name, allow_pickle=True)
high_entropy_cont = np.load(STIMULUS_DIR / high_entropy_name, allow_pickle=True)

low_entropy_cont = low_entropy_cont
high_entropy_cont = high_entropy_cont

# low_entropy_cont = low_entropy_cont[2:]
# high_entropy_cont = high_entropy_cont[2:]


for i in range(len(STIMULI)):
    High_word_entropy_cont[i].x = high_entropy_cont[i]
    Low_word_entropy_cont[i].x = low_entropy_cont[i]



for i in range(len(High_word_entropy_cont)):
    for t in range(np.shape(gammatone_onsets[i])[1]):
        if High_word_entropy_cont[i].x[t] == 0:
            gammatone_high[i].x[:,t] = np.zeros(8)
            gammatone_onsets_high[i].x[:,t] = np.zeros(8)
            word_freq_high[i].x[t] = 0.0
        if High_word_entropy[i].x[t] == 0:
            phoneme_onsets_high[i].x[t] = 0.0
            phoneme_surprisal_high[i].x[t] = 0.0
            phoneme_entropy_high[i].x[t] = 0.0

        if Low_word_entropy_cont[i].x[t] == 0:
            gammatone_low[i].x[:,t] = np.zeros(8)
            gammatone_onsets_low[i].x[:,t] = np.zeros(8)
            word_freq_low[i].x[t] = 0.0

        if Low_word_entropy[i].x[t] == 0:
            phoneme_onsets_low[i].x[t] = 0.0
            phoneme_surprisal_low[i].x[t] = 0.0
            phoneme_entropy_low[i].x[t] = 0.0


#Models
#------
#Pre-define models here to have easier access during estimation. In the future, additional models could be added here and the script re-run to generate additional TRFs.
models = {
    #Analysis Part 1
    'acoustic': [gammatone, gammatone_onsets],
    'acoustic+phonemes': [gammatone, gammatone_onsets,phoneme_onsets],
    'acoustic+phonemes+surprisal': [gammatone, gammatone_onsets,phoneme_onsets,phoneme_surprisal],
    'acoustic+phonemes+surprisal+entropy': [gammatone, gammatone_onsets,phoneme_onsets,phoneme_surprisal,phoneme_entropy],
    'acoustic+phonemes+surprisal+entropy+freq': [gammatone, gammatone_onsets,phoneme_onsets,phoneme_surprisal,phoneme_entropy,word_freq],

    #Analysis Part 2 and 3
    'equal_signal_acoustic+phonemes+surprisal+entropy_low': [gammatone_low, gammatone_onsets_low,phoneme_onsets_low,phoneme_surprisal_low,phoneme_entropy_low,word_freq_low],
    'equal_signal_acoustic+phonemes+surprisal+entropy_high': [gammatone_high, gammatone_onsets_high,phoneme_onsets_high,phoneme_surprisal_high,phoneme_entropy_high,word_freq_high],
    'equal_signal_phonemes+surprisal+entropy_low': [phoneme_onsets_low,phoneme_surprisal_low,phoneme_entropy_low,word_freq_low],
    'equal_signal_phonemes+surprisal+entropy_high': [phoneme_onsets_high,phoneme_surprisal_high,phoneme_entropy_high,word_freq_high],
    'equal_signal_acoustic_low': [gammatone_low, gammatone_onsets_low,word_freq_low],
    'equal_signal_acoustic_high': [gammatone_high, gammatone_onsets_high,word_freq_high],
    
    #Revision suggestion Full Model
    'equal_signal_acoustic+phonemes+surprisal+entropy_all': [gammatone_low, gammatone_high, gammatone_onsets_low,gammatone_onsets_high, phoneme_onsets_low,phoneme_onsets_high,phoneme_surprisal_low,phoneme_surprisal_high,phoneme_entropy_low,phoneme_entropy_high,word_freq_low,word_freq_high],
    'acoustic+phonemes+surprisal+entropy_all_minus_high_entropy_phoneme_features': [gammatone_low, gammatone_high, gammatone_onsets_low,gammatone_onsets_high, phoneme_onsets_low,phoneme_surprisal_low,phoneme_entropy_low,word_freq_low,word_freq_high],
    'acoustic+phonemes+surprisal+entropy_all_minus_low_entropy_phoneme_features': [gammatone_low, gammatone_high, gammatone_onsets_low,gammatone_onsets_high, phoneme_onsets_high,phoneme_surprisal_high,phoneme_entropy_high,word_freq_low,word_freq_high],
    'acoustic+phonemes+surprisal+entropy_all_minus_low_entropy_acoustic_features': [gammatone_high,gammatone_onsets_high, phoneme_onsets_low,phoneme_onsets_high,phoneme_surprisal_low,phoneme_surprisal_high,phoneme_entropy_low,phoneme_entropy_high,word_freq_low,word_freq_high],
    'acoustic+phonemes+surprisal+entropy_all_minus_high_entropy_acoustic_features': [gammatone_low,gammatone_onsets_low, phoneme_onsets_low,phoneme_onsets_high,phoneme_surprisal_low,phoneme_surprisal_high,phoneme_entropy_low,phoneme_entropy_high,word_freq_low,word_freq_high],
   
}

# Estimate TRFs
subject_trf_dir = TRF_DIR / subject
subject_trf_dir.mkdir(exist_ok=True)
# Generate all TRF paths so we can check whether any new TRFs need to be estimated
trf_paths_lh = {model: subject_trf_dir / f'{subject} {model}_lh.pickle' for model in models}
trf_paths_rh = {model: subject_trf_dir / f'{subject} {model}_rh.pickle' for model in models}

stc_lh_all = []
stc_rh_all = []
for story in STIMULI:       

    stc_lh, stc_rh = make_source_space(subject, language, region, story, DATA_ROOT)
    
    stc_lh_ndvar = eelbrain.load.fiff.stc_ndvar(stc = stc_lh, subject = 'fsaverage',  src  ='ico-4',subjects_dir= subjects_dir, parc = 'aparc', check = True)
    stc_rh_ndvar = eelbrain.load.fiff.stc_ndvar(stc = stc_rh, subject = 'fsaverage',  src  ='ico-4',subjects_dir = subjects_dir, parc = 'aparc', check = True)

    del stc_rh, stc_lh

    stc_lh_all.append(stc_lh_ndvar)
    stc_rh_all.append(stc_rh_ndvar)
    
# Since trials are of unequal length, we will concatenate them for the TRF estimation.
stc_lh_ndvar_concatenated = eelbrain.concatenate(stc_lh_all)
stc_rh_ndvar_concatenated = eelbrain.concatenate(stc_rh_all)

for model, predictors in models.items():
    path_lh = trf_paths_lh[model]
    path_rh = trf_paths_rh[model]
    # Skip if this file already exists
    print(f"Estimating: {subject} ~ {model}")
    # Select and concetenate the predictors corresponding to the EEG trials
    predictors_concatenated = []
    for predictor in predictors:
        predictors_concatenated.append(eelbrain.concatenate([predictor[i] for i in range(len(STIMULI))]))
    # Fit the mTRF
    if not path_lh.exists():
        trf_lh = eelbrain.boosting(stc_lh_ndvar_concatenated, predictors_concatenated, -0.100, 0.800, error='l1', basis=0.050, partitions=5, test=1, selective_stopping=True)
        eelbrain.save.pickle(trf_lh, path_lh)
        del trf_lh
    if not path_rh.exists():
        trf_rh = eelbrain.boosting(stc_rh_ndvar_concatenated, predictors_concatenated, -0.100, 0.800, error='l1', basis=0.050, partitions=5, test=1, selective_stopping=True)
        eelbrain.save.pickle(trf_rh, path_rh)
        # Save the TRF for later analysis
        del trf_rh
    
       
