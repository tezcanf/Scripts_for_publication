"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
from pathlib import Path
import os
import eelbrain


language = 'French'
DATA_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).expanduser() 
STIMULUS_DIR = DATA_ROOT / 'stimuli' / language
PREDICTOR_DIR = DATA_ROOT / 'predictors' / language
list_stories = [f.split('_normalized')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('.wav')]

for segment in list_stories:
    segment_table = eelbrain.load.tsv(STIMULUS_DIR / f'{segment}_word_entropy_GPT_freq_corrected_equal_signal.csv',delimiter=';')

    ds = eelbrain.Dataset({'time': segment_table['phoneme_onset']}, info={'tstop': segment_table[-1, 'phoneme_offset']})

    for key in ['cohort_entropy', 'cohort_surprisal', 'word_entropy_GPT', 'word_freq_log', 'entropy_class']:
        ds[key] = segment_table[key]
        
    ds['High_word_entropy'] = segment_table['entropy_class'] == 1
    ds['Low_word_entropy'] = segment_table['entropy_class'] == -1
    # save
    eelbrain.save.pickle(ds, PREDICTOR_DIR / f'{segment}~phoneme_GPT_equal_signal_freq_corrected.pickle')
    
    ds = eelbrain.Dataset({'time': segment_table['phoneme_offset']})

    # save
    eelbrain.save.pickle(ds, PREDICTOR_DIR / f'{segment}~phoneme_offset_GPT_equal_signal_freq_corrected.pickle')
