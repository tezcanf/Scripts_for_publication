"""Generate high-resolution gammatone spectrograms"""
from pathlib import Path
import os
from eelbrain import *
from trftools import gammatone_bank


language = 'French'
DATA_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).expanduser() 
STIMULUS_DIR = DATA_ROOT / 'stimuli' / language
PREDICTOR_DIR = DATA_ROOT / 'predictors' / language
list_stories = [f.split('_normalized')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('.wav')]

PREDICTOR_DIR.mkdir(exist_ok=True)


for i in list_stories:
    dst = STIMULUS_DIR / f'{i}-gammatone.pickle'
    if dst.exists():
        continue
    wav = load.wav(STIMULUS_DIR / f'{i}_normalized.wav')
    wav = resample(wav, 11025)
    gt = gammatone_bank(wav, 20, 5000, 256, location='left', pad=False)
    gt = resample(gt, 1000)
    save.pickle(gt, dst)
