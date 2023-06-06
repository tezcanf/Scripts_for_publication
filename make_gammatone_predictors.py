"""Predictors based on gammatone spectrograms"""
from pathlib import Path
import os
import numpy as np
from eelbrain import *
from trftools.neural import edge_detector

language = 'French'
DATA_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).expanduser() 
STIMULUS_DIR = DATA_ROOT / 'stimuli' / language
PREDICTOR_DIR = DATA_ROOT / 'predictors' / language
list_stories = [f.split('_normalized')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('.wav')]

PREDICTOR_DIR.mkdir(exist_ok=True)


for i in list_stories:
    gt = load.unpickle(STIMULUS_DIR / f'{i}-gammatone.pickle')

    # Remove resampling artifacts
    gt = gt.clip(0, out=gt)
    # apply log transform
    gt = (gt + 1).log()
    # generate on- and offset detector model
    gt_on = edge_detector(gt, c=30)

    # 1 band predictors
    save.pickle(gt.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-1.pickle')
    save.pickle(gt_on.sum('frequency'), PREDICTOR_DIR / f'{i}~gammatone-on-1.pickle')
    # 8 band predictors
    x = gt.bin(nbins=8, func=np.sum, dim='frequency')
    save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-8.pickle')
    x = gt_on.bin(nbins=8, func=np.sum, dim='frequency')
    save.pickle(x, PREDICTOR_DIR / f'{i}~gammatone-on-8.pickle')
    
gammatone = [gt.sum('frequency')]
# Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
gammatone = [x.bin(0.005, dim='time', label='start') for x in gammatone]
# Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
gammatone = [trftools.pad(x, tstart=0, tstop=x.time.tstop, name='gammatone') for x in gammatone]
# Load the broad-band envelope and process it in the same way