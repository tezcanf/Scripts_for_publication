#!/usr/bin/env python
# -*- coding: utf-8 -*-

import textgrid as tg
import pandas as pd

def textGrid_to_dataFrame(tgfile, tier='words'):
    """
    Simply return a dataframe with the tier mark label aligned with time onset and offset.

    Parameters
    ----------
    tgfile : str
        Path to TextGrid file
    tier : str (optional)
        Which tier to extract in the TextGrid data (default: "words").

    Returns
    -------
    df : pd.DataFrame
        Columns of dataframe are ["onset", "offset", tier]
    """
    data = tg.TextGrid.fromFile(tgfile)
    phonsTier = data.getFirst('ORT-MAU')
    ton, toff, words = zip(*[(t.minTime, t.maxTime, t.mark) for t in phonsTier.intervals])
    return pd.DataFrame({'onset': ton, 'offset': toff, tier: words})

def show_available_tier(tgfile):
    "Simply show what tier are present in the file"
    data = tg.TextGrid.fromFile(tgfile)
    return data.getNames()

if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    language='Dutch' # or 'French'
          
    textgrid_folder = os.path.join(dir_path,  'raw_data', language, 'TextGrid_Alignments') 
    word_folder= os.path.join(dir_path,  'raw_data', language, 'Word_transcription_of_stories')
    
    if not os.path.exists(word_folder): #Create the folder if not exists
        os.mkdir(word_folder)
    
    stories = [f.split('.')[0] for f in os.listdir(textgrid_folder) if f.endswith('.TextGrid')]
    
    for s in stories:
        print("Processing %s"%s)
        tgfile = os.path.join(textgrid_folder, s+'.TextGrid')
        df = textGrid_to_dataFrame(tgfile)
        df.to_csv(os.path.join(word_folder,s+'_words.csv'))
    print("Done")
    
