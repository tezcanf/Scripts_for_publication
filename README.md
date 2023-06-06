# A-tradeoff-between-acoustic-and-linguistic-feature-encoding-
Codes used in the paper "A tradeoff between acoustic and linguistic feature encoding in spoken language comprehension"

- Generate .csv files that contains word and phoneme onsets of stories
1) Run WebMaus forced aligner to generate .TextGrid files
2) Run extract_phonemes_timing.py
3) Run extract_word_timing.py
4) Run word_phoneme_bridge.py

- Generate Cohor Model, add word frequencies and word entropy by using GPT2 model
1) Run Clean_freq_file.py for both Dutch and French
2) Run Cohort_model.py for both Dutch and French
3) Run Word_entropy_GPT.py for both Dutch and French
4) Manually correct for the missing word frequency values by replacing NaN values with the mean frequency
5) Run Word_entropy_low_vs_high.py
6) Run Generate_High_Low_entropy_cont_arrays.py

- Generate Predictors 
1) Run make_gammatone.py
2) Run make_gammatone_predictors.py
3) Run make_word_predictors.py

- Generate TRF models
1) Run estimate_trfs.py from Scripts_for_publication/TRFs/Generate_TRF_models/sub001/ (In utils_TRF.py set Sources_saved = False for the first time you run the TRF model so it saves the source reconstructed signal for the next TRF models. Then set it to True)

- Accuracy Analysis
1) Run Whole_brain_accuracies_basic_models.py, Word_entropy_effect_acoustic_features.py and Word_entropy_effect_phoneme_features.py to generate .csv files.
2) Run LMM in R for further analysis

- TRF Weights Statistical Analysis
1) Run ANOVA_weigths_all_features.py and ANOVA_weigths_phoneme_features.py for Part 1 and Part 2.
2) Run Visualize_brains_all_phoneme_features.py to generate contrast on brain surface and the statistical analysis report. 

- Generate TRF graphs
1) Run TRF_LH.py and TRF_RH.py for Part 1 and Part 2

