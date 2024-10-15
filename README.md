# Data and analysis scripts in "A Domain-General Strategy for Hidden-State Inference in Humans and Neural Networks"

## Description of sub-directories and contents

```python
.
├── README.md
├── analysis.m # main analysis
├── /constants # some reused constants used in the analysis code (MATLAB)
│   ├── condrgb.mat
│   ├── constants_ques.mat
│   └── constants_rlinf_sample2.mat
├── /figs # directory for generated figures
├── /fits # directory for files containing model fits
│   └── /merged
│       ├── fit_model_inf_dist_sample2_10111.mat # model fits (test)
│       ├── fit_model_inf_dist_sample2_retest_10111.mat # model fits (retest)
├── preprocess_data.m # function to preprocess raw data
├── /processed
│   ├── age_sex_sample2.mat # age and sex data for participants
│   ├── /sample2
│   │   ├── icar_sample2.mat # ICAR scores (test)
│   │   ├── idx_excl_ques.mat 
│   │   ├── preprocessed_data_sample2.mat # preprocessed behavioral data
│   │   ├── ques_struct_sample2.mat # raw questionnaire data (test)
│   │   └── subj_struct_sample2.mat # raw behavioral data (test)
│   └── /sample2_retest
│       ├── icar_sample2_retest.mat # ICAR scores (retest)
│       ├── idx_excl_ques_retest.mat
│       ├── preprocessed_data_sample2_retest.mat # preprocessed behavioral data (retest)
│       ├── ques_struct_sample2_retest.mat # raw questionnaire data (retest)
│       └── subj_struct_sample2_retest.mat # raw behavioral data (retest)
├── /rnn
│   └── analysis_noisyRNN.ipynb # analysis of trained RNN save files
│   └── eval_noisyRNN.py # functions used in analysis
│   └── run_noisyRNN_bayes.py # RNN training function
│   └── /saved_runs # saved RNN runs stored here
│   └── script_run_noisyRNN.ipynb # example script for running RNNs

├── /sem
│   ├── /dat # contains data to be used in R code for running the SEM 
│   └── sem_icar_sigma.R # SEM analysis code
└── /toolbox # contains various external functions used in analysis
    ├── /fit_functions
    ├── /plot_functions
    ├── /ques_functions
    └── /stat_functions
```

## General usage
- To run behavioral analyses, run the scripts found in `analysis.m`
- To run the SEM, run the script in `sem/sem_icar_sigma.R`
- To train RNNs, use the function `run_noisyRNN(args)` in rnn_noisyRNN_bayes.py with minimally necessary arguments `run_noisyRNN(hidden_size, noise_value, cos_penalty)` as described in the paper. Default  values for the arguments are `hidden_size=64`, `noise_value=0.5`, `cos_penalty=40`.
  - For example usage, see the notebook `script_run_noisyRNN.ipynb`. 


