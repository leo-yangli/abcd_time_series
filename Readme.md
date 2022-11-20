# ABCD Time Series

This repository contains the code for [Accounting for Temporal Variability in Functional Magnetic Resonance Imaging Improves Prediction of Intelligence](https://arxiv.org/abs/2211.07429).


## Requirements
    pytorch>1.0.0
    tqdm
    numpy
    scipy
    matplotlib
    pandas

# Data Preparing
Convert each individual's fMRI data into a single numpy file (shape: time_steps * # of features). fMRI data from different tasks should be put in separate folders.

    |-ROOT
     |-- MID
     |-- NBACK
     |-- SST
     |-- REST
      |- NDAR_INVPG04NJDC.npy
      |- ...

## Usage
Train fMRI data example

    python train_fmri.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --data_root [DATA_DIR]

Train with feature selection example

    python fea_slct.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --regcoef 0.25 --data_root [DATA_DIR]

Feature selection fine-tuning/evaluation example

    python fea_slct_eval.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --model_path [MODEL_PATH] --data_root [DATA_DIR]

        
## Citation
If you found this code useful, please cite our paper.

    @misc{abcd_time_series,
        url = {https://arxiv.org/abs/2211.07429},
        author = {Li, Yang and Ma, Xin and Sunderraman, Raj and Ji, Shihao and Kundu, Suprateek},
        title = {Accounting for Temporal Variability in Functional Magnetic Resonance Imaging Improves Prediction of Intelligence},
        year = {2022},
    }
