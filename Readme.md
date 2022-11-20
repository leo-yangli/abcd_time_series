# L0-ARM

This repository contains the code for [Accounting for Temporal Variability in Functional Magnetic Resonance Imaging Improves Prediction of Intelligence](https://arxiv.org/abs/2211.07429).


## Requirements
    pytorch>1.0.0
    tqdm
    numpy
    scipy
    matplotlib
    pandas

# Data Preparing
Convert each individual's fMRI data into a single numpy file. fMRI data from different tasks should be put in separate folders.

    |-ROOT
     |-- MID
     |-- NBACK
     |-- SST
     |-- REST
      |- NDAR_INVPG04NJDC.npy
      |- ...

## Usage
Train fMRI Data Example

    python train_fmri.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --data_root [DATA_DIR]

Feature Selection Example

    python fea_slct.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --regcoef 0.25 --data_root [DATA_DIR]

Feature Selection Fine-tuning Example

    python fea_slct_eval.py --task mid --target nihtbx_cryst_uncorrect --test_fold 0 --model_path [MODEL_PATH] --data_root [DATA_DIR]

        
## Citation
If you found this code useful, please cite our paper.

    @misc{lstm_abcd,
        doi = {10.48550/ARXIV.2211.07429},
        url = {https://arxiv.org/abs/2211.07429},
        author = {Li, Yang and Ma, Xin and Sunderraman, Raj and Ji, Shihao and Kundu, Suprateek},
        title = {Accounting for Temporal Variability in Functional Magnetic Resonance Imaging Improves Prediction of Intelligence},
        publisher = {arXiv},
        year = {2022},
    }