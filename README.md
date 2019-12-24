# Speaker-Anti-Spoofing-Classifiers

This repository provides a basic speaker anti-spoofing system using neural networks in pytorch.

Requirements are:

```
pandas==0.25.3
tqdm==4.28.1
torch==1.3.1
matplotlib==3.1.1
numpy==1.17.4
tabulate==0.8.6
fire==0.2.1
pytorch_ignite==0.2.1
h5py==2.10.0
six==1.13.0
adabound==0.0.5
ignite==1.1.0
librosa==0.7.1
metrics==0.3.3
pypeln==0.1.10
PyYAML==5.2
scikit_learn==0.22
```



Evaluation scripts are directly taken from the baseline of the ASVspoof2019 challenge, seen [here](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip)


## Datasets

The most broadly used datasets for spoofing detection (currently) are:

* [ASVspoof2019](https://datashare.is.ed.ac.uk/handle/10283/3336) encompassing logical and physical attacks alike
* [ASVspoof2017](https://datashare.is.ed.ac.uk/handle/10283/3055) encompassing only physical attacks with the focus on `in the wild` devices and scenes.
* [ASVspoof2015](https://datashare.is.ed.ac.uk/handle/10283/853) encompassing only logical attacks with the focus on synthesize and voice conversion attacks.

## Feature extraction

Features are extracted using the [librosa](https://github.com/librosa/librosa) toolkit. We provide four commonly used features: `CQT`,(Linear)log-`Spectrograms`, log-`Mel-Spectrograms` and `raw wave` features.

## Models

Currently, the (what I think) most popular model is [LightCNN](https://arxiv.org/abs/1511.02683), which is the winner of the ASVspoof2017 challenge [paper](https://pdfs.semanticscholar.org/a2b4/c396dc1064fb90bb5455525733733c761a7f.pdf).

My current implementation can be seen in `models.py`.
