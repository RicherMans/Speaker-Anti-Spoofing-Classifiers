# Speaker-Anti-Spoofing-Classifiers

This repository provides a basic speaker anti-spoofing system using neural networks in pytorch.

Python requirements are:

```
pandas==0.25.3
tqdm==4.28.1
torch==1.3.1
matplotlib==3.1.1
numpy==1.17.4
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

Moreover, to download the data, you will need `wget`.


Evaluation scripts are directly taken from the baseline of the ASVspoof2019 challenge, seen [here](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip)


## Datasets

The most broadly used datasets for spoofing detection (currently) are:

* [ASVspoof2019](https://datashare.is.ed.ac.uk/handle/10283/3336) encompassing logical and physical attacks alike.
* [ASVspoof2017](https://datashare.is.ed.ac.uk/handle/10283/3055) encompassing only physical attacks with the focus on `in the wild` devices and scenes.
* [ASVspoof2015](https://datashare.is.ed.ac.uk/handle/10283/853) encompassing only logical attacks with the focus on synthesize and voice conversion attacks.

For mixed logical and physical attacks, the mixed [AVspoof](https://www.idiap.ch/dataset/avspoof) dataset ( a subpart of it is the BTAS16 dataset) can be also used. The dataset is publicly available, but only for research purposes.

Moreover, rather recent the Fake-or-Real (FoR) dataset was introduced using openly available synthesizers such as Baidu, Google, Amazon to create spoofs for logical access.

## Feature extraction

Features are extracted using the [librosa](https://github.com/librosa/librosa) toolkit. We provide four commonly used features: `CQT`,(Linear)log-`Spectrograms`, log-`Mel-Spectrograms` and `raw wave` features.

## Models

Currently, the (what I think) most popular model is [LightCNN](https://arxiv.org/abs/1511.02683), which is the winner of the ASVspoof2017 challenge [paper](https://pdfs.semanticscholar.org/a2b4/c396dc1064fb90bb5455525733733c761a7f.pdf).
Another model called [CGCNN](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2170.pdf), which modified the `MFM` activation to gated linear unit (GLU) activations has been successfully employed in the ASVspoof2019 challenge.
My current implementation can be seen in `models.py`.

All experiments conducted in this repository use 90% of the training set as training data and 10% as cross-validation. 
Development data is not used at all.
All results are only evaluated on the respective evaluation set.

The baseline results are as follows:

| Dataset  | Feature | Model    | EER   |
|----------|---------|----------|-------|
| ASV15    | Spec    | LightCNN | 2.24|
| ASV15    | Spec    | CGCNN    | 0.36  |
| ASV17    | Spec    | LightCNN | 12.18 |
| ASV17    | Spec    | CGCNN    | 9.55  |
| BTAS16   | Spec    | LightCNN | 2.17  |
| BTAS16   | Spec    | CGCNN    | 3.11  |
| FoR-norm | Spec    | LightCNN | 15.49 |
| FoR-norm | Spec    | CGCNN    | 5.69  |



## Config

The framework here uses a combination of [google-fire](https://github.com/google/python-fire) and `yaml` parsing to enable a convenient interface for anti-spoofing model training.
By default one needs to pass a `*.yaml` configuration file into any of the command scripts. However parameters of the `yaml` files can also be overwritten on the fly:

`python3 run.py train config/asv17.yaml --model MYMODEL`, searches for the model `MYMODEL` in `models.py` and runs the experiment using that model.

Other notable arguments are:

* `--model_args '{filters:[60,60]}'` sets the filter sizes of a convolutional model to `60, 60`.
* `--batch_size 32 --num_workers 2'` sets training hyper parameters `batch_size` as well as the number of async workers for dataloading.
* `--transforms '[timemask,freqmask]'` applies augmentation on the training data, defined in `augment.py`.


## Commands

The main script of this repository is `run.py`. Five commands are available:

* `train` e.g., `python3 run.py train config/asv17.yaml` trains a specified model on specified data.
* `score` e.g., `python3 run.py score EXPERIMENT_PATH  OUTPUTFILE.tsv --testlabel data/filelists/asv17/eval.tsv --testdata data/hdf5/asv17/spec/eval.h5` scores a given experiment and produces `OUTPUTFILE.tsv` containing the respective scores. End-to-End scoring is utilized, where the `genuine` class scores are representative of the model belief. Only a single dataset can be scored.
* `evaluate_eer` uses the library contained in `evaluation/` to calculate an EER. Example usage is: `python3 run.py evaluate_eer experiments/asv17/LightCNN/SOMEEXPERIEMNT/scored.txt data/filelists/asv17/eval.tsv output.txt`. `output.txt` is generated with the results ( which are also printed to console ).
* `run` e.g., `python3 run.py run config/asv17.yaml` trains, scores and evaluates an experiment. Can also support multiple tests using `--testlabel ['a.tsv','b.tsv]'` or just updating the config file. Is effectively `train, score, evaluate_eer` in one and the most recommended way of running any experiment.


## Usage

For a simple e.g., ASVspoof2017 dataset run, please run the following:


```bash
pip3 install -r requirements.txt
cd data/scripts
bash download_asv17.sh
bash prepare_asv17.sh
cd ../../features/
python3 extract_feature.py ../data/filelists/asv17/train.tsv -o hdf5/asv17/spec/train.h5 # Extracts spectrogram features
python3 extract_feature.py ../data/filelists/asv17/eval.tsv -o hdf5/asv17/spec/eval.h5 #Spectrogram features
cd ../
python3 run.py run config/asv17.yaml # Runs LightCNN Model. Results will be displayed in the console and a directory experiments/asv17 will be created.
python3 run.py run config/asv17.yaml --model CGCNN # Runs CGCNN Model
```
