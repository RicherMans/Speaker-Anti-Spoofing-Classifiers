# Data processing scripts

For each dataset ASV{15, 17, 19} and FoR datasets, here are some scripts to download and pre-process the data for later feature extraction.

For most usecases, just firstly download a dataset e.g., `download_asv17.sh` and then run `prepare_asv17.sh`.

If for some reason, you have downloaded all the dataset files into another directory, just pass that root directory to the `prepare*` script.


For all datasets except the [BTAS16](https://www.idiap.ch/dataset/avspoof) dataset, which is a subset of the larger and multi-modal `AVspoof` one, the downloads are available directly. For the BTAS16 dataset one needs to first agree with their EULA in order to download the dataset.

I recommend to start with the ASVspoof2017 dataset, since it is rather small and experiments can be conducted on a CPU without GPU acceleration.

## Datasets

| Name            | \# Utterances | Size (extracted) |
|-----------------|---------------|------------------|
| FoR-norm        | 69303         | 6.6 Gb           |
| ASVspoof2015    | 263151        | 29 Gb            |
| ASVspoof2017    | 18032         | 1.8 Gb           |
| ~~BTAS16~~      | 137624        | 19.5 Gb          |
| ASVspoof2019-LA | 122302        | 7.5 Gb           |
| ASVspoof2019-PA | 241059        | 17.2 Gb          |
