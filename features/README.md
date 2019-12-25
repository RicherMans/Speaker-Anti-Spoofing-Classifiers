# Feature extraction

Please first obtain the data and run `prepare*.sh` for the respective dataset in `data/scripts`.

Four different standard features can be extracted:


|Feature | Frame 


## Examples


### Spectrogram features (default)

```bash
# Extracting asvspoof2017 spectrogram features
python3 extract_features.py ../data/filelists/asv17/train.tsv -o hdf5/asv17/spec/train.h5
python3 extract_features.py ../data/filelists/asv17/dev.tsv -o hdf5/asv17/spec/dev.h5
python3 extract_features.py ../data/filelists/asv17/eval.tsv -o hdf5/asv17/spec/eval.h5
```


### Log-Mel spectrogram features


```bash
# Extracting asvspoof2017 spectrogram features
python3 extract_features.py ../data/filelists/asv17/train.tsv -o hdf5/asv17/spec/train.h5 --feat lms
python3 extract_features.py ../data/filelists/asv17/dev.tsv -o hdf5/asv17/spec/dev.h5 --feat lms
python3 extract_features.py ../data/filelists/asv17/eval.tsv -o hdf5/asv17/spec/eval.h5 --feat lms
```


### Log constant q-transform (CQT) features


```bash
# Extracting asvspoof2017 spectrogram features
python3 extract_features.py ../data/filelists/asv17/train.tsv -o hdf5/asv17/spec/train.h5 --feat cqt
python3 extract_features.py ../data/filelists/asv17/dev.tsv -o hdf5/asv17/spec/dev.h5 --feat cqt
python3 extract_features.py ../data/filelists/asv17/eval.tsv -o hdf5/asv17/spec/eval.h5 --feat cqt
```

### Raw wave features


```bash
# Extracting asvspoof2017 spectrogram features
python3 extract_features.py ../data/filelists/asv17/train.tsv -o hdf5/asv17/spec/train.h5 --feat raw
python3 extract_features.py ../data/filelists/asv17/dev.tsv -o hdf5/asv17/spec/dev.h5 --feat raw
python3 extract_features.py ../data/filelists/asv17/eval.tsv -o hdf5/asv17/spec/eval.h5 --feat raw
```
