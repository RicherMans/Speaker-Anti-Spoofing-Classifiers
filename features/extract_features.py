import h5py
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import librosa
import pathlib
from sklearn.preprocessing import StandardScaler
from pypeln import process as pr

EPS = np.spacing(1)  # epsilon for log division by zero avoidance

FEATURES = {
    'cqt':
    lambda y, sr: np.log(
        np.abs(librosa.core.cqt(y, sr=sr, **{'hop_length': 128})) + EPS).T,
    'spec':
    lambda y, sr: np.log(
        np.abs(
            librosa.core.stft(y,
                              sr=sr,
                              **{
                                  'hop_length': 160,
                                  'win_length': 400,
                                  'n_fft': 512
                              })) + EPS).T,
    'lms':
    lambda y, sr: np.log(
        np.abs(
            librosa.feature.melspectrogram(
                y, sr=sr, **{
                    'hop_length': 160,
                    'n_mels': 64
                }))).T,
    'raw':
    lambda y, sr: y.reshape(1, -1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', type=str)
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-sep',
                        default=',',
                        help='Separator for input csvfile')
    parser.add_argument('-f',
                        '--feat',
                        type=str,
                        choices=FEATURES.keys(),
                        default='spec')
    parser.add_argument('-sr', default=16000, type=int)
    parser.add_argument('-c', default=4, type=int)
    parser.add_argument('-cmn', default=False, action='store_true')
    parser.add_argument('-cvn', default=False, action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.csvfile, sep=args.sep,
                     usecols=[0])  #Just use first column

    CMVN_SCALER = StandardScaler(with_mean=args.cmn, with_std=args.cvn)

    feature_fun = FEATURES[args.feat]

    def extract_feature(fname):
        y, sr = librosa.load(fname, sr=args.sr)
        y = feature_fun(y.astype(np.float32), sr)
        y = CMVN_SCALER.fit_transform(y)
        return fname, y

    all_files = df[0].unique()

    output_path = pathlib.Path(args.out)
    if output_path.is_file():
        print("File exists {}. Removing ..".format(output_path.resolve()))
        output_path.unlink()  # Remove if exists
    with h5py.File(args.out, 'w') as hdf5_file, tqdm(total=len(all_files),
                                                     unit='file') as pbar:
        for fname, feat in pr.map(extract_feature,
                                  all_files,
                                  workers=args.c,
                                  maxsize=int(2 * args.c)):
            # Scale feature directly
            hdf5_file[fname] = feat
            pbar.set_postfix(name=fname, shape=feat.shape)
            pbar.update()


if __name__ == "__main__":
    main()
