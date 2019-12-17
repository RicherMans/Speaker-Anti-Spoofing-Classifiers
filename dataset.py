import torch
import numpy as np
import pandas as pd
import h5py
import itertools, random
import logging
import torch.utils.data as tdata


class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: h5py.File,
                 labels: pd.DataFrame,
                 transform=None,
                 colname=('filename', 'encoded')):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self.dataset = None
        self._labels = labels
        assert len(colname) == 2
        self._colname = colname
        self._len = len(self._labels)
        # IF none is passed still use no transform at all
        self._transform = transform
        with h5py.File(self._h5file, 'r') as store:
            fname, _ = self._labels.iloc[0].reindex(self._colname)
            self.datadim = store[str(fname)].shape[-1]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self._h5file, 'r')
        fname, target = self._labels.iloc[index].reindex(self._colname)
        data = self.dataset[str(fname)][()]
        data = torch.as_tensor(data)
        if self._transform:
            data = self._transform(data)
        return data, target, fname


class MinimumOccupancySampler(tdata.Sampler):
    """
        docstring for MinimumOccupancySampler
        samples at least one instance from each class sequentially
    """
    def __init__(self, labels, sampling_factor=1, random_state=None):
        self.labels = labels
        n_samples, n_labels = labels.shape
        self.label_to_idx_list = []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in range(n_labels):
            label_indexes = np.where(labels[:, lb_idx] == 1)[0]
            self.random_state.shuffle(label_indexes)
            self.label_to_idx_list.append(label_indexes)

        data_source = []
        for _ in range(sampling_factor):
            self.random_state.shuffle(self.label_to_idx_list)
            for indexes in itertools.zip_longest(*self.label_to_idx_list):
                indexes = np.array(indexes)
                to_pad_indexes = np.where(indexes == None)[0]
                for idx in to_pad_indexes:
                    indexes[idx] = random.choice(self.label_to_idx_list[idx])
                data_source.append(indexes)
        self.data_source = np.array(data_source)
        self.data_length = np.prod(self.data_source.shape)

    def __iter__(self):
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(self.data_source[random_indices])
        return iter(data)

    def __len__(self):
        return self.data_length


def getdataloader(data_frame,
                  data_file,
                  transform=None,
                  colname=None,
                  sampler=None,
                  batch_size=16,
                  shuffle=False,
                  num_workers=2):

    dset = HDF5Dataset(data_file,
                       data_frame,
                       colname=colname,
                       transform=transform)

    return tdata.DataLoader(dset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            sampler=sampler,
                            num_workers=num_workers,
                            collate_fn=sequential_collate)


def pad(tensorlist, batch_first=True, padding_value=0., min_length=501):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    if padded_seq.shape[1] < min_length:  # Pad to a minimum length
        new_pad = torch.zeros(size=(padded_seq.shape[0], min_length,
                                    padded_seq.shape[2]))
        new_pad[:, :padded_seq.shape[1], :] = padded_seq

    return padded_seq


def sequential_collate(batches):
    # sort length wise
    # batches.sort(key=lambda x: len(x), reverse=True)
    seqs = []
    for data_seq in zip(*batches):
        if isinstance(data_seq[0],
                      (torch.Tensor, np.ndarray)):  # is tensor, then pad
            data_seq = pad(data_seq)
        elif type(data_seq[0]) is list or type(
                data_seq[0]) is tuple:  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs


if __name__ == '__main__':
    labels = pd.read_csv('features/flists/weak.csv', sep='\t')
    import os
    labels['filename'] = labels['filename'].apply(os.path.basename)
    dloader = getdataloader(labels,
                            'features/hdf5/weak.h5',
                            num_workers=2,
                            transform=ToTensor())

    from tqdm import tqdm
    for d in tqdm(dloader):
        pass
