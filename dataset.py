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
        super().__init__()
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
    def __init__(self, labels, sampling_mode='same', random_state=None):
        """__init__
 
         :param labels: numpy array ( or dataframe ) of shape n_samples, n_labels
         :param sampling_mode: same or all. In mode 'same' we will only randomly sample n_samples, in mode all we oversample to the "most common" seen clip.
         :param random_state: RandomState
         """
        self.labels = labels
        assert labels.ndim == 1, "Only cross-entropy labels expected"
        data_samples = labels.shape[0]
        label_names = np.unique(labels)
        self.label_to_idx_list = []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in label_names:
            label_indexes = np.where(labels == lb_idx)[0]
            self.random_state.shuffle(label_indexes)
            self.label_to_idx_list.append(label_indexes)

        data_source = []
        self.random_state.shuffle(self.label_to_idx_list)
        for sample_idx, indexes in enumerate(
                itertools.zip_longest(*self.label_to_idx_list)):
            indexes = np.array(indexes)
            to_pad_indexes = np.where(indexes == None)[0]
            for idx in to_pad_indexes:
                indexes[idx] = random.choice(self.label_to_idx_list[idx])
            data_source.append(indexes)
        self.data_source = np.array(data_source)
        if sampling_mode == 'same':
            self.data_length = data_samples
        elif sampling_mode == 'over':  # Sample all items
            self.data_length = np.prod(self.data_source.shape)

    def __iter__(self):
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(
            self.data_source[random_indices])[:self.data_length]
        return iter(data)

    def __len__(self):
        return self.data_length


def getdataloader(data_frame,
                  data_file,
                  transform=None,
                  colname=None,
                  **dataloader_kwargs):

    dset = HDF5Dataset(data_file,
                       data_frame,
                       colname=colname,
                       transform=transform)

    return tdata.DataLoader(dset,
                            collate_fn=sequential_collate,
                            **dataloader_kwargs)


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
                data_seq[0]) is tuple or isinstance(
                    data_seq[0],
                    np.int64):  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs


if __name__ == '__main__':
    pass
