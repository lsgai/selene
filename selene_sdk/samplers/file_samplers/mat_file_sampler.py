"""
This module provides the `MatFileSampler` class and its supporting
methods.
"""
import h5py
import numpy as np
import scipy.io

from .file_sampler import FileSampler

import itertools

def _load_mat_file(filepath, sequence_key, targets_key=None):
    """
    Loads data from a `*.mat` file or a `*.h5` file.

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    sequence_key : str
        The key for the sequences data matrix.
    targets_key : str, optional
        Default is None. The key for the targets data matrix.

    Returns
    -------
    (sequences, targets, h5py_filehandle) : \
            tuple(array-like, array-like, h5py.File)
        If the matrix files can be loaded with `scipy.io`,
        the tuple will only be (sequences, targets). Otherwise,
        the 2 matrices and the h5py file handle are returned.

    """
    try:  # see if we can load the file using scipy first
        mat = scipy.io.loadmat(filepath)
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return (mat[sequence_key], targets)
    except (NotImplementedError, ValueError):
        mat = h5py.File(filepath, 'r')
        sequences = mat[sequence_key]
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return (sequences, targets, mat)



def convert2kmer(idx_sequences, k, kmerD):
    '''given sequences of indices 0,1,2,3, return seq of kmer encoded as indices'''

    # make dict of all possible kmers and corresponding index. maxKmerIdx + 1 = padding
    if k > 8:
        print('Error: Are you sure you want kmers of len 9+? 4^9 = 260k combinations. \
               Comment out line if sure')
        exit(1)

    # preserve original seqlen (pad with max idx at end)
    n_sequences, orig_seqlen = idx_sequences.shape
    kmer_sequences = len(kmerD.keys())*np.ones(idx_sequences.shape)
    # convert to kmer index. sliding window, pad instead of partial kmers
    idx_sequences = idx_sequences
    for pos in range(orig_seqlen-k+1):
        subset = idx_sequences[:,pos:(pos+k)] 
        kmer_sequences[:,pos] = [kmerD[tuple(kmer)] for kmer in subset]
    return kmer_sequences


class MatFileSampler(FileSampler):
    """
    A sampler for which the dataset is loaded directly from a `*.mat` file.

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    sequence_key : str
        The key for the sequences data matrix.
    targets_key : str, optional
        Default is None. The key for the targets data matrix.
    random_seed : int, optional
        Default is 436. Sets the random seed for sampling.
    shuffle : bool, optional
        Default is True. Shuffle the order of the samples in the matrix
        before sampling from it.
    sequence_batch_axis : int, optional
        Default is 0. Specify the batch axis.
    sequence_alphabet_axis : int, optional
        Default is 1. Specify the alphabet axis.
    targets_batch_axis : int, optional
        Default is 0. Speciy the batch axis.
    convert_to_index: int, optional
        If non-zero, convert one hot of nucleotides to index.
    convert_to_kmer_size: int, optional
        If non-zero, convert to kmers of that size

    Attributes
    ----------
    n_samples : int
        The number of samples in the data matrix.
    """

    def __init__(self,
                 filepath,
                 sequence_key,
                 targets_key=None,
                 random_seed=436,
                 shuffle=True,
                 sequence_batch_axis=0,
                 sequence_alphabet_axis=1,
                 targets_batch_axis=0,
                 convert_to_index=0, 
                 convert_to_kmer_size=0):
        """
        Constructs a new `MatFileSampler` object.
        """
        super(MatFileSampler, self).__init__()
        out = _load_mat_file(
            filepath,
            sequence_key,
            targets_key=targets_key)
        self._sample_seqs = out[0]
        self._sample_tgts = out[1]
        self._mat_fh = None
        if len(out) > 2:
            self._mat_fh = out[2]
        self._seq_batch_axis = sequence_batch_axis
        self._seq_alphabet_axis = sequence_alphabet_axis
        self._seq_final_axis = 3 - sequence_batch_axis - sequence_alphabet_axis
        if self._sample_tgts is not None:
            self._tgts_batch_axis = targets_batch_axis
        self.n_samples = self._sample_seqs.shape[self._seq_batch_axis]

        self._sample_indices = np.arange(
            self.n_samples).tolist()
        self._sample_next = 0

        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._sample_indices)

        self._convert_to_index = convert_to_index
        self._convert_to_kmer_size = convert_to_kmer_size
        if self._convert_to_kmer_size > 0:
            self._convert_to_index = True  # currently coded so kmer generated from index
            
        self._kmerD = None # kmerD[ [ (0,1,2) ] = corresponding index 
        if self._convert_to_kmer_size > 0:
            kmerL = list(itertools.product(range(4), repeat=self._convert_to_kmer_size))
            kmerD = {}
            for kmer, i in zip(kmerL, range(len(kmerL))):
                kmerD[kmer] = i
            self._kmerD = kmerD



    def sample(self, batch_size=1):
        """
        Draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.
        """
        sample_up_to = self._sample_next + batch_size
        use_indices = None
        if sample_up_to >= len(self._sample_indices):
            if self._shuffle:
                np.random.shuffle(self._sample_indices)
            self._sample_next = 0
            use_indices = self._sample_indices[:batch_size]
        else:
            use_indices = self._sample_indices[self._sample_next:sample_up_to]
        self._sample_next += batch_size
        use_indices = sorted(use_indices)
        if self._seq_batch_axis == 0:
            sequences = self._sample_seqs[use_indices, :, :].astype(float)
        elif self._seq_batch_axis == 1:
            sequences = self._sample_seqs[:, use_indices, :].astype(float)
        else:
            sequences = self._sample_seqs[:, :, use_indices].astype(float)

        if self._seq_batch_axis != 0 or self._seq_alphabet_axis != 2:
            sequences = np.transpose(
                sequences, (self._seq_batch_axis,
                            self._seq_final_axis,
                            self._seq_alphabet_axis))

        if self._convert_to_index:
            # convert one-hot to index, where last axis is alphabet (also required for kmer)
            sequences = np.argmax(sequences, axis=-1)
        if self._convert_to_kmer_size:
            # then convert indices to kmer indices
            sequences = convert2kmer(sequences, self._convert_to_kmer_size, self._kmerD)
            #print("mat file sampler: sequences after convert2kmer")
            #print(sequences[0:2,0:4])
            #print(sequences.shape)

        if self._sample_tgts is not None:
            if self._tgts_batch_axis == 0:
                #print("mat file sampler: _sample_tgts")
                #print(self._sample_tgts[0:2,0:4])
                #print(self._sample_tgts.shape)
                targets = self._sample_tgts[use_indices, :].astype(float)
            else:
                targets = self._sample_tgts[:, use_indices].astype(float)
                targets = np.transpose( targets, (1, 0) )
            return (sequences, targets)
            
        return sequences,


    def get_data(self, batch_size, n_samples=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int, optional
            Default is None. The total number of samples to retrieve.

        Returns
        -------
        sequences : list(np.ndarray)
            The list of sequences grouped into batches.
            An element in the `sequences` list is of
            the shape :math:`B \\times L \\times N`, where :math:`B`
            is `batch_size`, :math:`L` is the sequence length,
            and :math:`N` is the size of the sequence type's alphabet.
        """
        if not n_samples:
            n_samples = self.n_samples
        sequences = []

        count = batch_size
        while count < n_samples:
            seqs, = self.sample(batch_size=batch_size)
            sequences.append(seqs)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, = self.sample(batch_size=remainder)
        sequences.append(seqs)
        return sequences

    def get_data_and_targets(self, batch_size, n_samples=None):
        """
        This method fetches a subset of the sequence data and
        targets from the sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int, optional
            Default is None. The total number of samples to retrieve.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.
        """
        if self._sample_tgts is None:
            raise ValueError(
                "No targets matrix was specified during sampler "
                "initialization. Please use `get_data` instead.")
        if not n_samples:
            n_samples = self.n_samples
        sequences_and_targets = []
        targets_mat = []

        count = batch_size
        while count < n_samples:
            seqs, tgts = self.sample(batch_size=batch_size)
            sequences_and_targets.append((seqs, tgts))
            targets_mat.append(tgts)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, tgts = self.sample(batch_size=remainder)
        sequences_and_targets.append((seqs, tgts))
        targets_mat.append(tgts)
        # TODO: should not assume targets are always integers
        targets_mat = np.vstack(targets_mat).astype(float)
        return sequences_and_targets, targets_mat
