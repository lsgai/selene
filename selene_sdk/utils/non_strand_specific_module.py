"""
This module provides the NonStrandSpecific class.
"""
import torch
from torch.nn.modules import Module

from . import _is_lua_trained_model

import itertools
import numpy as np

def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.

    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
            torch.arange(x.size(1)-1, -1, -1),
            ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def revAndCom(input, useTensor=True, reverseAxis=1):
    '''if useTensor, expects input to be a tensor, returns reverse complement as tensor.
    reverseAxis: axis to reverse, 0 if input is 1d, 1 if input is batch'''
    if useTensor:
        reverse_input = torch.empty_like(input) 
        # not sure if contiguous matters as long as it's a copy
        # put 3 wherever input has 0, etc.
        reverse_input[input==0]=3; reverse_input[input==3]=0;
        reverse_input[input==1]=2; reverse_input[input==2]=1;
        # then reverse each sample
        reverse_input = _flip(reverse_input,1)
    else:
        reverse_input = np.array(input)
        # not sure if contiguous matters as long as it's a copy
        # put 3 wherever input has 0, etc.
        reverse_input[input==0]=3; reverse_input[input==3]=0;
        reverse_input[input==1]=2; reverse_input[input==2]=1;
        # then reverse each sample
        reverse_input = np.flip(reverse_input, axis=reverseAxis)
    return reverse_input

def makeReverseComKmerD(kmer_size):
    # kmerD[tup] = kmerIndex
    kmerL = list(itertools.product(range(4), repeat=kmer_size))
    kmerD = {}
    for kmer, i in zip(kmerL, range(len(kmerL))):
        kmerD[kmer] = i
    kmerD["padding"] = len(kmerL)

    # reverse_kmerD[ kmerIndex ] = index of reverse complement tuple
    reversecom_kmerD = {}
    for (fwd_kmer, i) in kmerD.items():
        # get reverse complement of fwd_kmer, then look up its index
        if fwd_kmer == "padding":
            reversecom_kmerD[i] = len(kmerL) # padding is largest index + 1
        else:
            fwd_kmer = torch.Tensor(fwd_kmer)
            revcomTup = tuple(revAndCom(fwd_kmer, useTensor=False, reverseAxis=0))
            revcom_i = kmerD[revcomTup]
            reversecom_kmerD[i] = kmerD[revcomTup]

    return reversecom_kmerD



class NonStrandSpecific(Module):
    """
    A torch.nn.Module that wraps a user-specified model architecture if the
    architecture does not need to account for sequence strand-specificity.

    Parameters
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}, optional
        Default is 'mean'. NonStrandSpecific will pass the input and the
        reverse-complement of the input into `model`. The mode specifies
        whether we should output the mean or max of the predictions as
        the non-strand specific prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}
        How to handle outputting a non-strand specific prediction.
    input_format : {'onehot', 'index', 'kmer_index'}
        Whether sequence is one-hot encoded or integer indices, or kmer
    kmer_size : int
        size of kmer, 0 if not kmer
    """

    def __init__(self, model, mode="mean", input_format="onehot", kmer_size=0):
        super(NonStrandSpecific, self).__init__()
        print("kmer size", kmer_size)
        self.model = model

        if mode != "mean" and mode != "max":
            raise ValueError("Mode should be one of 'mean' or 'max' but was"
                             "{0}.".format(mode))
        self.mode = mode
        self.input_format = input_format
        self.kmer_size = kmer_size
        self.from_lua = _is_lua_trained_model(model)
        self.reversecom_kmerD = None # reversecom_kmerD[ kmerIndex ] = index of reverse complement tuple
        if self.kmer_size > 0:
            self.reversecom_kmerD = makeReverseComKmerD(self.kmer_size)

    def forward(self, input):
        reverse_input = None

        #print("Nonstrand: kmersize and format", self.kmer_size, self.input_format)
        if self.input_format == "index" and self.kmer_size == 0:
            # one-hot is in ACGT = 0123 order (or some other order such that 0,3 and 1,2 are complement)
            reverse_input = revAndCom(input)

        elif self.kmer_size > 0: # convert to kmer after getting reverse
            # there's prob a more efficient way to do thus
            reverse_input = torch.empty_like(input)
            for i, revcom_i in self.reversecom_kmerD.items():
                reverse_input[input==i] = revcom_i
            reverse_input = _flip(reverse_input, 1)

        elif self.from_lua: 
            reverse_input = _flip(
                _flip(torch.squeeze(input, 2), 1), 2).unsqueeze_(2)
            reverse_input = _flip(_flip(input, 1), 2)
        else: # otherwise input shape is (batch,4,1000)
            reverse_input = _flip(_flip(input, 1), 2)

        #print('Nonstrand forward')
        #print('in', input[0:5, 0:10])
        #print('rev', reverse_input[0:5, 0:10])
        output = self.model.forward(input)
        output_from_rev = self.model.forward(reverse_input)

        if self.mode == "mean":
            return (output + output_from_rev) / 2
        else:
            return torch.max(output, output_from_rev)

