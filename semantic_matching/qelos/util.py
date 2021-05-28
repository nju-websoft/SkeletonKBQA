import argparse
import collections
import inspect
import re
import os
import signal
import sys
from datetime import datetime as dt
import pickle
import nltk
import traceback
from copy import deepcopy as deepcopy

import numpy as np
import unidecode as unidecode

from IPython import embed
import torch
from functools import partial

import qelos as q
from torch.utils.data import Dataset, DataLoader
import random
from scipy import sparse
from tqdm import tqdm


__all__ = ["ticktock", "argprun", "deep_copy", "copy_params", "seq_pack", "seq_unpack", "iscuda", "hyperparam", "v",
           "intercat", "masked_mean", "tensor_dataset", "datacat", "dataload", "datasplit", "MixedTensorDataset",
           "BucketedRandomBatchSampler", "padclip_collate_fn", "pad_clip",
           "iscallable", "isfunction", "getnumargs", "getkw", "issequence", "iscollection", "isnumber", "isstring",
           "StringMatrix", "tokenize", "recmap", "inf_batches", "set_lr", "remove_lr", "paramgroups_of", "split_dataset",
           "percentagebar"]

# region torch-related utils

def set_lr(m, lr):
    """ attaches a custom learning rate to passed parameter or parameters of passed module
        :param m:   parameter or module
        :param lr:  learning rate for this parameter or module
    """
    if isinstance(m, torch.nn.Module):
        for p in m.parameters():
            set_lr(p, lr)
    elif isinstance(m, torch.nn.Parameter):
        m.__q_lr = lr
    else:
        pass


def remove_lr(m):
    """ removes custom learning rates from this module """
    if isinstance(m, torch.nn.Module):
        for p in m.parameters():
            remove_lr(p)
    elif isinstance(m, torch.nn.Parameter):
        if hasattr(m, "__q_lr"):
            delattr(m, "__q_lr")


def paramgroups_of(m):
    """ gets parameters of given module as parameter groups. useful when set_lr has been used """
    params = m.parameters()
    default_group = {"params": []}
    paramgroups = []
    for param in params:
        g = None
        if hasattr(param, "__q_lr"):
            g = {"params": [param], "lr": param.__q_lr}
        if hasattr(param, "__q_l2"):
            g = {"params": [param], "weight_decay": param.__q_l2}
        if g is None:
            default_group["params"].append(param)
        else:
            paramgroups.append(g)
    paramgroups.append(default_group)
    return paramgroups


def copy_params(source, target):
    """ Copies parameters from source to target such that target has the same parameter values as source.
        (if source params change, so does target's)"""
    for k, v in source.named_parameters():
        ks = k.split(".")
        src_obj = source
        tgt_obj = target
        for _k in ks[:-1]:
            src_obj = getattr(src_obj, _k)
            tgt_obj = getattr(tgt_obj, _k)
        if not isinstance(getattr(src_obj, ks[-1]), torch.nn.Parameter):
            print("Couldn't copy: {}".format(k))
        setattr(tgt_obj, ks[-1], v)


def deep_copy(source, share_params=False):
    tgt = deepcopy(source)
    if share_params:
        copy_params(source, tgt)
    return tgt


# SEQUENCE PACKING AND UNPACKING
def seq_pack(x, mask, ret_sorter=False):  # mask: (batsize, seqlen)
    """ given N-dim sequence "x" (N>=2), and 2D mask (batsize, seqlen)
        returns packed sequence (sorted) and indexes to un-sort (also used by seq_unpack) """
    x = x.float()
    mask = mask.float()
    # 1. get lengths
    lens = torch.sum(mask.float(), 1)
    # 2. sort by length
    assert(lens.dim() == 1)
    _, sortidxs = torch.sort(lens, descending=True)
    unsorter = torch.zeros(sortidxs.size()).to(sortidxs.device).long()
    # print ("test unsorter")
    # print (unsorter)
    unsorter.scatter_(0, sortidxs,
           torch.arange(0, len(unsorter), dtype=torch.int64, device=sortidxs.device))
    # 3. pack
    sortedseq = torch.index_select(x, 0, sortidxs)
    sortedmsk = torch.index_select(mask, 0, sortidxs)
    sortedlens = sortedmsk.long().sum(1)
    # sortedlens = list(sortedlens.cpu().detach().numpy())
    packedseq = torch.nn.utils.rnn.pack_padded_sequence(sortedseq, sortedlens, batch_first=True)

    # 4. return
    if ret_sorter:
        return packedseq, unsorter, sortidxs
    else:
        return packedseq, unsorter


def seq_unpack(x, order, padding_value=0):
    """ given packed sequence "x" and the un-sorter "order",
        returns padded sequence (un-sorted by "order") and a binary 2D mask (batsize, seqlen),
            where padded sequence is padded with "padding_value" """
    unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=padding_value)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.int64, device=unpacked.device)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    out = torch.index_select(unpacked, 0, order)        # same as: unpacked[order]
    outmask = torch.index_select(mask, 0, order)        # same as: mask[order]
    return out, outmask


def iscuda(x):
    if isinstance(x, torch.nn.Module):
        params = list(x.parameters())
        return params[0].is_cuda
    else:
        raise q.SumTingWongException("unsupported type")


class hyperparam(object):
    def __init__(self, initval):
        super(hyperparam, self).__init__()
        self._initval = initval
        self._v = initval

    def reset(self):
        self._v = self._initval

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


def v(x):
    if hasattr(x, "__q_v__"):
        return x.__q_v__()
    elif isinstance(x, hyperparam):
        return x._v
    elif isinstance(x, torch.autograd.Variable):
        raise Exception("autograd.Variable should not be used anymore")
        return x.data
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


def intercat(tensors, axis=-1):
    if axis != -1 and axis != tensors[0].dim()-1:
        tensors = [tensor.transpose(axis, -1) for tensor in tensors]
    t = torch.stack(tensors, -1)
    t = t.view(t.size()[:-2] + (-1,))
    if axis != -1 and axis != tensors[0].dim()-1:
        t = t.transpose(axis, -1)
    return t


def masked_mean(x, dim=None, mask=None, keepdim=False):
    """
    Computes masked mean.
    :param x:           input tensor
    :param mask:        mask
    :param dim:
    :param keepdim:
    :return:
    """
    EPS = 1e-6
    if mask is None:
        return torch.mean(x, dim, keepdim=keepdim)
    else:
        mask = mask.float()
        x = x * mask
        x_sum = torch.sum(x, dim, keepdim=keepdim)
        mask_sum = torch.sum(mask, dim, keepdim=keepdim)
        ret = x_sum / (mask_sum + EPS)
        if mask.size(dim) != x.size(dim):
            assert(mask.size(dim) == 1)
            ret = ret / x.size(dim)
        return ret
# endregion


# region data-related utils
class RandomContiguousBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Samples contiguous batches of elements, choosing a random starting point every time.
    """
    def __init__(self, numexamples, batch_size):
        super(RandomContiguousBatchSampler, self).__init__(None)
        self.numexamples = numexamples
        self.batsize = batch_size

    def __iter__(self):
        randids = torch.randperm(len(self))
        batch = []
        for idx in randids:
            idx = idx.item()
            for i in range(idx * self.batsize,
               min((idx + 1) * self.batsize, self.numexamples)):
                batch.append(torch.tensor(i))
            yield batch
            batch = []

    def __len__(self):
        return self.numexamples // self.batsize + int(self.numexamples % self.batsize > 0)


class BucketedRandomBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Assumes examples are sorted by length.
    Divides numexamples into numbuckets buckets evenly.
    First chooses which bucket to sample from, then samples a batch only from that bucket.
    Best to be used with padclip collate fn.
    """
    def __init__(self, numexamples, numbuckets, batch_size):
        super(BucketedRandomBatchSampler, self).__init__(None)
        self.numexamples, self.numbuckets, self.batsize = numexamples, numbuckets, batch_size
        # divide example ids into buckets
        allids = range(self.numexamples)
        bucketlen = numexamples / numbuckets
        buckets = []
        buckets.append([])
        acc = bucketlen
        for id in allids:
            if round(acc) == 0:
                buckets.append([])
                acc += bucketlen
            buckets[-1].append(id)
            acc -= 1
        self.buckets = buckets

    def __iter__(self):
        # prepare batches
        batches = []
        for bucket in self.buckets:
            random.shuffle(bucket)
            batches.append([])
            for id in bucket:
                if len(batches[-1]) == self.batsize:
                    batches.append([])
                batches[-1].append(torch.tensor(id))
        batchids = list(range(len(batches)))
        random.shuffle(batchids)
        for batchid in batchids:
            outb = batches[batchid]
            outb = sorted(outb, reverse=False)
            yield outb

    def __len__(self):
        acc = 0
        for bucket in self.buckets:
            numbats = len(bucket) // self.batsize + int(len(bucket) % self.batsize > 0)
            acc += numbats
        return acc


def padclip_collate_fn(batch, padidx=0):  # batch is list of things  # TODO: 3D
    batch = torch.utils.data.dataloader.default_collate(batch)
    if not isinstance(batch, collections.Sequence):
        batch = [batch,]
    newbatch = []
    ret = recmap(batch, partial(_padclip_collate_fn_rec, padidx=padidx))
    return ret


def _padclip_collate_fn_rec(batch_e, padidx=0):
    if isinstance(batch_e, torch.Tensor) \
            and batch_e.dtype in (torch.int64, torch.int32, torch.int16):
        return pad_clip(batch_e, padidx=padidx)
    else:
        return batch_e


def pad_clip(tensor, padidx=0):
    """ cuts away contiguous shared pieces of padding index at the end of sequence.
        clips only along the last dimension."""
    if isinstance(tensor, torch.Tensor):
        assert(tensor.dtype in (torch.int64, torch.int32, torch.int16))
        lens = (tensor != padidx).long().sum(-2)
        lens = (lens > 0).long()
        arng = torch.arange(lens.size(-1)).to(lens.device)
        while lens.dim() > arng.dim():
            arng = arng.unsqueeze(0)
        i = lens * arng
        while i.dim() > 0:
            i, _ = i.max(0)
        slices = [slice(None, None, None) for _ in range(tensor.dim() - 1)] \
                 + [slice(None, i+1, None)]
        ret = tensor[tuple(slices)]
        return ret



def inf_batches(dataloader, with_info=True):
    """
    iteration over this produces infinite batches from the dataloader
    returns <batch_data>, (<batch_number>, <epoch_number>) if with_info=True
        else just <batch_data>
    """
    epoch = 0
    while True:
        for i, _batch in enumerate(dataloader):
            if with_info:
                yield _batch, (i, epoch)
            else:
                yield _batch
        epoch += 1


def tensor_dataset(*x):
    """ Creates a torch TensorDataset from list of tensors
        :param x: tensors as numpy arrays or torch tensors
    """
    tensors = []
    for xe in x:
        if isinstance(xe, np.ndarray):
            xe = torch.tensor(xe)
        tensors.append(xe)
    for xe in tensors:
        assert(xe.shape[0] == tensors[0].shape[0])
    ret = torch.utils.data.dataset.TensorDataset(*tensors)
    return ret


class MixedTensorDataset(torch.utils.data.dataset.Dataset):
    """ Like TensorDataset from pytorch but accepts sparse matrices, numpy arrays in addition to torch tensors.
        Converts sp.sparse and numpy arrays to torch tensors in __getitem__()"""
    def __init__(self, *tensors, convert_ints=False):
        """ if convert_ints, automatically converts all uint{8-32} and int{8-32} to int64 """
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.convert_ints = convert_ints

    def __getitem__(self, index):
        ret = tuple()
        for tensor in self.tensors:
            retadd = tensor[index]
            if isinstance(retadd, sparse.spmatrix):
                retadd = retadd.toarray()[0]
            if isinstance(retadd, np.ndarray):
                retadd = torch.tensor(retadd)
            if self.convert_ints and retadd.dtype in (torch.uint8, torch.int8, torch.uint16, torch.int16, torch.uint32, torch.int32):
                retadd = retadd.long()
            ret += (retadd,)
        return ret

    def __len__(self):
        return self.tensors[0].shape[0]


def datacat(datasets, mode=1):
    """
    Concatenates given pytorch datasets. If mode == 0, creates pytorch ConcatDataset, if mode == 1, creates a MultiDataset.
    :return:
    """
    if mode == 0:
        return torch.utils.data.dataset.ConcatDataset(datasets)
    elif mode == 1:
        return MultiDatasets(datasets)
    else:
        raise q.SumTingWongException("mode {} not recognized".format(mode))


class MultiDatasets(Dataset):
    """ A dataset consisting of sub-datasets, to be indexed together. """
    def __init__(self, datasets):
        """ datasets to index together, result will be concatenated in one list """
        for xe in datasets:
            assert(len(xe) == len(datasets[0]))
        super(MultiDatasets, self).__init__()
        self.datasets = datasets

    def __getitem__(self, item):
        ret = tuple()
        for dataset in self.datasets:
            ret_a = dataset[item]
            if not isinstance(ret_a, tuple):
                ret_a = (ret_a,)
            ret += ret_a
        return ret

    def __len__(self):
        return len(self.datasets[0])


def dataload(*tensors, batch_size=1, shuffle=False, **kw):
    """ Loads provided tensors (numpy arrays, torch tensors, or torch datasets) into a torch dataloader.

    """
    if len(tensors) > 0 and isinstance(tensors[0], Dataset):
        if len(tensors) == 1:
            tensordataset = tensors[0]
        else:
            tensordataset = q.datacat(*tensors, mode=1)
    else:
        tensordataset = tensor_dataset(*tensors)
    dataloader = DataLoader(tensordataset, batch_size=batch_size, shuffle=shuffle, **kw)
    return dataloader


from torch._utils import _accumulate
def split_dataset(dataset, lengths=(80, 20), random=True):
    """
    split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        # lengths are proportions
        mult = len(dataset) / sum(lengths)
        lengths = [round(l * mult) for l in lengths]

    if not random:
        indices = torch.arange(0, sum(lengths)).long() #
    else:
        indices = torch.randperm(sum(lengths))
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def datasplit(npmats, splits=(80, 20), random=True):
    """ Splits given numpy arrays according to given split ratio's. Random split if random=True"""
    splits = np.round(len(npmats[0]) * np.cumsum(splits) / sum(splits)).astype("int32")

    whatsplit = np.zeros((len(npmats[0]),), dtype="int64")
    for i in range(1, len(splits)):
        a, b = splits[i-1], splits[i]
        whatsplit[a:b] = i

    if random is not False and random is not None:
        if isinstance(random, int):
            np.random.seed(random)
            random = True

        if random is True:
            np.random.shuffle(whatsplit)

    ret = []
    for i in range(0, len(splits)):
        splitmats = [npmat[whatsplit == i] for npmat in npmats]
        ret.append(splitmats)
    return ret

# endregion


# region other utils
def recmap(x, mapf):      # datastructure, mapping function for elements
    if isinstance(x, dict):
        for k in x:
            x[k] = recmap(x[k], mapf)
        return x
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = recmap(x[i], mapf)
        return x
    elif isinstance(x, tuple):
        newtup = []
        for i in range(len(x)):
            newtup.append(recmap(x[i], mapf))
        newtup = tuple(newtup)
        return newtup
    elif isinstance(x, set):
        newset = set()
        for k in x:
            newset.add(recmap(k, mapf))
        return newset
    else:
        return mapf(x)


percentagebarmap = "\u2581 \u2582 \u2583 \u2584 \u2585 \u2586 \u2587 \u2588 \u2589".split()
def percentagebar(x):
    assert(0.0 <= x <= 1.0)
    x = round(x * (len(percentagebarmap)-1))
    return percentagebarmap[x]


def iscallable(x):
    return hasattr(x, "__call__")


def isfunction(x):
    return iscallable(x)


def getnumargs(f):
    return len(inspect.getargspec(f).args)


def getkw(kw, name, default=None, nodefault=False, remove=True):
    """ convenience function for getting certain kwargs out of function """
    if name in kw:
        ret = kw[name]
        if remove:
            del kw[name]
    else:
        if nodefault:
            raise Exception("kwarg {} must be specified (no default)".format(name))
        ret = default
    return ret


def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, str)


def iscollection(x):
    return issequence(x) or isinstance(x, set)


def isnumber(x):
    return isinstance(x, float) or isinstance(x, int)


def isstring(x):
    return isinstance(x, str)


class StringMatrix():       # TODO: use csr_matrix here
    protectedwords = ["<MASK>", "<RARE>", "<START>", "<END>"]

    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None,
                 indicate_start_end=False, indicate_start=False, indicate_end=False,
                 specialtoks=None):
        self._strings = []
        if specialtoks is not None:
            self.protectedwords = self.protectedwords + specialtoks
        self._wordcounts_original = dict(zip(self.protectedwords, [0] * len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._dictionary_external = False
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_e, self._indic_s = False, False
        if indicate_start_end:
            self._indic_s, self._indic_e = True, True
        if indicate_start:
            self._indic_s = indicate_start
        if indicate_end:
            self._indic_e = indicate_end
        self._rarewords = set()
        self.tokenize = tokenize
        self._cache_p = None
        self.unseen_mode = False
        self._no_rare_sorted = False

    def clone(self):
        n = StringMatrix()
        n.tokenize = self.tokenize
        if self._matrix is not None:
            n._matrix = self._matrix.copy()
            n._dictionary = self._dictionary.copy()
            n._rd = self._rd.copy()

        n._strings = self._strings
        return n

    def clone(self):
        n = StringMatrix()
        n.tokenize = self.tokenize
        if self._matrix is not None:
            n._matrix = self._matrix.copy()
            n._dictionary = self._dictionary.copy()
            n._rd = self._rd.copy()

        n._strings = self._strings
        return n

    def __len__(self):
        if self._matrix is None:
            return len(self._strings)
        else:
            return self.matrix.shape[0]

    def cached(self, p):
        self._cache_p = p
        if os.path.isfile(p):
            pickle.load()

    def __getitem__(self, item, *args):
        if self._matrix is None:
            return self._strings[item]
        else:
            ret = self.matrix[item]
            if len(args) == 1:
                ret = ret[args[0]]
            ret = self.pp(ret)
            return ret

    @property
    def numwords(self):
        return len(self._dictionary)

    @property
    def numrare(self):
        return len(self._rarewords)

    @property
    def matrix(self):
        if self._matrix is None:
            raise Exception("finalize first")
        return self._matrix

    @property
    def D(self):
        return self._dictionary

    def set_dictionary(self, d):
        """ dictionary set in this way is not allowed to grow,
        tokens missing from provided dictionary will be replaced with <RARE>
        provided dictionary must contain <RARE> if missing tokens are to be supported"""
        print("setting dictionary")
        self._dictionary_external = True
        self._dictionary = {}
        self._dictionary.update(d)
        self._next_available_id = max(self._dictionary.values()) + 1
        self._wordcounts_original = dict(zip(list(self._dictionary.keys()), [0]*len(self._dictionary)))
        self._rd = {v: k for k, v in self._dictionary.items()}

    @property
    def RD(self):
        return self._rd

    def d(self, x):
        return self._dictionary[x]

    def rd(self, x):
        return self._rd[x]

    def pp(self, matorvec):
        def pp_vec(vec):
            return " ".join([self.rd(x) if x in self._rd else "<UNK>" for x in vec if x != self.d("<MASK>")])
        ret = []
        if matorvec.ndim == 2:
            for vec in matorvec:
                ret.append(pp_vec(vec))
        else:
            return pp_vec(matorvec)
        return ret

    def add(self, x):
        tokens = self.tokenize(x)
        tokens = tokens[:self._max_allowable_length]
        if self._indic_s is not False and self._indic_s is not None:
            indic_s_sym = "<START>" if not isstring(self._indic_s) else self._indic_s
            tokens = [indic_s_sym] + tokens
        if self._indic_e is not False and self._indic_e is not None:
            indic_e_sym = "<END>" if not isstring(self._indic_e) else self._indic_e
            tokens = tokens + [indic_e_sym]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                if not self._dictionary_external and not self.unseen_mode:
                    self._dictionary[token] = self._next_available_id
                    self._next_available_id += 1
                    self._wordcounts_original[token] = 0
                else:
                    assert("<RARE>" in self._dictionary)
                    token = "<RARE>"    # replace tokens missing from external D with <RARE>
            self._wordcounts_original[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)
        return len(self._strings)-1

    def finalize(self):
        print("finalizing")
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int64")
        for i, string in tqdm(enumerate(self._strings)):
            ret[i, :len(string)] = string
        # print("done")
        self._matrix = ret
        self._do_rare_sorted()
        self._rd = {v: k for k, v in self._dictionary.items()}
        self._strings = None

    def _do_rare_sorted(self):
        """ if dictionary is not external, sorts dictionary by counts and applies rare frequency and dictionary is changed """
        if not self._dictionary_external and not self._no_rare_sorted:
            sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                             ([self.d(x) for x, y
                              in sorted(list(self._wordcounts_original.items()), key=lambda x_y: x_y[1], reverse=True)
                              if y >= self._rarefreq and x not in self.protectedwords][:self._topnwords])
            transdic = zip(sortedwordidxs, range(len(sortedwordidxs)))
            transdic = dict(transdic)
            self._rarewords = {x for x in self._dictionary.keys() if self.d(x) not in transdic}
            rarewords = {self.d(x) for x in self._rarewords}
            self._numrare = len(rarewords)
            transdic.update(dict(zip(rarewords, [self.d("<RARE>")]*len(rarewords))))
            # translate matrix
            self._matrix = np.vectorize(lambda x: transdic[x])(self._matrix)
            # change dictionary
            self._dictionary = {k: transdic[v] for k, v in self._dictionary.items() if self.d(k) in sortedwordidxs}

    def save(self, p):
        pickle.dump(self, open(p, "w"))

    @staticmethod
    def load(p):
        if os.path.isfile(p):
            return pickle.load(open(p))
        else:
            return None


def tokenize(s, preserve_patterns=None, extrasubs=True):
    if not isinstance(s, str):
        s = s.decode("utf-8")
    s = unidecode.unidecode(s)
    repldic = None
    if preserve_patterns is not None:
        repldic = {}
        def _tokenize_preserve_repl(x):
            id = max(list(repldic.keys()) + [-1]) + 1
            repl = "replreplrepl{}".format(id)
            assert(repl not in s)
            assert(id not in repldic)
            repldic[id] = x.group(0)
            return repl
        for preserve_pattern in preserve_patterns:
            s = re.sub(preserve_pattern, _tokenize_preserve_repl, s)
    if extrasubs:
        s = re.sub("[-_\{\}/]", " ", s)
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    if repldic is not None:
        repldic = {"replreplrepl{}".format(k): v for k, v in repldic.items()}
        tokens = [repldic[token] if token in repldic else token for token in tokens]
    s = re.sub("`", "'", s)
    return tokens


class ticktock(object):
    """ timer-printer thingy """
    def __init__(self, prefix="-", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()

    def tick(self, state=None):
        if self.verbose and state is not None:
            print("%s: %s" % (self.prefix, state))
        self._tick()

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of, action="", live=False):
        if self.verbose:
            self.perc = int(round(100. * x / of))
            if self.perc != self.prevperc:
                if action != "":
                    action = " " + action + " -"
                topr = "%s:%s %d" % (self.prefix, action, self.perc) + "%"
                if live:
                    self._live(topr)
                else:
                    print(topr)
                self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        duration = self._tock()
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s in %s" % (prefix, action, self._getdurationstr(duration)))
        return self

    def msg(self, action=None, prefix=None):
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s" % (prefix, action))
        return self

    def _getdurationstr(self, duration):
        if duration >= 60:
            duration = int(round(duration))
            seconds = duration % 60
            minutes = (duration // 60) % 60
            hours = (duration // 3600) % 24
            days = duration // (3600*24)
            acc = ""
            if seconds > 0:
                acc = ("%d second" % seconds) + ("s" if seconds > 1 else "")
            if minutes > 0:
                acc = ("%d minute" % minutes) + ("s" if minutes > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if hours > 0:
                acc = ("%d hour" % hours) + ("s" if hours > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if days > 0:
                acc = ("%d day" % days) + ("s" if days > 1 else "") + (", " + acc if len(acc) > 0 else "")
            return acc
        else:
            return ("%.3f second" % duration) + ("s" if duration > 1 else "")

    def _live(self, x, right=None):
        if right:
            try:
                #ttyw = int(os.popen("stty size", "r").read().split()[1])
                raise Exception("qsdf")
            except Exception:
                ttyw = None
            if ttyw is not None:
                sys.stdout.write(x)
                sys.stdout.write(right.rjust(ttyw - len(x) - 2) + "\r")
            else:
                sys.stdout.write(x + "\t" + right + "\r")
        else:
            sys.stdout.write(x + "\r")
        sys.stdout.flush()

    def live(self, x):
        if self.verbose:
            self._live(self.prefix + ": " + x, "T: %s" % self._getdurationstr(self._tock()))

    def stoplive(self):
        if self.verbose:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    """ use this to enable command-line access to kwargs of function (useful for main run methods) """
    def handler(sig, frame):
        # find the frame right under the argprun
        print("custom handler called")
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            __toexposelocals = previous_frame.f_locals     # f-level frame locals
            class L(object):
                pass
            l = L()
            for k, v in __toexposelocals.items():
                setattr(l, k, v)
            stopprompt = False
            while not stopprompt:
                whattodo = input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

        try:
            with open(os.devnull, 'w') as f:
                oldstdout = sys.stdout
                sys.stdout = f
                from pygame import mixer
                sys.stdout = oldstdout
            mixer.init()
            mixer.music.load(os.path.join(os.path.dirname(__file__), "../resources/jubilation.mp3"))
            mixer.music.play()
        except Exception as e:
            pass
    except KeyboardInterrupt as e:
        print("Interrupted by Keyboard")
    except Exception as e:
        traceback.print_exc()
        try:
            with open(os.devnull, 'w') as f:
                oldstdout = sys.stdout
                sys.stdout = f
                from pygame import mixer
                sys.stdout = oldstdout
            mixer.init()
            mixer.music.load(os.path.join(os.path.dirname(__file__), "../resources/job-done.mp3"))
            mixer.music.play()
        except Exception as e:
            pass
# endregion