import torch
import operator
import numpy as np

DUMMY_WORD = 458

# Trim a tensor by the length of the max thing


trim = lambda x : x[:, :(x.shape[1] - torch.min(torch.sum(x.eq(0).long(), dim=1))).item()]
def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def no_one_left_behind(t):
    """
        In case a tensor is empty at any pos, append a random key there.
        The key used is 458 which is 'nothing' in glove vocab
    :param t: 2d torch tensor
    :return: 2d torch tensor
    """
    superimposed = torch.zeros_like(t)
    superimposed[:,0] = (torch.sum(t, dim=1).eq(0)).int().view(1, -1)*DUMMY_WORD
    return superimposed + t


def compute_mask(t, padding_idx=0):
    """
    compute mask on given tensor t
    :param t:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(t, padding_idx).float()
    return mask


def is_eq_twopaths(gold_path, pred_path):
    if len(gold_path) != len(pred_path):
        return False
    # if operator.eq(gold_path, pred_path):
    if np.array_equal(gold_path, pred_path):
        return True
    new_gold_path = []
    new_pred_path = []
    for gold_elem in gold_path:
        new_gold_path.append(gold_elem.replace('http://dbpedia.org/ontology/','').replace('http://dbpedia.org/property/',''))
    for pred_elem in pred_path:
        new_pred_path.append(pred_elem.replace('http://dbpedia.org/ontology/','').replace('http://dbpedia.org/property/',''))
    # if operator.eq(new_gold_path, new_pred_path):
    if np.array_equal(new_gold_path, new_pred_path):
        return True
    return False
