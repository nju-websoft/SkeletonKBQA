import torch
import qelos as q
from torch import nn
import numpy as np
import re
import math
from nltk.translate.bleu_score import sentence_bleu
import warnings

EPS = 1e-6


__all__ = ["PenaltyGetter", "Accuracy", "SeqAccuracy", "SeqElemAccuracy", "MacroBLEU", "nan2zero", "inf2zero",
           "SmoothedCELoss", "CELoss", "DistillLoss", "LinearLoss", "DiffSmoothedCELoss",
           "SelectedLinearLoss"]


class PenaltyGetter(torch.nn.Module):
    def __init__(self, model:torch.nn.Module, name=None, factor=1., reduction="mean"):
        super(PenaltyGetter, self).__init__()
        self.model = model
        self.name = name
        self.factor = factor
        self.reduction = reduction

    def forward(self, *_, **__):
        factor = q.v(self.factor)
        penalty_value = 0
        for submodule in self.model.modules():
            if hasattr(submodule, self.name):
                pval = getattr(submodule, self.name)()
                if self.reduction in ["elementwise_mean", "mean"]:
                    pval = pval.mean()
                elif self.reduction == "sum":
                    pval = pval.sum()
                else:
                    pass
                penalty_value += pval
        return penalty_value * factor


class LinearLoss(torch.nn.Module):
    """ LinearLoss or NoLoss. Use this if model returns loss """
    def __init__(self, reduction="mean"):
        super(LinearLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, gold, **kw):
        """ x is minimized, gold is ignored (and should be None) """
        if self.reduction in ["elementwise_mean", "mean"]:
            ret = x.mean()
        elif self.reduction == "sum":
            ret = x.sum()
        else:
            ret = x
        return ret


class SelectedLinearLoss(torch.nn.Module):
    """ Same as LinearLoss, but with selection from tuple of outputs from model (that specifies losses)
        To be used to output multiple losses from the model/ select one model output as training loss
    """
    def __init__(self, which, reduction="mean", **kw):
        super(SelectedLinearLoss, self).__init__(**kw)
        self.which = which
        self.reduction = reduction

    def forward(self, model_outs, gold, **kw):
        if q.issequence(model_outs):
            x = model_outs[self.which]
        else:
            assert(self.which == 0)
            x = model_outs

        if self.reduction in ["elementwise_mean", "mean"]:
            ret = x.mean()
        elif self.reduction == "sum":
            ret = x.sum()
        else:
            ret = x
        return ret


def logsumexp(x, axis=-1):
    xmax, _ = torch.max(x, axis, keepdim=True)
    _x = x - xmax
    _x = torch.exp(_x)
    _x = torch.sum(_x, axis, keepdim=True)
    lse = xmax + torch.log(_x)
    return lse.squeeze(-1)


def nan2zero(x):
    nanmask = torch.isnan(x)
    if torch.any(nanmask).item() == 1:
        _x_cpy = torch.zeros_like(x)
        _xv = x.masked_select(~nanmask)
        _x_cpy.masked_scatter_(~nanmask, _xv)
        return _x_cpy
    return x


def inf2zero(x):
    infmask = ((x == -np.inf) | (x == np.inf))
    if torch.any(infmask).item() == 1:
        _x_cpy = torch.zeros_like(x)
        _xv = x.masked_select(~infmask)
        _x_cpy.masked_scatter_(~infmask, _xv)
        return _x_cpy
    return x


class DiscreteLoss(torch.nn.Module):
    """ Loss with ignore_index(es), provides default implementation of _get_ignore_mask """
    def __init__(self, size_average=True, ignore_index=None, **kw):
        super(DiscreteLoss, self).__init__(**kw)
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                self.ignore_indices = [ignore_index]
        else:
            self.ignore_indices = None
        self.size_average = size_average

    @staticmethod
    def get_ignore_mask(gold, ignore_indices):
        if ignore_indices is not None and not q.issequence(ignore_indices):
            ignore_indices = [ignore_indices]
        mask = None     # (batsize,)
        if ignore_indices is not None:
            for ignore in ignore_indices:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask & mask_i
        if mask is None:
            mask = torch.ones_like(gold).byte()
        return mask

    def forward(self, x, gold, **kw):
        y, ignoremask = self._forward(x, gold, **kw)
        y = y.float()

        if ignoremask is not None:
            y = y * ignoremask.float().clamp(0, 1)  # ensure ignoremask is not higher than 1

        total = y.size(0)

        loss = y.sum()
        if self.size_average:
            loss /= total
        return loss


class OldAccuracy(DiscreteLoss):
    def _forward(self, x, gold):
        ignoremask = self.get_ignore_mask(gold, self.ignore_indices)
        maxes, best = torch.max(x, 1)
        same = best == gold
        if ignoremask is not None:
            same = same | ~ ignoremask
        return same.float(), ignoremask


class OldSeqAccuracy(DiscreteLoss):
    """ very basic explicit seqaccuracy implementation.
        does not support batchable sparse mask """
    def _forward(self, x, gold):
        """
        :param x: (batsize, seqlen, vocsize) - probabilities over output symbols for every time step
        :param gold: (batsize, seqlen) - ids of gold output symbols at every time step
        :return: loss value, ignormask
        """
        ignoremask = self.get_ignore_mask(gold, self.ignore_indices)
        _, best = torch.max(x, 2)       # (batsize, seqlen) - most probable symbols at every time step
        same = best == gold
        outignoremask = None
        if ignoremask is not None:
            same = same | ~ ignoremask   # set ignored portions to be same[i,j]=True
            outignoremask = ignoremask.long().sum(1) > 0
        sameseqs = same.long().sum(1)
        sameseqs = sameseqs == int(same.size(1))
        return sameseqs, outignoremask


class SeqElemAccuracy(DiscreteLoss):    # TODO take end of sequence token into account
    def forward(self, x, gold):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        ignoremask = self.get_ignore_mask(gold, self.ignore_indices)
        maxes, argmaxes = torch.max(x, dim=2)
        diff = argmaxes == gold
        if ignoremask is not None:
            diff = diff * ignoremask
            total = torch.sum(ignoremask.long()).item()
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum(diff.float())
        if self.size_average:
            acc = acc / total
        return acc, total


class MacroBLEU(DiscreteLoss):      # TODO take end of sequence token into account
    """ macro-averaged BLEU over sequences """
    def __init__(self, order=4, predcut=None, ignore_index=None, **kw):
        """
        :param order:           n-gram order of BLEU
        :param predcut:         function to cut prediction. Gets the argmax over prediction and ignore_index kwarg.
                                Must fill all elements after end of sequence with provided ignore_index
        """
        super(MacroBLEU, self).__init__(ignore_index=ignore_index, **kw)
        self.order = order
        self.weights = tuple([1. / self.order for _ in range(self.order)])
        self.predcut = predcut
        warnings.filterwarnings("ignore", module="nltk")

    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        ignoremask = self.get_ignore_mask(gold, self.ignore_indices)
        maxes, argmaxes = torch.max(x, dim=2)
        ignore_id = None
        if self.ignore_indices is not None:
            ignore_id = self.ignore_indices[0]
        argmaxes = argmaxes.cpu()
        if self.predcut is not None:
            argmaxes = self.predcut(argmaxes, ignore_index=ignore_id)
        gold = gold.cpu()
        bleus = 0.
        for i in range(gold.size(0)):
            predseq = [str(a) for a in list(argmaxes[i]) if a != ignore_id]
            goldseq = [str(a) for a in list(gold[i]) if a not in self.ignore_indices]
            bleu = sentence_bleu([goldseq], predseq, weights=self.weights)
            bleus += bleu

        total = gold.size(0)
        if self.size_average:
            bleus = bleus / total
        return bleus, total


class CELoss(torch.nn.Module):
    """ Cross entropy loss. """
    def __init__(self, weight=None, reduction="mean", ignore_index=-100, mode="logits", **kw):
        super(CELoss, self).__init__(**kw)
        self.mode = mode
        if mode in ("logprobs", "probs"):
            self.ce = torch.nn.NLLLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        elif mode == "logits":
            self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        else:
            raise q.SumTingWongException("unknown mode {}".format(mode))

    def forward(self, probs, gold):     # (batsize, ..., vocsize), (batsize, ...)
        logprobs = probs if self.mode != "probs" else torch.log(probs)
        if logprobs.dim() > 2:
            permaxis = [0, logprobs.dim()-1] + list(range(1, logprobs.dim()-1))
            logprobs = logprobs.permute(*permaxis)    # (batsize, vocsize, ...)
        ret = self.ce(logprobs, gold)
        return ret


class SmoothedCELoss(torch.nn.Module):
    """ CrossEntropyLoss with label smoothing. """
    def __init__(self, reduction="mean", ignore_index=-100, smoothing=0., mode="logits", weight=None, **kw):
        super(SmoothedCELoss, self).__init__(**kw)
        self.reduction, self.ignore_indices, self.smoothing = reduction, ignore_index, smoothing
        self.mode = mode        # "logits", "probs", "logprobs"
        self.kl = torch.nn.KLDivLoss(reduction="none")
        self.sm = torch.nn.LogSoftmax(-1) if self.mode == "logits" else None
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, probs, gold):
        """
        :param probs:   (batsize, ..., vocsize) logits
        :param gold:    (batsize, ..., ) int ids of correct class
        :return:
        """
        _prob_mask_crit = -np.infty if self.mode in "logits logprobs".split() else 0
        lsv = q.v(self.smoothing)   # get value of label smoothing hyperparam
        assert(lsv >= 0 and lsv <= 1)
        prob_mask = (probs > _prob_mask_crit).float()     # (batsize, ..., vocsize) where probs are > 0, reverse engineering a -infty mask applied outside
        prob_mask_weights = lsv / prob_mask.sum(-1, keepdim=True)
        _gold = torch.ones_like(probs) * prob_mask_weights * prob_mask
        _gold.scatter_(-1, gold.unsqueeze(-1), (1 - lsv) + prob_mask_weights)   # (batsize, ..., vocsize) probs
        assert((_gold.sum(-1) - torch.ones_like(gold).float()).norm().cpu().item() < 1e-5)

        logprobs = self.sm(probs) if self.mode == "logits" else (probs if self.mode == "logprobs" else torch.log(probs))
        kl_divs = self.kl(logprobs, _gold.detach())
        # kl_divs = inf2zero(kl_divs)
        kl_div = kl_divs.sum(-1)        # (batsize, ...) kl div per element

        if self.weight is not None:
            kl_div = kl_div * self.weight[gold]

        mask = DiscreteLoss.get_ignore_mask(gold, self.ignore_indices).float()
        kl_div = kl_div * mask
        ret = kl_div.sum()
        if self.reduction in ["elementwise_mean", "mean"]:
            total = mask.sum()
            ret = ret / total
        elif self.reduction == "none":
            ret = kl_div
        return ret


class DiffSmoothedCELoss(torch.nn.Module):      # DON'T USE THIS: works not better than normal label smoothing
    def __init__(self, reduction="mean", ignore_index=-100, alpha=0.7, beta=0.2, mode="logits", **kw):
        """
        :param reduction:
        :param ignore_index:
        :param alpha:   weight of predicted distribution when mixing target for correctly predicted tokens
        :param beta:    weight of uniform distribution when mixing target for incorrectly predicted tokens (normal label smoothing)
        :param mode:
        :param kw:
        """
        super(DiffSmoothedCELoss, self).__init__(**kw)
        self.reduction, self.ignore_indices = reduction, ignore_index
        self.mode = mode
        self.alpha, self.beta = alpha, beta
        self.kl = torch.nn.KLDivLoss(reduction="none")
        self.logsm = torch.nn.LogSoftmax(-1) if self.mode == "logits" else None
        self.sm = torch.nn.Softmax(-1) if self.mode == "logits" else None

    def forward(self, probs, gold):
        """
        :param probs:   (batsize, ..., vocsize) logits
        :param gold:    (batsize, ..., ) int ids of correct class
        :return:
        """
        _prob_mask_crit = -np.infty if self.mode in "logits logprobs".split() else 0
        # construction target distribution
        lsv = q.v(self.beta)   # get value of label smoothing hyperparam
        assert(lsv >= 0 and lsv <= 1)
        # mix in some masked uniform into onehot normal target
        prob_mask = (probs > _prob_mask_crit).float()     # (batsize, ..., vocsize) where probs are > 0, reverse engineering a -infty mask applied outside
        prob_mask_weights = lsv / prob_mask.sum(-1, keepdim=True)
        _gold = torch.ones_like(probs) * prob_mask_weights * prob_mask
        _gold.scatter_(-1, gold.unsqueeze(-1), (1 - lsv) + prob_mask_weights)   # (batsize, ..., vocsize) probs
        # assert((_gold.sum(-1) - torch.ones_like(gold).float()).norm().item() < 1e-5)

        # mix predictive with target
        alpha = q.v(self.alpha)
        _probs = self.sm(probs) if self.mode == "logits" else (torch.exp(probs) if self.mode == "logprobs" else probs)
        _gold2 = torch.zeros_like(probs)
        _gold2.scatter_(-1, gold.unsqueeze(-1), 1)   # (batsize, ..., vocsize) probs
        _gold2 = _probs.detach() * alpha +  _gold2 * (1 - alpha)
        # assert((_gold2.sum(-1) - torch.ones_like(gold).float()).norm().item() < 1e-5)

        # mix _gold and _gold2
        best = torch.argmax(probs, -1)
        mixmask = (best == gold).float().unsqueeze(-1)
        _gold = mixmask * _gold2 + (1 - mixmask) * _gold

        logprobs = self.logsm(probs) if self.mode == "logits" else (probs if self.mode == "logprobs" else torch.log(probs))
        kl_divs = self.kl(logprobs, _gold.detach())
        # kl_divs = inf2zero(kl_divs)
        kl_div = kl_divs.sum(-1)        # (batsize, ...) kl div per element

        mask = DiscreteLoss.get_ignore_mask(gold, self.ignore_indices).float()
        kl_div = kl_div * mask
        ret = kl_div.sum()
        if self.reduction in ["elementwise_mean", "mean"]:
            total = mask.sum()
            ret = ret / total
        elif self.reduction == "none":
            ret = kl_div
        return ret


class DistillLoss(torch.nn.Module):
    """ Distillation (KD) loss for sequences of categorical distributions """
    def __init__(self, weight=None, reduction="mean", ignore_index=-100,
                 temperature=1., mixture=0.5, **kw):
        """
        :param ignore_index:    gold ids whose time steps will be ignored
        :param temperature:     softmax temperature (!: will not be applied if soft_gold_mode is not "logits")
        :param mixture:         mixing portion of soft and hard gold (1 => only soft kl, 0 => only hard ce)
        :param kw:
        """
        super(DistillLoss, self).__init__(**kw)
        self.ignore_indices = ignore_index
        self.reduction = reduction
        self.temperature = temperature
        self.mixture = mixture
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        self.hardCE = CELoss(reduction="none", ignore_index=ignore_index, weight=weight, mode="logits")
        self.kl = torch.nn.KLDivLoss(reduction="none")

    def forward(self, probs, gold):
        """
        :param probs:       (batsize, ..., numsym) logit predictions
        :param gold:        tuple of (batsize, ..., numsym) soft gold logits and (batsize, ...) ints for hard gold
        """
        softgold, hardgold = gold
        t = q.v(self.temperature)
        mix = q.v(self.mixture)

        # hard gold - normal CE, !!: probs are logits
        hard_ce = 0
        if mix < 1:
            hard_ce = self.hardCE(probs, hardgold)

        # soft gold
        kl_div = 0
        if mix > 0:
            _log_probs = self.logsm(probs / t)
            if torch.any(probs == -np.infty).item() == 1:
                softgold = softgold + torch.log((probs != -np.infty).float())
            _softgold = self.sm(softgold / t)
            kl_divs = self.kl(_log_probs, _softgold)
            # kl_divs = inf2zero(kl_divs)
            kl_div = kl_divs.sum(-1)

        # mix
        loss = mix * kl_div + (1 - mix) * hard_ce        # (batsize, seqlen)

        mask = DiscreteLoss.get_ignore_mask(hardgold, self.ignore_indices).float()
        loss = loss * mask
        ret = loss.sum()
        if self.reduction in ["elementwise_mean", "mean"]:
            total = mask.sum()
            ret = ret / total
        elif self.reduction == "none":
            ret = loss
        return ret


class Accuracy(torch.nn.Module):
    """ Accuracy in the last dimension. For sequences, use SeqAccuracy """

    def __init__(self, reduction="mean", ignore_index=-100, **kw):
        super(Accuracy, self).__init__(**kw)
        self.reduction = reduction
        self.ignore_index = ignore_index

    @staticmethod
    def get_ignore_mask(gold, ignore_index):
        if ignore_index < 0:
            return None
        else:
            mask = gold != ignore_index
            return mask

    def forward(self, x, gold):
        # assert(gold.dim() == 1)     # use a different accuracy if not 1D target
        ignoremask = self.get_ignore_mask(gold, self.ignore_index)
        maxes, best = torch.max(x, -1)
        same = best == gold
        same = same.float()
        if self.reduction in ["elementwise_mean", "mean"]:
            if ignoremask is None:
                total = 1.
                for i in range(same.dim()):
                    total *= same.size(i)
            else:
                total = ignoremask.float().sum()
            ret = same.sum() / total
        elif self.reduction == "none":
            ret = same
        elif self.reduction == "sum":
            ret = same.sum()
        return ret


class SeqAccuracy(torch.nn.Module):
    """ very basic explicit seqaccuracy implementation.
        does not support batchable sparse mask """

    def __init__(self, reduction="mean", ignore_index=-100, **kw):
        super(SeqAccuracy, self).__init__(**kw)
        self.reduction = reduction
        self.ignore_index = ignore_index

    @staticmethod
    def get_ignore_mask(gold, ignore_index):
        if ignore_index < 0:
            return None
        else:
            mask = gold != ignore_index
            return mask

    def forward(self, x, gold):
        """
        :param x: (batsize, seqlen, vocsize) - probabilities over output symbols for every time step
        :param gold: (batsize, seqlen) - ids of gold output symbols at every time step
        :return: loss value, ignormask
        """
        ignoremask = self.get_ignore_mask(gold, self.ignore_index)
        _, best = torch.max(x, 2)  # (batsize, seqlen) - most probable symbols at every time step
        same = best == gold
        outignoremask = None
        if ignoremask is not None:
            same = same | ~ ignoremask  # set ignored portions to be same[i,j]=True
            outignoremask = ignoremask.long().sum(1) > 0
        sameseqs = same.long().sum(1)
        sameseqs = sameseqs == int(same.size(1))
        sameseqs = sameseqs.float()
        if self.reduction in ["elementwise_mean", "mean"]:
            total = x.size(0)
            ret = sameseqs.sum() / total
        elif self.reduction == "sum":
            ret = sameseqs.sum()
        else:
            ret = sameseqs
        return ret