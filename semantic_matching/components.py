import math
import torch
import qelos as q
from copy import deepcopy

from torch.autograd import Variable
from math import floor
import tensor_utils as tu
from torch import nn


class AdaptedBERTEncoderSingle(torch.nn.Module):
    pad_id = 0

    def __init__(self, bert, numout=2, oldvocab=None, specialmaps={0: [0]}, **kw):
        """
        :param bert:
        :param numout:      number of outputs. If -1, output layer is not applied and the encoding is returned.
                                If 0, number of outputs is actually 1 and output is squeezed before returning.
        :param oldvocab:    word vocab used for inpids fed to this model. Will be translated to new (bert) vocab.
        :param kw:
        """
        super(AdaptedBERTEncoderSingle, self).__init__()
        self.bert = bert
        specialmaps = deepcopy(specialmaps)
        self.numout = numout
        if self.numout >= 0:
            numout = numout if numout > 0 else 1
            self.lin = torch.nn.Linear(bert.dim, numout)
            self.dropout = torch.nn.Dropout(p=bert.dropout)
            self.reset_parameters()
        self.oldD = oldvocab
        self.D = self.bert.D
        self.wp_tok = q.bert.WordpieceTokenizer(self.bert.D)
        # create mapper
        self.oldD2D = {}
        for k, v in self.oldD.items():
            if v in specialmaps:
                self.oldD2D[v] = specialmaps[v]
            else:
                key_pieces = self.wp_tok.tokenize(k)
                key_piece_ids = [self.D[key_piece] for key_piece in key_pieces]
                self.oldD2D[v] = key_piece_ids

    def reset_parameters(self):
        if self.numout >= 0:
            torch.nn.init.normal_(self.lin.weight, 0, self.bert.init_range)
            torch.nn.init.zeros_(self.lin.bias)
        #self.bert.reset_parameters()


"""score1"""


class AdaptedBERTEncoderPairSlotPtr(AdaptedBERTEncoderSingle):

    oldvocab_norel_tok = "ft"
    norel_tok = "—"
    def __init__(self, bert, oldvocab=None, numout=0, max_num_relations=2, specialmaps={0: [0]}, **kw):
        # kw.pop("numout")
        if self.oldvocab_norel_tok in oldvocab:
            specialmaps[oldvocab[self.oldvocab_norel_tok]] = [bert.D[self.norel_tok]]
        super(AdaptedBERTEncoderPairSlotPtr, self).__init__(bert, numout=0, oldvocab=oldvocab, specialmaps=specialmaps, **kw)
        self.max_num_relations = max_num_relations
        self.attlin = torch.nn.Linear(bert.dim, max_num_relations)
        self.attsm = torch.nn.Softmax(1)

    def forward(self, q, rel1, rel2, rel3, rel4):
        """
        :param q:   (batsize, seqlen_q)
        :param rel1:   (batsize, seqlen_rel1)
        :param rel2:   (batsize, seqlen_rel2)
        :return:
        """
        # TODO: take into account "ft"
        assert(len(q) == len(rel1))
        # transform inpids
        newinp = torch.zeros_like(q)
        rels = [rel1, rel2, rel3, rel4]
        rels_maxnumrel = [rels[i] for i in range(self.max_num_relations)]
        relidxs = [torch.zeros_like(
            torch.zeros(q.size(0), device=q.device, dtype=torch.long)
            ) for _ in range(self.max_num_relations)]

        typeflip = []
        maxlen = 0
        for i in range(len(q)):    # iter over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(q.size(1)):     # iter over seqlen
                wordpieces = self.oldD2D[q[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
                else:
                    break

            typeflip.append(last_nonpad)
            for m in range(self.max_num_relations):
                newinp[i, last_nonpad] = self.D["[SEP]"]
                k = last_nonpad + 1
                relidxs[m][i] = k - 1
                for j in range(rels_maxnumrel[m].size(1)):
                    if rels_maxnumrel[m][i, j].cpu().item() in self.oldD2D:
                        wordpieces = self.oldD2D[rels_maxnumrel[m][i, j].cpu().item()]
                    else:
                        wordpieces = []
                    for wordpiece in wordpieces:
                        newinp[i, k] = wordpiece
                        k += 1
                        if newinp.size(1) <= k + 1:
                            newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                    if newinp[i, k - 1].cpu().item() != self.pad_id:
                        last_nonpad = k
                    else:
                        break
            newinp[i, last_nonpad] = self.D["[SEP]"]
            maxlen = max(maxlen, last_nonpad + 1)

        newinp = newinp[:, :maxlen]
        # do forward
        typeids = torch.zeros_like(newinp)
        for i, flip in enumerate(typeflip):
            typeids[i, flip:] = 1
        padmask = newinp != self.pad_id
        layerouts, poolout = self.bert(newinp, typeids, padmask)

        _ys = layerouts[-1]

        # compute scores
        scores = self.attlin(_ys)
        qmask = typeids != 1
        scores = scores + torch.log(qmask.float().unsqueeze(-1))
        scores = self.attsm(scores)  # (batsize, seqlen, 2)

        # get summaries
        ys = _ys.unsqueeze(2)
        scores = scores.unsqueeze(3)
        b = ys * scores  # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)  # (batsize, 2, dim)
        summaries = summaries.view(summaries.size(0), -1)  # (batsize, 2*dim)

        # get relation encodings based on indexes stored before
        rels_enc = [_ys.gather(1, relidxs[m].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _ys.size(-1))).squeeze(1) for m in range(self.max_num_relations)]
        relenc = torch.cat(rels_enc, 1)  # #32, 768*2 按照指定维度拼接

        # compute output scores
        dots = torch.einsum("bd,bd->b", [summaries, relenc])

        # normalizin dots by dim: COMMENT THIS OUT IF DON'T WANT NORM DOTS BY DIM
        dots = dots / math.sqrt(summaries.size(1))
        return dots


"""score12"""


class AdptedBERTEncoderPairSlotDepPathPtr(AdaptedBERTEncoderSingle):
    oldvocab_norel_tok = "ft"
    norel_tok = "—"
    def __init__(self, bert, max_num_relations=2, numout=0, oldvocab=None, specialmaps={0: [0]}, **kw):
        if self.oldvocab_norel_tok in oldvocab:
            specialmaps[oldvocab[self.oldvocab_norel_tok]] = [bert.D[self.norel_tok]]
        super(AdptedBERTEncoderPairSlotDepPathPtr, self).__init__(bert, numout=0, oldvocab=oldvocab, specialmaps=specialmaps,**kw)
        self.max_num_relations = max_num_relations
        self.attlin = torch.nn.Linear(bert.dim, self.max_num_relations)
        self.finallin = torch.nn.Linear(2, 1)
        self.attsm = torch.nn.Softmax(1)
        self.dep_attlin = torch.nn.Linear(bert.dim, self.max_num_relations)

    def bert_encode(self, q, dep, rels_maxnumrel):
        '''
        bert-based encoder of question with dependency path and predicate path
        :return:
        '''
        newinp = torch.zeros_like(q)
        # rel1idxs = torch.zeros(q.size(0), device=q.device, dtype=torch.long)
        relidxs = [torch.zeros_like(
            torch.zeros(q.size(0), device=q.device, dtype=torch.long) ) for _ in range(self.max_num_relations)]
        typeflip = []
        endflip = []
        dep_start_flip = []
        dep_end_flip = []
        maxlen = 0
        for i in range(len(q)):  # iterate over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(q.size(1)):
                wordpieces = self.oldD2D[q[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1) #相当于空间不够了增大一倍空间
                if newinp[i, k - 1].cpu().item() != self.pad_id:
                    last_nonpad = k
                else:
                    break
            dep_start_flip.append(last_nonpad)
            for z in range(dep.size(1)):
                newinp[i, last_nonpad] = self.D['[SEP]']
                k = last_nonpad + 1
                if dep[i, z].cpu().item() in self.oldD2D:
                    wordpieces = self.oldD2D[dep[i,z].cpu().item()]
                else:
                    wordpieces = []
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k - 1].cpu().item() != self.pad_id:
                    last_nonpad = k
                else:
                    break
            dep_end_flip.append(last_nonpad)
            typeflip.append(last_nonpad)
            for m in range(self.max_num_relations):
                newinp[i, last_nonpad] = self.D["[SEP]"]
                k = last_nonpad + 1
                relidxs[m][i] = k - 1
                for j in range(rels_maxnumrel[m].size(1)):
                    # if rels_maxnumrel[m][i, j].cpu().item() in self.oldD2D:
                    # print('self.oldD[rels_maxnumrel[m][i, j].cpu().item()]',self.oldD[rels_maxnumrel[m][i, j].cpu().item()])
                    if rels_maxnumrel[m][i, j].cpu().item() in self.oldD2D:
                        wordpieces = self.oldD2D[rels_maxnumrel[m][i, j].cpu().item()]
                    else:
                        wordpieces = []
                    for wordpiece in wordpieces:
                        newinp[i, k] = wordpiece
                        k += 1
                        if newinp.size(1) <= k + 1:
                            newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                    if newinp[i, k - 1].cpu().item() != self.pad_id:
                        last_nonpad = k
                    else:
                        break
            newinp[i, last_nonpad] = self.D["[SEP]"]
            endflip.append(last_nonpad)
            maxlen = max(maxlen, last_nonpad + 1)
        newinp = newinp[:, :maxlen]
        typeids = torch.zeros_like(newinp) # token type embedding type 为 rel
        for i, flip in enumerate(typeflip):
            typeids[i, flip:] = 1    #typeids predicate path设为1
        padmask = newinp != self.pad_id # 不是填充符的位置是可以用来mask的
        layerouts, poolout = self.bert(newinp, typeids, padmask)
        _ys = layerouts[-1]
        return newinp, _ys, relidxs, typeflip, endflip, dep_start_flip, dep_end_flip, typeids

    def score1(self, _ys, relenc, typeids):
        '''
        newinp, dep_start_flip,
        bert_based slot matching scoring
        #typeids 前半部分设为0, predicate path设为1
        :return:
        '''
        q_words_scores = self.attlin(_ys)
        qmask = typeids != 1  #qmask 前半部分设为1, 后半部分typeids dependency path and predicate path设为0
        q_words_scores = q_words_scores + torch.log(qmask.float().unsqueeze(-1))  #q_words_scores的前半部分是原来的, 后半部分设为-inf
        q_words_scores = self.attsm(q_words_scores)  # (batsize, seqlen, 2)   q_word_scores的前半部分softmax，后半部分设为0

        # get summaries
        ys = _ys.unsqueeze(2)
        q_words_scores = q_words_scores.unsqueeze(3)
        b = ys * q_words_scores  # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)  # (batsize, 2, dim)
        summaries = summaries.view(summaries.size(0), -1)  # (batsize, 2*dim)  #新的问句表示, 展开为一个向量(num_relation*768)

        # compute output scores
        dots = torch.einsum("bd,bd->b", [summaries, relenc])
        # normalizin dots by dim: COMMENT THIS OUT IF DON'T WANT NORM DOTS BY DIM
        dots = dots / math.sqrt(summaries.size(1))
        return dots

    def score2(self, _ys, relenc, typeids):
        '''
        bert_based dependency path scoring
        :return:
        '''
        # step 0 mask typeids  #predicate path设为1, 其余部分设为0, 16*24
        mask = torch.zeros_like(_ys)
        size_i, size_j = typeids.shape
        mask_size = _ys.shape[-1]
        for i in range(size_i):
            for j in range(size_j):
                if typeids[i, j] > 0:
                    mask[i, j, :] = torch.ones(mask_size, dtype=torch.float)
                else:
                    mask[i, j, :] = torch.zeros(mask_size, dtype=torch.float)
        ys = _ys.masked_fill(mask.byte(), float('0'))

        # step 1 attention weights
        attention = torch.bmm(ys, relenc.transpose(2, 1))  # 16,10,2
        typeids = typeids.unsqueeze(2)
        typeids = typeids.repeat(1, 1, attention.shape[-1])
        attention = attention.masked_fill(typeids.byte(), -float('1e10'))
        attention = torch.softmax(attention, dim=1)  # 16,10,2

        # step 2 question representation
        New_ys = torch.bmm(ys.transpose(1, 2), attention)  # 16,768,2
        New_ys = New_ys.transpose(1, 2)  # 16,2,768

        # step 3 dot and sum
        scores = (New_ys * relenc).sum(-1)  # 16×2
        scores = torch.sum(scores, dim=1)  # 16,1
        scores = scores / math.sqrt(New_ys.shape[-1]*New_ys.shape[-2])
        return scores

    def merge_two_scores(self, q, dep, rels_maxnumrel):
        newinp, _ys, relidxs, typeflip, endflip, dep_start_flip, dep_end_flip, typeids = self.bert_encode(q=q, dep=dep, rels_maxnumrel=rels_maxnumrel)
        rels_enc = [
            _ys.gather(1,
                       relidxs[m].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _ys.size(-1))
                       ).squeeze(1)
            for m in range(self.max_num_relations)
        ]

        """score 1"""
        slot_relenc = torch.cat(rels_enc, 1)  # #32, 768*2 按照指定维度拼接
        score1_typeids = torch.zeros_like(newinp)  # token type embedding type 为 rel
        for i, flip in enumerate(dep_start_flip):
            score1_typeids[i, flip:] = 1  # typeids dependency path and predicate path设为1
        q_words_summaries_score = self.score1(_ys=_ys, relenc=slot_relenc, typeids=score1_typeids)

        """score 2"""
        dep_relenc = torch.cat([x.unsqueeze(1) for x in rels_enc], dim=1)
        score2_typeids = torch.ones_like(newinp)
        temp_index = 0
        for dep_start, dep_end in zip(dep_start_flip, dep_end_flip):
            score2_typeids[temp_index, dep_start:dep_end+1] = 0  # typeids word部分 and predicate path设为1
            temp_index += 1
        deps_summaries_score = self.score2(_ys=_ys, relenc=dep_relenc, typeids=score2_typeids)

        """combine scores"""
        sim_cat = torch.cat([q_words_summaries_score.unsqueeze(1), deps_summaries_score.unsqueeze(1)], 1)
        sim = self.finallin(sim_cat)
        sim = sim.squeeze(1)
        return sim

    def forward(self, q, dep, dep_mask, rel1, rel2, rel3, rel4):
        assert len(dep) == len(rel1) and len(q) == len(rel1)  # batchsize is same
        rels = [rel1, rel2, rel3, rel4]
        rels_maxnumrel = [rels[i] for i in range(self.max_num_relations)]

        # step 1 cat # temp_tensor = (batch size, L)
        temp_tensor = torch.zeros((dep.shape[0], dep.shape[1]), device=dep.device)
        for dep_index in range(dep.size(1)):  # (batch size, 1)
            current_dep = dep[:, dep_index, :]
            combine_score = self.merge_two_scores(q=q, dep=current_dep, rels_maxnumrel=rels_maxnumrel)
            combine_score = combine_score.unsqueeze(1)
            temp_tensor[:, dep_index] = combine_score[:, 0]

        # step 2 mask (batch size, L)
        temp_tensor = temp_tensor * dep_mask

        # step 3 max (batch size, 1)
        # scores = torch.max(temp_tensor, dim=1)[0]
        scores = torch.sum(temp_tensor, dim=1)

        # scores = scores.unsqueeze(1)
        return scores


"""score2"""


class AdptedBERTEncoderDepPath(AdaptedBERTEncoderSingle):

    oldvocab_norel_tok = "ft"
    norel_tok = "—"
    def __init__(self, bert, max_num_relations=2, numout=0, oldvocab=None, specialmaps={0: [0]}, **kw):
        if self.oldvocab_norel_tok in oldvocab:
            specialmaps[oldvocab[self.oldvocab_norel_tok]] = [bert.D[self.norel_tok]]
        super(AdptedBERTEncoderDepPath, self).__init__(bert, numout=0, oldvocab=oldvocab, specialmaps=specialmaps,**kw)
        self.max_num_relations = max_num_relations
        self.attsm = torch.nn.Softmax(1)
        self.dep_attlin = torch.nn.Linear(bert.dim, self.max_num_relations)
        self.finallin = torch.nn.Linear(2, 1)

    def bert_encode(self, dep, rels_maxnumrel):
        '''
        bert-based encoder of question with dependency path and predicate path
        :return:
        '''
        newinp = torch.zeros_like(dep)
        # rel1idxs = torch.zeros(q.size(0), device=q.device, dtype=torch.long)
        relidxs = [torch.zeros_like(
            torch.zeros(dep.size(0), device=dep.device, dtype=torch.long)
        ) for _ in range(self.max_num_relations)]

        typeflip = []
        maxlen = 0
        for i in range(len(dep)):  # iterate over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(dep.size(1)):
                wordpieces = self.oldD2D[dep[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1) #相当于空间不够了增大一倍空间
                if newinp[i, k - 1].cpu().item() != self.pad_id:
                    last_nonpad = k
                else:
                    break
            typeflip.append(last_nonpad)
            for m in range(self.max_num_relations):
                newinp[i, last_nonpad] = self.D["[SEP]"]
                k = last_nonpad + 1
                relidxs[m][i] = k - 1
                for j in range(rels_maxnumrel[m].size(1)):
                    if rels_maxnumrel[m][i, j].cpu().item() in self.oldD2D:
                        wordpieces = self.oldD2D[rels_maxnumrel[m][i, j].cpu().item()]
                    else:
                        wordpieces = []
                    for wordpiece in wordpieces:
                        newinp[i, k] = wordpiece
                        k += 1
                        if newinp.size(1) <= k + 1:
                            newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                    if newinp[i, k - 1].cpu().item() != self.pad_id:
                        last_nonpad = k
                    else:
                        break
            newinp[i, last_nonpad] = self.D["[SEP]"]
            maxlen = max(maxlen, last_nonpad + 1)
        newinp = newinp[:, :maxlen]
        typeids = torch.zeros_like(newinp) # token type embedding type 为 rel
        for i, flip in enumerate(typeflip):
            typeids[i, flip:] = 1    #typeids predicate path设为1
        padmask = newinp != self.pad_id # 不是填充符的位置是可以用来mask的
        layerouts, poolout = self.bert(newinp, typeids, padmask)
        _ys = layerouts[-1]
        return newinp, _ys, relidxs, typeflip, typeids, poolout

    def predicate_vector(self, _ys, relidxs):
        # relation path SEP representation
        rels_enc = [
            _ys.gather(1,
                       relidxs[m].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _ys.size(-1))).squeeze(1)
            for m in range(self.max_num_relations)
        ]
        # relenc = torch.cat(rels_enc, 1)  # #32, 768*2 按照指定维度拼接
        d = [x.unsqueeze(1) for x in rels_enc]
        rels_enc = torch.cat(d, dim=1)
        return rels_enc

    def score2(self, _ys, relenc, typeids):
        # step 0 mask typeids  #predicate path设为1, 其余部分设为0, 16*24
        mask = torch.zeros_like(_ys)
        size_i,size_j = typeids.shape
        mask_size = _ys.shape[-1]
        for i in range(size_i):
            for j in range(size_j):
                if typeids[i, j] > 0:
                    mask[i, j, :] = torch.ones(mask_size, dtype=torch.float)
                else:
                    mask[i, j, :] = torch.zeros(mask_size, dtype=torch.float)
        ys = _ys.masked_fill(mask.byte(), float('0'))

        # step 1 attention weights
        attention = torch.bmm(ys, relenc.transpose(2, 1))  # 16,10,2
        typeids = typeids.unsqueeze(2)
        typeids = typeids.repeat(1, 1, attention.shape[-1])
        attention = attention.masked_fill(typeids.byte(),-float('1e10'))
        attention = torch.softmax(attention, dim=1)  # 16,10,2

        # step 2 question representation
        New_ys = torch.bmm(ys.transpose(1, 2), attention)  # 16,768,2
        New_ys = New_ys.transpose(1, 2)  # 16,2,768

        # step 3 dot and sum
        scores = (New_ys * relenc).sum(-1)  # 16×2
        scores = torch.sum(scores, dim=1)  # 16,1
        return scores

    def forward(self, dep, dep_mask, rel1, rel2, rel3, rel4):
        assert (len(dep) == len(rel1))  # batchsize is same
        rels = [rel1, rel2, rel3, rel4]
        rels_maxnumrel = [rels[i] for i in range(self.max_num_relations)]

        # step 1 cat # temp_tensor = (batch size, L)
        temp_tensor = torch.zeros((dep.shape[0], dep.shape[1]), device=dep.device)
        for dep_index in range(dep.size(1)): #(batch size, 1)
            current_dep = dep[:,dep_index,:]
            _, _ys, relidxs, typeflip, typeids, poolout = self.bert_encode(dep=current_dep, rels_maxnumrel=rels_maxnumrel)
            relenc = self.predicate_vector(_ys=_ys, relidxs=relidxs)
            dep_score = self.score2(_ys=_ys, relenc=relenc, typeids=typeids)
            dep_score = dep_score.unsqueeze(1)
            temp_tensor[:, dep_index] = dep_score[:,0]

        # step 2 mask (batch size, L)
        temp_tensor = temp_tensor * dep_mask

        # step 3 max (batch size, 1)
        # scores = torch.max(temp_tensor, dim=1)[0]
        scores = torch.sum(temp_tensor, dim=1)

        # scores = scores.unsqueeze(1)
        return scores


class NotSuchABetterEncoder(torch.nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 embedding_layer=None, debug=False, residual=False):
        '''
            :param max_length: Max length of the sequence.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
            :param enable_layer_norm: Bool/ layer normalization.
            :param mode: LSTM/GRU.
            :param residual: Bool/ return embedded state of the input.

        TODO: Implement multilayered shit someday.
        '''
        super(NotSuchABetterEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug
        self.mode = mode
        self.residual = residual

        self.embedding_layer=embedding_layer
        assert self.mode in ['LSTM', 'GRU']

        # if vectors is not None:
        #     self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
        #     self.embedding_layer.weight.requires_grad = False
        # else:
        #     # Embedding layer
        #     self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim)

        # Mode
        if self.mode == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=1,
                                     bidirectional=self.bidirectional)
        elif self.mode == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    bidirectional=self.bidirectional)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, h):
        """

        :param x: input (batch, seq)
        :param h: hiddenstate (depends on mode. see init hidden)
        :param device: torch device
        :return: depends on booleans passed @ init.
        """

        if self.debug:
            print ("\tx:\t", x.shape)
            if self.mode is "LSTM":
                print ("\th[0]:\t", h[0].shape)
            else:
                print ("\th:\t", h.shape)

        mask = tu.compute_mask(x)

        x = self.embedding_layer(x).transpose(0, 1)

        if self.debug: print ("x_emb:\t\t", x.shape)

        if self.enable_layer_norm:
            seq_len, batch, input_size = x.shape
            x = x.view(-1, input_size)
            x = self.layer_norm(x)
            x = x.view(seq_len, batch, input_size)

        if self.debug: print("x_emb bn:\t", x.shape)

        # get sorted v
        # print('mask.size()',mask.size())
        lengths = mask.eq(1).long().sum(1)
        # print('lengths.size()', lengths.size())
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # print('lengths_sort.size()', lengths_sort.size())
        # print('idx_sort.size()', idx_sort.size())
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # print('idx_unsort.size()', idx_unsort.size())
        # print('x.size()', x.size())
        x_sort = x.index_select(1, idx_sort)
        # print('x_sort.size()', x_sort.size())
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) \
            if self.mode is "LSTM" else h.index_select(1, idx_sort)
        # print(x_sort,lengths_sort)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort)
        x_dropout = self.dropout.forward(x_pack.data)
        x_pack_dropout = torch.nn.utils.rnn.PackedSequence(x_dropout, x_pack.batch_sizes)

        if self.debug:
            print("\nidx_sort:", idx_sort.shape)
            print("idx_unsort:", idx_unsort.shape)
            print("x_sort:", x_sort.shape)
            if self.mode is "LSTM":
                print ("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print ("h_sort:\t\t", h_sort.shape)


        o_pack_dropout, h_sort = self.rnn.forward(x_pack_dropout, h_sort)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # Unsort o based ont the unsort index we made
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) \
            if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


        # @TODO: Do we also unsort h? Does h not change based on the sort?

        if self.debug:
            if self.mode is "LSTM":
                print("h_sort\t\t", h_sort[0].shape)
            else:
                print("h_sort\t\t", h_sort.shape)
            print("o_unsort\t\t", o_unsort.shape)
            if self.mode is "LSTM":
                print("h_unsort\t\t", h_unsort[0].shape)
            else:
                print("h_unsort\t\t", h_unsort.shape)
        # print('o_unsort.size()',o_unsort.size())
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        # print('len_idx.size()', len_idx.size())
        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            # print('x',x.size())
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask

    @property
    def layers(self):
        return torch.nn.ModuleList([
            torch.nn.ModuleList([self.embedding_layer, self.rnn, self.dropout]),
        ])


class NotSuchABetterEncoder_v2(torch.nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 embedding_layer=None, debug=False, residual=False):
        '''
            :param max_length: Max length of the sequence.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
            :param enable_layer_norm: Bool/ layer normalization.
            :param mode: LSTM/GRU.
            :param residual: Bool/ return embedded state of the input.

        TODO: Implement multilayered shit someday.
        '''
        super(NotSuchABetterEncoder_v2, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug
        self.mode = mode
        self.residual = residual


        assert self.mode in ['LSTM', 'GRU']

        self.embedding_layer=embedding_layer
        # if vectors is not None:
        #     self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
        #     self.embedding_layer.weight.requires_grad = True
        # else:
        #     # Embedding layer
        #     self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Mode
        if self.mode == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=1,
                                     bidirectional=self.bidirectional)
        elif self.mode == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    bidirectional=self.bidirectional)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, h,mask):
        """

        :param x: input (batch, seq)
        :param h: hiddenstate (depends on mode. see init hidden)
        :param device: torch device
        :return: depends on booleans passed @ init.
        """

        if self.debug:
            print ("\tx:\t", x.shape)
            if self.mode is "LSTM":
                print ("\th[0]:\t", h[0].shape)
            else:
                print ("\th:\t", h.shape)

#         mask = tu.compute_mask(x)

#         x = self.embedding_layer(x).transpose(0, 1)

        if self.debug: print ("x_emb:\t\t", x.shape)

        if self.enable_layer_norm:
            seq_len, batch, input_size = x.shape
            x = x.view(-1, input_size)
            x = self.layer_norm(x)
            x = x.view(seq_len, batch, input_size)

        if self.debug: print("x_emb bn:\t", x.shape)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x_sort = x.index_select(1, idx_sort)
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) \
            if self.mode is "LSTM" else h.index_select(1, idx_sort)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort)
        x_dropout = self.dropout.forward(x_pack.data)
        x_pack_dropout = torch.nn.utils.rnn.PackedSequence(x_dropout, x_pack.batch_sizes)

        if self.debug:
            print("\nidx_sort:", idx_sort.shape)
            print("idx_unsort:", idx_unsort.shape)
            print("x_sort:", x_sort.shape)
            if self.mode is "LSTM":
                print ("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print ("h_sort:\t\t", h_sort.shape)

#         print("YABADABADOOO")
#         print(x_pack_dropout.shape, h_sort[0].shape)
        o_pack_dropout, h_sort = self.rnn.forward(x_pack_dropout, h_sort)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # Unsort o based ont the unsort index we made
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) \
            if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


        # @TODO: Do we also unsort h? Does h not change based on the sort?

        if self.debug:
            if self.mode is "LSTM":
                print("h_sort\t\t", h_sort[0].shape)
            else:
                print("h_sort\t\t", h_sort.shape)
            print("o_unsort\t\t", o_unsort.shape)
            if self.mode is "LSTM":
                print("h_unsort\t\t", h_unsort[0].shape)
            else:
                print("h_unsort\t\t", h_unsort.shape)

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?

        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask


class QelosSlotPtrQuestionEncoder(torch.nn.Module):
    # TODO: (1) skip connection, (2) two outputs (summaries weighted by forwards)
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 embedding_layer=None, residual=True, dropout_in=0., dropout_rec=0, debug=False):

        super(QelosSlotPtrQuestionEncoder, self).__init__()
        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device
        self.embedding_layer=embedding_layer
        # if vectors is not None:
        #     self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
        #     self.embedding_layer.weight.requires_grad = False
        # else:
        #     self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim)

        #self.lstm = q.FastestLSTMEncoder(self.embedding_dim, self.hidden_dim, bidir=self.bidirectional,
        #                                  dropout_in=self.dropout_in, dropout_rec=self.dropout_rec)

        self.lstm = NotSuchABetterEncoder(
            number_of_layer=self.number_of_layer,
            bidirectional=self.bidirectional,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            embedding_layer=embedding_layer,
            enable_layer_norm=False,
            mode='LSTM',
            debug=self.debug,
            residual=self.residual)


        dims = [self.hidden_dim]
        self.linear = torch.nn.Linear(dims[-1] * (1+self.bidirectional), 4)
        self.sm = torch.nn.Softmax(1)
        # for adapter
        outdim = dims[-1] * (1+self.bidirectional)
        self.adapt_lin = None

        if outdim != self.embedding_dim:
            self.adapt_lin = torch.nn.Linear(self.embedding_dim, outdim, bias=False)

    def return_encoder(self):
        return self.lstm

    '''original'''
    def forward(self, x):
        # embs = self.embedding_layer(x)
        # mask = tu.compute_mask(x)

        h = self.lstm.init_hidden(x.shape[0], self.device)

        if self.residual:
            ys, final_state, _, mask, embs, _ = self.lstm(x, h)
        else:
            ys, final_state, _, mask = self.lstm(x, h)

        ys = ys.transpose(1,0)
        # print('ys',ys.size())
        # ys = self.lstm(embs, mask=mask)

        # final_state = final_state.contiguous().view(x.size(0), -1)


        # get attention scores
        linear_scores = self.linear(ys)
        # s1 = scores
        scorer = linear_scores + torch.log(mask[:, :ys.size(1)].float().unsqueeze(2))
        scores = self.sm(scorer)  # (batsize, seqlen, 2)
        # scores[:,:,0],scores[:,:,1] =
        # get summaries
        # region skipper
        # print('embs',embs.size())
        skipadd = embs[:, :ys.size(1), :]
        # print('skipadd',skipadd.size())
        if self.adapt_lin is not None:
            skipadd = self.adapt_lin(skipadd)
        if not self.residual:
            ys = ys + skipadd
        # endregion
        ys = ys.unsqueeze(2)  # (batsize, seqlen, 1, dim)
        scores = scores.unsqueeze(3)  # (batsize, seqlen, 2, 1)
        b = ys * scores  # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)  # (batsize, 2, dim)

        ret = torch.cat([summaries[:, 0, :], summaries[:, 1, :],summaries[:, 2, :], summaries[:, 3, :]], 1)
        # print('ret',ret.size())

        # ret=torch.max(summaries,1)[0]
        return ret,scores


class QelosFlatEncoder(torch.nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 embedding_layer=None, residual=False, dropout_in=0., dropout_rec=0, debug=False,encoder=False):
        '''
               :param max_length: Max length of the sequence.
               :param hidden_dim: dimension of the output of the LSTM.
               :param number_of_layer: Number of LSTM to be stacked.
               :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
               :param vocab_size: Size of vocab / number of rows in embedding matrix
               :param bidirectional: boolean - if true creates BIdir LStm
               :param vectors: embedding matrix
               :param debug: Bool/ prints shapes and some other meta data.
               :param enable_layer_norm: Bool/ layer normalization.
               :param mode: LSTM/GRU.
               :param residual: Bool/ return embedded state of the input.

           TODO: Implement multilayered shit someday.
        '''
        super(QelosFlatEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device


        # if vectors is not None:
        #     self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
        #     self.embedding_layer.weight.requires_grad = True
        # else:
        #     self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        if encoder:
            self.lstm = encoder
        else:
            self.lstm = NotSuchABetterEncoder(
                number_of_layer=self.number_of_layer,
                bidirectional=self.bidirectional,
                embedding_dim=self.embedding_dim,
                max_length = self.max_length,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size,
                dropout=self.dropout,
                embedding_layer=embedding_layer,
                enable_layer_norm=False,
                mode = 'LSTM',
                debug = self.debug,
                residual=self.residual)

        self.adapt_lin = None   # Make layer if dims mismatch
        if residual and self.hidden_dim*2 != self.embedding_dim:
            self.adapt_lin = torch.nn.Linear(self.embedding_dim, self.hidden_dim*2, bias=False)

    def forward(self, x):
        # embs = self.embedding_layer(x)
        # mask = tu.compute_mask(x)

        h = self.lstm.init_hidden(x.shape[0],self.device)

        if self.residual:
            _, final_state, _, mask, embs, _ = self.lstm(x, h)
        else:
            _, final_state, _, mask = self.lstm(x, h)
        # final_state = self.lstm.y_n[-1]
        # print('final_state', final_state.size())
        final_state = final_state.contiguous().view(x.size(0), -1)
        # print('final_state', final_state.size())
        # if self.residual:
        #     if self.adapt_lin is not None:
        #         embs = self.adapt_lin(embs)
        #     meanpool = embs.sum(0)
        #     masksum = mask.float().sum(1).unsqueeze(1)
        #     meanpool = meanpool / masksum
        #     final_state = final_state + meanpool
        return final_state


class QelosSlotPtrChainEncoder(torch.nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 embedding_layer=None, residual=False, dropout_in=0., dropout_rec=0,debug=False,encoder=False):
        '''
               :param max_length: Max length of the sequence.
               :param hidden_dim: dimension of the output of the LSTM.
               :param number_of_layer: Number of LSTM to be stacked.
               :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
               :param vocab_size: Size of vocab / number of rows in embedding matrix
               :param bidirectional: boolean - if true creates BIdir LStm
               :param vectors: embedding matrix
               :param debug: Bool/ prints shapes and some other meta data.
               :param enable_layer_norm: Bool/ layer normalization.
               :param mode: LSTM/GRU.
               :param residual: Bool/ return embedded state of the input.

           TODO: Implement multilayered shit someday.
        '''
        super(QelosSlotPtrChainEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = \
            int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.device = device

        self.enc = QelosFlatEncoder(max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional, device, dropout=0.5, mode='LSTM',
                 enable_layer_norm=False, embedding_layer=embedding_layer, residual=self.residual,
                 dropout_in=self.dropout_in, dropout_rec=self.dropout_rec, debug=False,encoder=encoder)#.to(device)

    def forward(self, firstrels, secondrels,thirdrels, fourthrels):
        firstrels_enc = self.enc(firstrels)
        secondrels_enc = self.enc(secondrels)
        thirdrels_enc = self.enc(thirdrels)
        fouthrels_enc = self.enc(fourthrels)
        # cat???? # TODO
        enc = torch.cat([firstrels_enc, secondrels_enc,thirdrels_enc, fouthrels_enc], 1)
        # firstrels_enc0=firstrels_enc.unsqueeze(1)
        # secondrels_enc0=secondrels_enc.unsqueeze(1)
        # encs=torch.cat([firstrels_enc0,secondrels_enc0],1)
        # enc = torch.max(encs, 1)[0]
        return enc


class SMCNN(torch.nn.Module):
    """
        ## Improved Relation Detection

        Implementation of the paper here: https://arxiv.org/pdf/1704.06194.pdf.
        In our implementation, we first add then pool instead of the other way round.

        **NOTE: We assume that path encoder's last states are relevant, and we pool from them.**
    """

    def __init__(self, config,embedding_layer,device
                ):

        super(SMCNN, self).__init__()
        self.config=config
        self.embedding_layer=embedding_layer
        self.device=device
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, config.channel_size, (config.conv_kernel_1, config.conv_kernel_2), stride=1,
                            padding=(config.conv_kernel_1 // 2, config.conv_kernel_2 // 2)),
            # channel_in=1, channel_out=8, kernel_size=3*3
            nn.ReLU(True)).to(self.device)

        self.seq_maxlen = config.seq_maxlen + (config.conv_kernel_1 + 1) % 2
        self.rel_maxlen = config.rel_word_maxlen + (config.conv_kernel_2 + 1) % 2

        self.pooling = nn.MaxPool2d((config.seq_maxlen, 1),
                                    stride=(config.seq_maxlen, 1), padding=0).to(self.device)

        self.pooling2 = nn.MaxPool2d((1, config.rel_word_maxlen),
                                     stride=(1, config.rel_word_maxlen), padding=0).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(config.rel_word_maxlen * config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(20, 1)).to(self.device)

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_maxlen * config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(20, 1)).to(self.device)

        self.fc2 = nn.Sequential(
            nn.Linear(2, 1)).to(self.device)
        self.fc3 = nn.Sequential(
            nn.Linear(3, 1)).to(self.device)


    def matchPyramid(self, seq, rel, seq_len, rel_len):
        '''
        param:
            seq: (batch, _seq_len, embed_size)
            rel: (batch, _rel_len, embed_size)
            seq_len: (batch,)
            rel_len: (batch,)
return:
            score: (batch, 1)
        '''
        batch_size = seq.size(0)

        rel_trans = torch.transpose(rel, 1, 2)
        # (batch, 1, seq_len, rel_len)
        seq_norm = torch.sqrt(torch.sum(seq * seq, dim=2, keepdim=True))
        rel_norm = torch.sqrt(torch.sum(rel_trans * rel_trans, dim=1, keepdim=True))
        # if seq_norm==0:
        #     seq_norm=1
        # if rel_norm==0:
        #     rel_norm=1
        ones_seq=torch.ones_like(seq_norm)
        ones_rel=torch.ones_like(rel_norm)
        seq_norm=torch.where(seq_norm==0,ones_seq,seq_norm)
        rel_norm=torch.where(rel_norm==0,ones_rel,rel_norm)
        cross = torch.bmm(seq / seq_norm, rel_trans / rel_norm).unsqueeze(1)
        # print(cross)
        # (batch, channel_size, seq_len, rel_len)
        conv1 = self.conv(cross)
        channel_size = conv1.size(1)

        # (batch, seq_maxlen)
        # (batch, rel_maxlen)
        # print(seq_len, rel_len, self.seq_maxlen,self.rel_maxlen)
        dpool_index1, dpool_index2 = self.dynamic_pooling_index(seq_len, rel_len, self.seq_maxlen,
                                                                self.rel_maxlen)
        # print(dpool_index1)
        # print(dpool_index1)
        # print(dpool_index1.size())
        dpool_index1 = dpool_index1.unsqueeze(1).unsqueeze(-1).expand(batch_size, channel_size,
                                                                      self.seq_maxlen, self.rel_maxlen)
        # print(dpool_index1)
        # print(dpool_index1.size())
        # print(conv1.size())
        dpool_index2 = dpool_index2.unsqueeze(1).unsqueeze(2).expand_as(dpool_index1)
        conv1_expand = torch.gather(conv1, 2, dpool_index1)
        conv1_expand = torch.gather(conv1_expand, 3, dpool_index2)

        # (batch, channel_size, p_size1, p_size2)
        pool1 = self.pooling(conv1_expand).view(batch_size, -1)

        # (batch, 1)
        out = self.fc(pool1)

        pool2 = self.pooling2(conv1_expand).view(batch_size, -1)
        out2 = self.fc1(pool2)

        return out, out2

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = float(1.0 * max_len1 / float(len1_one))
            stride2 = float(1.0 * max_len2 / float(len2_one))
            # print(max_len1)
            # print(stride1)
            idx1_one = [floor(i / stride1) for i in range(max_len1)]
            # print(idx1_one)
            idx2_one = [floor(i / stride2) for i in range(max_len2)]
            return idx1_one, idx2_one

        batch_size = len(len1)
        index1, index2 = [], []
        for i in range(batch_size):
            idx1_one, idx2_one = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
            index1.append(idx1_one)
            index2.append(idx2_one)
        index1 = torch.LongTensor(index1)
        index2 = torch.LongTensor(index2)
        # if self.config.cuda:
        index1 = index1.to(self.device)
        index2 = index2.to(self.device)
        return Variable(index1), Variable(index2)



    def forward(self, ques_batch,pos_batch):
        """
        :params
            :ques: torch.tensor (batch, seq)
            :path_word: torch tenquessor (batch, seq)
            :path_rel_1: torch.tensor (batch, 1)
            :path_rel_2: torch.tensor (batch, 1)_q
        """


        # rows=[]
        # for i in range(pos_batch.size(0)):
        #     for j in range(pos_batch.size(1)):
        #         if pos_batch[i][j] not in [0,1,2,3]:
        #             rows.append(i)
        #             break
        # # pos_batch_ = torch.zeros((len(rows),pos_batch.size(1)))
        # pos_batch_=torch.zeros([pos_batch.size(0),pos_batch.size(1)],dtype=torch.long, device=self.device)
        # l=0
        # for i in range(pos_batch.size(0)):
        #     if i in rows:
        #         k=0
        #         for j in range(pos_batch.size(1)):
        #             if pos_batch[i][j] not in [0,1,2,3]:
        #                 pos_batch_[l][k]=pos_batch[i][j]
        #                 k+=1
        #         l+=1
        # for i in range(l,pos_batch.size(0)):
        #     pos_batch_[i]=pos_batch_[random.randint(0,l)]
        # pos_batch=pos_batch_
        #
        # ques_batch_ = torch.zeros([ques_batch.size(0), ques_batch.size(1)], dtype=torch.long, device=self.device)
        #
        # l=0
        # for i in range(ques_batch.size(0)):
        #     if i in rows:
        #         k = 0
        #         for j in range(ques_batch.size(1)):
        #             if ques_batch[i][j] not in [0, 1, 2, 3]:
        #                 ques_batch_[l][k] = ques_batch[i][j]
        #                 k += 1
        #         l+=1
        # for i in range(l,ques_batch.size(0)):
        #     ques_batch_[i]=ques_batch_[random.randint(0,l)]
        # ques_batch=ques_batch_
        # print('sdhtrjnty',pos_batch.size(),ques_batch.size())

        seqs = ques_batch
        seq_mask = tu.compute_mask(seqs)
        seq_len = (seq_mask.eq(1).long().sum(1))

        # print('pos_batch:', pos_batch.size())
        pos_mask = tu.compute_mask(pos_batch)
        # print('pos_mask:', pos_mask.size())
        pos_len = (pos_mask.eq(1).long().sum(1))




        seqs_emb = self.embedding_layer(seqs)
        pos_emb = self.embedding_layer(pos_batch)

        # for name, param in self.conv.named_parameters():
        #     print(name, param)
        #     if param.requires_grad:
        #         print('yes')
        #     else:
        #         print('no')
        pos_score3, pos_score4 = self.matchPyramid(seqs_emb, pos_emb, seq_len, pos_len)
        # neg_score3, neg_score4 = self.matchPyramid(seqs_emb, neg_emb, seq_len, neg_len)
        # print(pos_score3.size(),pos_score4.size())
        # print(torch.cat((pos_score3, pos_score4), 1).size())
        # print(self.fc2(torch.cat((pos_score3, pos_score4), 1)).size())
        pos_score = self.fc2(torch.cat((pos_score3, pos_score4), 1)).squeeze(1)
        # neg_score = self.fc2(torch.cat((neg_score3, neg_score4), 1)).squeeze()
        # print(pos_score.size())
        return pos_score


class Linear(torch.nn.Module):
    """
        ## Improved Relation Detection

        Implementation of the paper here: https://arxiv.org/pdf/1704.06194.pdf.
        In our implementation, we first add then pool instead of the other way round.

        **NOTE: We assume that path encoder's last states are relevant, and we pool from them.**
    """

    def __init__(self,device):

        super(Linear, self).__init__()
        self.device=device
        self.fc = nn.Sequential(
            nn.Linear(2, 1)).to(self.device)

    def forward(self, scores1,scores2):
        # print(scores1.size(),scores2.size())
        if len(scores1.size())==0:
            return scores1
        score=self.fc(torch.cat([scores1.unsqueeze(1),scores2.unsqueeze(1)],1)).squeeze()
        return score


class AdaptedBERTEncoderPair(AdaptedBERTEncoderSingle):
    """
    Adapts from worddic.
    Output is generated by encoding the pair in one sequence.
    """
    def forward(self, a, b):
        """
        :param a:   (batsize, seqlen_a)
        :param b:   (batsize, seqlen_b)
        :return:
        """
        assert(len(a) == len(b))
        # transform inpids
        newinp = torch.zeros_like(a)
        typeflip = []
        maxlen = 0
        for i in range(len(a)):    # iter over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(a.size(1)):     # iter over seqlen
                wordpieces = self.oldD2D[a[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
            newinp[i, last_nonpad] = self.D["[SEP]"]
            k = last_nonpad + 1
            typeflip.append(k)
            for j in range(b.size(1)):
                wordpieces = self.oldD2D[b[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
            newinp[i, last_nonpad] = self.D["[SEP]"]
            maxlen = max(maxlen, last_nonpad + 1)
        newinp = newinp[:, :maxlen]

        # do forward
        typeids = torch.zeros_like(newinp)
        for i, flip in enumerate(typeflip):
            typeids[i, flip:] = 1
        padmask = newinp != self.pad_id
        _, poolout = self.bert(newinp, typeids, padmask)
        poolout = self.dropout(poolout)
        logits = self.lin(poolout)
        return logits

