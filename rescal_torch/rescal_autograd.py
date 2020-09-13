

# refer to:
# https://torchkge.readthedocs.io/en/latest/_modules/torchkge/models/interfaces.html#Model
# and
# https://torchkge.readthedocs.io/en/latest/_modules/torchkge/models/bilinear.html#RESCALModel

# implementing rescal using autograd of pytorch

import torch
import torch.nn as nn


# helper function
def get_true_targets(dictionary, e_idx, r_idx, true_idx, i):
    # refer to : https://github.com/torchkge-team/torchkge/blob/master/torchkge/utils/modeling.py
    """

    Args:
        dictionary: keys (ent_idx, rel_idx), values: list of ent_idx s.t the triplet is true fact
        e_idx: torch.Tensor, shape: (batch_size) Long tensor of ent_idx
        r_idx: shape: (batch_size), Long Tensor of rel_idx
        true_idx: Long Tensor of ent_idx, s.t (e_idx, r_idx, true_idx) is a true fact
        i: the index of the batch is currently treated

    Returns:
        true_targets: Long Tensor of ent_idx s.t true_targets != true_idx and (e_idx, r_idx, and ent in true_targets) is a true fact
    """

    true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()

    if len(true_targets) == 1:
        return None
    true_targets.remove(true_idx[i].item())

    return torch.LongTensor(list(true_targets))

def get_rank(scores, true, low_values=False):
    # refer to : https://github.com/torchkge-team/torchkge/blob/master/torchkge/utils/operations.py
    """
    Compute the rank of entity at index true[i]
    Args:
        scores: torch.Tensor, shape: (b_size, n_ent), scores for each entities
        true: (b_size,)
              true[i] is the index of the true entity in the batch
        low_values: If True, best rank is the lowest score, else it is the highest

    Returns:
        ranks: torch.Tensor, shape: (b_size) ranks of the true entities in the batch
               ranks[i] - 1 is the number of entities which have better scores in scores than the one with index true[i]
    """
    true_scores = scores.gather(dim=1, index=true.long().view(-1, 1))

    if low_values:
        return (scores < true_scores).sum(dim=1) + 1
    else:
        return (scores > true_scores).sum(dim=1) + 1

class UniformNegativeSampler:
    def __init__(self, kg, n_neg=1):
        self.kg = kg
        self.n_ent = kg.n_ent
        self.n_facts = kg.n_facts

        self.n_neg = n_neg

    def corrupt_batch(self, heads, tails, relations=None, n_neg=None):
        """Sample negative examples from positive examples, according to Bordes et al. 2013"""
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device), 'heads and tails must be on a same device'

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted
        mask = torch.bernoulli(torch.ones(batch_size * n_neg, device=device)/2).double()

        n_h_cor = int(mask.sum().item())
        neg_heads[mask==1] = torch.randint(low=0, high=self.n_ent, size=(n_h_cor,), device=device)
        neg_tails[mask==0] = torch.randint(low=0, high=self.n_ent, size=(batch_size * n_neg - n_h_cor,), device=device)

        return neg_heads.long(), neg_tails.long()

class Rescal(nn.Module):

    def __init__(self, ent_num, rel_num, rank):
        super(Rescal, self).__init__()

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.rank = rank

        self.ent_embedding = nn.Parameter(data=torch.empty(self.ent_num, self.rank))
        self.rel_embedding = nn.Parameter(data=torch.empty(self.rel_num, self.rank, self.rank))
        nn.init.xavier_uniform_(self.ent_embedding.data)
        nn.init.xavier_uniform_(self.rel_embedding.data)

    def scoring_function(self, sub, obj, rel):
        """Compute the scoring function defined as in RESCAL: h^T \\cdot M_r \\cdot t"""

        sub_emb = nn.functional.normalize(self.ent_embedding[sub], p=2, dim=1)
        obj_emb = nn.functional.normalize(self.ent_embedding[obj], p=2, dim=1)

        #sub_emb = self.ent_embedding[sub]
        #obj_emb = self.ent_embedding[obj]

        rel_emb = self.rel_embedding[rel]

        sub_emb = sub_emb.unsqueeze(1)
        sub_rel = (sub_emb @ rel_emb).squeeze(1)
        score = sub_rel * obj_emb

        return score.sum(dim=1)

    def forward(self, sub, obj, n_sub, n_obj, rel):
        """
        compute the loss in (2) of Rescal
        Args:
            obj: torch.LongTensor: batch of indices of entities in object role
            sub: torch.LongTensor: batch of indices of entities in subject role
            rel: torch.LongTensor: batch of indices of relations between the entities in objs and subs

        Returns:
            the loss in (2) computed in batches of data
        """

        assert sub.shape[0] == n_sub.shape[0], 'The length of positive and negative examples must be equal'

        pos = self.scoring_function(sub, obj, rel)
        neg = self.scoring_function(n_sub, n_obj, rel)

        #pos_regul = self.regularization(sub, obj, rel)
        #neg_regul = self.regularization(n_sub, n_obj, rel)

        return pos, neg, #pos_regul, neg_regul

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in the original paper?????
        It should be called at the end of each training epoch and at the end of the traning as well.
        According to torchkge package
        """
        self.ent_embedding.data = nn.functional.normalize(self.ent_embedding.data, p=2, dim=1)


    def regularization(self, sub, obj, rel):
        h = self.ent_embedding[sub]
        t = self.ent_embedding[obj]
        r = self.rel_embedding[rel]

        regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3

        return regul

    def lp_scoring_function(self, h_emb, t_emb, r_emb):
        """
        Given an entities e and a relation r, compute the scores of triplet (e, r, e') or (e', r, e) for all entities e'
        Compute in batches (batch size: b_size)
        Depends on the shapes of h_emb and t_emb to determine (e, r, e') or (e', r, e) to be compute
        Args:
            h_emb: torch.Tensor, shape: (b_size, emb_dim) or (b_size, n_ent, emb_dim)
            t_emb: torch.Tensor, shape: (b_size, n_ent, emb_dim) or (b_size, emb_dim)
            r_emb: torch.Tensor, shape: (b_size, emb_dim)

        Returns:
            scores: torch.Tensor, shape: (b_size, n_ent)
                    Scores of each triplet (e, r, e') ( or (e', r, e) ) for all entities e'
        """

        b_size = h_emb.shape[0]

        if len(h_emb.shape) == 2 and len(t_emb.shape) == 3:
            # this is the tail completion case in link prediction: compute scores for (e, r, e')

            h_emb = h_emb.unsqueeze(1)
            scores = (h_emb @ r_emb) * t_emb

            return scores.sum(dim=-1)

        else:
            # this is the head completion case in link prediction: compute scores for (e', r, e)
            t_emb = t_emb.unsqueeze(1)
            scores = (h_emb @ r_emb) * t_emb

            return scores.sum(dim=-1)

    def lp_prep_cands(self, h_idx, t_idx, r_idx):

        b_size = h_idx.shape[0]

        h_emb = self.ent_embedding[h_idx]
        t_emb = self.ent_embedding[t_idx]
        r_emb = self.rel_embedding[r_idx]

        candidates = self.ent_embedding.data.unsqueeze(0)
        candidates = candidates.expand(b_size, -1, -1)

        return h_emb, t_emb, candidates, r_emb

    def lp_compute_ranks(self, e_emb, candidates, r_emb, e_idx, r_idx, true_idx, dictionary, heads=1):

        b_size = r_idx.shape[0]

        if heads == 1:
            scores = self.lp_scoring_function(e_emb, candidates, r_emb)
        else:
            scores = self.lp_scoring_function(candidates, e_emb, r_emb)

        # filter out the true negative samples by assigning -Inf score
        filt_scores = scores.clone()
        for i in range(b_size):
            true_targets = get_true_targets(dictionary, e_idx, r_idx, true_idx, i)

            if true_targets is None:
                continue

            filt_scores[i][true_targets] = - float('Inf')

        # extract the ranks of the true entity
        rank_true_entities = get_rank(scores, true_idx)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def lp_helper(self, h_idx, t_idx, r_idx, kg):
        """
        Compute the head and tail ranks and filtered ranks of the current batch
        Args:
            h_idx: shape: (b_size)
            t_idx: shape: (b_size)
            r_idx: shape: (b_size)
            kg: knowledge graph

        Returns:
            rank_true_tails: shape (b_size)
            filt_rank_true_tail: shape (b_size)
            rank_true_heads: shape (b_size)
            filt_rank_true_head: shape (b_size)
        """

        h_emb, t_emb, candidates, r_emb = self.lp_prep_cands(h_idx, t_idx, r_idx)

        rank_true_tails, filt_rank_true_tails = self.lp_compute_ranks(
            h_emb, candidates, r_emb, h_idx, r_idx, t_idx, kg.dict_of_tails, heads=1
        )

        rank_true_heads, filt_rank_true_heads = self.lp_compute_ranks(
            t_emb, candidates, r_emb, t_idx, r_idx, h_idx, kg.dict_of_heads, heads=-1
        )

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads






