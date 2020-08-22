

# implementing rescal using autograd of pytorch

import torch
import torch.nn as nn



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

        pos_regul = self.regularization(sub, obj, rel)
        neg_regul = self.regularization(n_sub, n_obj, rel)

        return pos, neg, pos_regul, neg_regul

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






