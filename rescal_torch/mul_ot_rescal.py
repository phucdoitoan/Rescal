
import torch
import torch.nn as nn
from rescal_autograd import Rescal
from itertools import combinations
from wasserstein_ot.sinkhorn import SinkhornDistance


class MulOt_Rescal(nn.Module):

    def __init__(self, model_configs, alphas, device='cpu'):

        super(MulOt_Rescal, self).__init__()

        self.model_list = []
        for config in model_configs:
            self.model_list.append(Rescal(*config))

        self.model_num = len(model_configs)

        self.device = device

        self.alphas = alphas
        self.P_dict = None

        self.to_device()

        self.sinkhorn = SinkhornDistance(eps=1e-9, max_iter=100, device=self.device)

    def forward(self, indices_list):
        """note: P_dict: keys are (i,j) as comb below"""

        loss_list = []
        # independent losses from each model
        for i in range(self.model_num):
            loss_list.append(self.model_list[i](*indices_list[i]))

        # sinkhorn losses from each pair of models
        # produce pair of indexes i, j indicating which models are in concerned
        w_loss = 0.0
        if sum(self.alphas.values()) != 0:
            comb = combinations(list(range(self.model_num)), 2)
            for i, j in comb:
                tmp_P = self.P_dict[(i,j)]
                tmp_alpha = self.alphas[(i,j)]
                tmp_cost = self.ot_loss(self.model_list[i], self.model_list[j], indices_list[i], indices_list[j], tmp_P)
                w_loss += tmp_alpha * tmp_cost

        return loss_list, w_loss

    def ot_loss(self, model1, model2, indices1, indices2, P):
        """
        Compute the ot_loss between two models, with indices1 and indices2
        Args:
            model1:
            model2:
            indices1: tuple of (heads, tails, n_heads, n_tails, rels)
            indices2: tuple of (heads, tails, n_heads, n_tails, rels)

        Returns:
            optimal transport between the two models
        """

        # todo: need to check if P is the plan between model1 and model2. not the inverse: between model2 and model1

        idx1 = torch.cat((indices1[0], indices1[1], indices1[2], indices1[3]), dim=0)
        idx2 = torch.cat((indices2[0], indices2[1], indices2[2], indices2[3]), dim=0)

        #assert len(idx1) == len(idx2), 'The length of entity indices must be equal'

        emb1 = model1.ent_embedding[idx1]
        emb2 = model2.ent_embedding[idx2]

        emb1 = emb1.unsqueeze(1)
        emb2 = emb2.unsqueeze(0)
        norm = torch.sum((emb1 - emb2)**2, dim=-1)

        P_sliced = P[idx1][:, idx2]

        cost = torch.sum(norm * P_sliced)

        return cost

    def to_device(self):
        for model in self.model_list:
            model.to(self.device)

    def update_P(self):
        sinkhorn_cost = 0.0
        with torch.no_grad():
            for (i, j) in combinations(list(range(self.model_num)), 2):
                skn_cost, tmp_P, _ = self.sinkhorn(self.model_list[i].ent_embedding,
                                              self.model_list[j].ent_embedding)
                self.P_dict[(i, j)] = tmp_P

                sinkhorn_cost += skn_cost.item()

        return sinkhorn_cost / 3


if __name__ == '__main__':
    configs = [(1000, 100, 50)]
    mul_ot_rescal = MulOt_Rescal(configs)
    indices_list = [(torch.LongTensor(range(10)), torch.LongTensor(range(10,20)), torch.LongTensor(range(20,30)), torch.LongTensor(range(30,40)), torch.LongTensor(range(40,50)))]
    mul_ot_rescal(indices_list)


