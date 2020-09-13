
import torch
import torch.nn as nn
from mul_ot_rescal import MulOt_Rescal
from data_structures import KnowledgeGraph
import pandas as pd
from torch.utils.data import DataLoader
from rescal_autograd import UniformNegativeSampler
from itertools import combinations
from tqdm import tqdm
from wasserstein_ot.sinkhorn import SinkhornDistance



# constant values
EMB_DIM = 10 #50
MARGIN = 1

EPOCHS = 25
BATCH_SIZE = 100
LEARNING_RATE = 0.01

ALPHAS = {
    (0,1): 5, #5,
    (0,2): 5, #5,
    (1,2): 5, #5,
}

data_name = 'FB15k-237'

df1 = pd.read_csv('../data/%s/divided/train1.csv' %data_name, delimiter='\t')
df2 = pd.read_csv('../data/%s/divided/train2.csv' %data_name, delimiter='\t')
df3 = pd.read_csv('../data/%s/divided/train3.csv' %data_name, delimiter='\t')
print('done load file into pd.DataFrame')

kg1 = KnowledgeGraph(df1)
kg2 = KnowledgeGraph(df2)
kg3 = KnowledgeGraph(df3)

print('df1: %s - kg1: n_ent: %s, n_rel: %s, n_facts: %s' %(df1.shape, kg1.n_ent, kg1.n_rel, kg1.n_facts))
print('df2: %s - kg2: n_ent: %s, n_rel: %s, n_facts: %s' %(df2.shape, kg2.n_ent, kg2.n_rel, kg2.n_facts))
print('df3: %s - kg3: n_ent: %s, n_rel: %s, n_facts: %s' %(df3.shape, kg3.n_ent, kg3.n_rel, kg3.n_facts))

kg_list = [kg1, kg2, kg3]

configs = []
for kg in kg_list:
    config = (kg.n_ent, kg.n_rel, EMB_DIM)
    configs.append(config)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initiate the model
mul_ot_model = MulOt_Rescal(model_configs=configs, alphas=ALPHAS, device=device)
print('mul_ot_model device: ', mul_ot_model.device)
print('MODEL NUM: ', mul_ot_model.model_num)

criterion = nn.MarginRankingLoss(margin=MARGIN, reduction='mean')

dataloader_list = [DataLoader(kg, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) for kg in kg_list] # can adjust the batch_size for each kg separatedly
negative_sampler_list = [UniformNegativeSampler(kg) for kg in kg_list]


optimizer = torch.optim.Adam([param for model in mul_ot_model.model_list for param in model.parameters()], lr=LEARNING_RATE, weight_decay=1e-5)

mul_ot_model.P_dict = {
    (i,j): torch.ones(kg_list[i].n_ent, kg_list[j].n_ent) / (kg_list[i].n_ent * kg_list[j].n_ent) for i,j in combinations(list(range(mul_ot_model.model_num)), 2)
}

sinkhorn = SinkhornDistance(eps=1e-9, max_iter=100, device=device)

epochs_iter = tqdm(range(EPOCHS), unit='epoch')

for epoch in epochs_iter:

    running_loss = 0.0

    total_batch = 0

    for batch_tuples in tqdm(zip(*dataloader_list)):
        total_batch += 1

        optimizer.zero_grad()
        #print('hello 1')

        pos_neg_indices = []
        for i, (h, t, r) in enumerate(batch_tuples):
            n_h, n_t = negative_sampler_list[i].corrupt_batch(h, t, r)
            pos_neg_indices.append((h, t, n_h, n_t, r))

        #print('hello 2')
        pos_neg_list, w_loss = mul_ot_model(indices_list=pos_neg_indices)

        #print('hello 3')

        loss = w_loss
        for (p_score, n_score) in pos_neg_list:
            loss += criterion(p_score, n_score, torch.ones_like(p_score))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print('hello 4')

        epochs_iter.set_description(
            'Epoch %s | mean loss: %.5f' % (epoch + 1, running_loss / total_batch)
        )

    # update P
    if sum(mul_ot_model.alphas.values()) != 0:
        print('compute sinkhorn distance between pairs of datasets')
        sinkhorn_cost = mul_ot_model.update_P()
    else:
        print('compute each dataset independently: do not use sinkhorn')
        sinkhorn_cost = None

    epochs_iter.set_description(
        'Epoch %s | mean loss: %.5f | sinkhorn_cost: %s' % (epoch + 1, running_loss / total_batch, sinkhorn_cost)
    )

    print('Epochs: %s - mean loss: %.5f' %(epoch + 1, running_loss/total_batch))


# compute OT luc nao => after each epochs




