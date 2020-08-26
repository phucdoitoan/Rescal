

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader
import pandas as pd
from data_structures import KnowledgeGraph
from rescal_autograd import Rescal, UniformNegativeSampler
from tqdm import tqdm

import pickle
from sklearn.metrics import precision_recall_curve, auc

from torchkge.models import RESCALModel


def array2dict(arr):
    dict_id = {}
    for data in arr:
        if data not in dict_id.keys():
            dict_id[data] = len(dict_id)

    return dict_id

def main():

    # load dataset
    #train_df = pd.read_csv('data/FB15k-237/train.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    #valid_df = pd.read_csv('data/FB15k-237/valid.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    #test_df = pd.read_csv('data/FB15k-237/test.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])

    # load dataset
    #train_df = pd.read_csv('data/WN18RR/train.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    #valid_df = pd.read_csv('data/WN18RR/valid.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    #test_df = pd.read_csv('data/WN18RR/test.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])

    # making dict with keys of entities and relations, values of integer index
    #ent2id = array2dict(pd.concat([train_df['head'], train_df['tail'], valid_df['head'], valid_df['tail'], test_df['head'], test_df['tail']]))
    #rel2id = array2dict(pd.concat([train_df['rel'], valid_df['rel'], test_df['rel']]))

    #train_kg = KnowledgeGraph(df=train_df, ent2id=ent2id, rel2id=rel2id)
    #valid_kg = KnowledgeGraph(df=valid_df, ent2id=ent2id, rel2id=rel2id)
    #test_kg = KnowledgeGraph(df=test_df, ent2id=ent2id, rel2id=rel2id)
    # NOTE: need to have a same ent2id and rel2id for all train, valid, test kg

    # load kinship dataset
    kinship_df = pd.read_csv('data/filtered_uml.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    kinship_kg = KnowledgeGraph(df=kinship_df)

    #train_kg, test_kg = kinship_kg.split_kg(size=(0.9,))
    train_kg = kinship_kg

    print('n_ent: ', train_kg.n_ent)
    print('n_rel: ', train_kg.n_rel)
    print('n_facts: ', train_kg.n_facts)

    # define hyperparameters for training
    emb_dim = 50 #100
    lr = 0.001
    margin = 1
    n_epochs = 50 #1000
    b_size = 100 #10000 #32768

    alpha = 10 #0 #10

    print('alpha: ', alpha)
    print('emb_dim: ', emb_dim)

    # define model and criterion
    model = Rescal(ent_num=train_kg.n_ent, rel_num=train_kg.n_rel, rank=emb_dim)
    #model = RESCALModel(emb_dim, train_kg.n_ent, train_kg.n_rel)
    criterion = nn.MarginRankingLoss(margin=margin, reduction='sum')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer to use
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = UniformNegativeSampler(train_kg)
    dataloader = DataLoader(train_kg, batch_size=b_size)

    epochs_iter = tqdm(range(n_epochs), unit='epoch')
    for epoch in epochs_iter:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            #pos, neg, pos_regul, neg_regul = model(h, t, n_h, n_t, r)
            pos, neg, *_ = model(h, t, n_h, n_t, r)

            loss = criterion(pos, neg, torch.ones_like(pos)) #+ alpha * (pos_regul + neg_regul) / 2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epochs_iter.set_description(
            'Epoch %s | mean loss: %.5f' %(epoch + 1, running_loss / len(dataloader))
        )

    #model.normalize_parameters()

    with open('data/filtered_uml_test.pkl', 'rb') as file:
        test_dict = pickle.load(file)

    test_heads = test_dict['head']
    test_tails = test_dict['tail']
    test_rels = test_dict['rel']
    test_label = test_dict['label']

    test_heads = [train_kg.ent2id[ent] for ent in test_heads]
    test_rels = [train_kg.rel2id[rel] for rel in test_rels]
    test_tails = [train_kg.ent2id[ent] for ent in test_tails]

    with torch.no_grad():
        scores = model.scoring_function(torch.LongTensor(test_heads), torch.LongTensor(test_tails), torch.LongTensor(test_rels))

    print('len scores: ', len(scores))
    print('len labels: ', len(test_label))
    print('n_facts in test: ', test_label.sum())

    prec, recall, _ = precision_recall_curve(test_label, scores)
    auc_pr = auc(recall, prec)

    print('PR-AUC is: ', auc_pr)

if __name__ == '__main__':
    main()


