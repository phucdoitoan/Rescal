from torch.optim import Adam

from torchkge.models import RESCALModel
from torchkge.utils import MarginLoss

from torchkge.data_structures import KnowledgeGraph
from data_structures import KnowledgeGraph as custome_KnowledgeGraph
import pandas as pd
from rescal_autograd import Rescal
import pickle
from sklearn.metrics import precision_recall_curve, auc
import torch
from torchkge.sampling import UniformNegativeSampler
from tqdm import tqdm
from torch.utils.data import DataLoader

def main():
    # Define some hyper-parameters for training
    emb_dim = 40
    lr = 0.001 #0.0004
    margin = 1 #0.5
    n_epochs = 50 #1000
    b_size = 100 #32768

    print('emb_dim: ', emb_dim)

    # Load dataset
    #kg_train, kg_val, kg_test = load_fb15k()
    #kg_df = pd.read_csv('data/filtered_uml.txt', delimiter='\t', header=None, names=['from', 'rel', 'to'])
    kg_df = pd.read_csv('data/filtered_uml.txt', delimiter='\t', header=None, names=['head', 'rel', 'tail'])
    #kg = KnowledgeGraph(df=kg_df)
    kg = custome_KnowledgeGraph(df=kg_df)
    #kg_train, kg_test = kg.split_kg(share=0.9)
    kg_train = kg

    print('n_ent: ', kg_train.n_ent)
    print('n_rel: ', kg_train.n_rel)
    print('n_facts: ', kg_train.n_facts)

    # Define the model and criterion
    #model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    #model = RESCALModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
    model = Rescal(kg_train.n_ent, kg_train.n_rel, emb_dim)
    #print('Used Rescal model insteads')

    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = UniformNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=b_size)

    epochs_iter = tqdm(range(n_epochs), unit='epoch')
    for epoch in epochs_iter:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            pos, neg = model(h, t, n_h, n_t, r)
            #loss = criterion(pos, neg, torch.ones_like(pos))
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epochs_iter.set_description(
            'Epoch %s | mean loss: %.5f' % (epoch + 1, running_loss / len(dataloader))
        )

    model.normalize_parameters()

    with open('data/filtered_uml_test.pkl', 'rb') as file:
        test_dict = pickle.load(file)

    test_heads = test_dict['head']
    test_tails = test_dict['tail']
    test_rels = test_dict['rel']
    test_label = test_dict['label']

    test_heads = [kg_train.ent2id[ent] for ent in test_heads]
    test_rels = [kg_train.rel2id[rel] for rel in test_rels]
    test_tails = [kg_train.ent2id[ent] for ent in test_tails]

    with torch.no_grad():
        scores = model.scoring_function(torch.LongTensor(test_heads), torch.LongTensor(test_tails), torch.LongTensor(test_rels))

    print('len scores: ', len(scores))
    print('len labels: ', len(test_label))
    print('n_facts in test: ', test_label.sum())

    prec, recall, _ = precision_recall_curve(test_label, scores)
    auc_pr = auc(recall, prec)

    print('PR-AUC is: ', auc_pr)


if __name__ == "__main__":
    main()
