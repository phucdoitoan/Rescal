from torch.optim import Adam

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import TransEModel, RESCALModel
from torchkge.utils.datasets import load_fb15k
from torchkge.utils import Trainer, MarginLoss
from torchkge.data_structures import KnowledgeGraph
import pandas as pd

from rescal_autograd import Rescal


def main():
    # Define some hyper-parameters for training
    emb_dim = 100
    lr = 0.0004
    margin = 1
    n_epochs = 10
    batch_size = 10000 #32768

    # Load dataset
    kg_train, kg_val, kg_test = load_fb15k()
    #kinship_df = pd.read_csv('data/kinship.txt', delimiter='\t', header=None, names=['from', 'rel', 'to'])
    #kinship_kg = KnowledgeGraph(df=kinship_df)
    #kg_train, kg_test = kinship_kg.split_kg(share=0.9)

    print('n_ent: ', kg_train.n_ent)
    print('n_rel: ', kg_train.n_rel)
    print('n_facts: train: %s, test: %s' % (kg_train.n_facts, kg_test.n_facts))
    #print('n_facts: total: %s, train: %s, test: %s' %(kinship_kg.n_facts, kg_train.n_facts, kg_test.n_facts))

    # Define the model and criterion
    model = RESCALModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
    print('RESCALModel')
    #model = Rescal(kg_train.n_ent, kg_train.n_rel, emb_dim)
    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                      optimizer=optimizer, sampling_type='bern', use_cuda=None,)

    for _ in range(50):
        trainer.run()

        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(200, 10)
        evaluator.print_results(k=[1,3,10])


if __name__ == "__main__":
    main()