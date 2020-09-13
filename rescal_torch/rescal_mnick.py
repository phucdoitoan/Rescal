

from scipy.io import loadmat
from skge import RESCAL, StochasticTrainer

import pickle
from sklearn.metrics import precision_recall_curve, auc


def load_data(mat_file):

    data = loadmat(mat_file)['Rs']

    xs = []
    ys = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                xs.append((i, j, k))
                ys.append(data[i,j,k])

    return data.shape[0], data.shape[2], xs, ys

def main():

    N, K, xs, ys = load_data('data/filtered_kinship.mat')

    print('N: ', N)
    print('K: ', K)
    print('xs: ', len(xs))

    model = RESCAL((N, N, K), 50)

    trainer = StochasticTrainer(
        model,
        nbatches=100,
        max_epochs=1,
        learning_rate=0.01,
    )

    print('xs: ', xs[:5])
    print('ys: ', ys[:5])

    #trainer.fit(xs, ys)

    with open('data/filtered_kinship_test.pkl', 'rb') as file:
        test_dict = pickle.load(file)

    test_heads = test_dict['head']
    test_tails = test_dict['tail']
    test_rels = test_dict['rel']
    test_label = test_dict['label']

    print('test_labels: ', len(test_label), test_label.dtype)
    print('n_facts in test: ', test_label.sum())

    max_epochs = 100

    for epoch in range(max_epochs):
        trainer.fit(xs, ys)

        scores = model._scores(test_heads, test_rels, test_tails)
        prec, recall, _ = precision_recall_curve(test_label, scores)
        auc_pr = auc(recall, prec)
        print('PR-AUC is: ', auc_pr)


if __name__ == '__main__':
    main()