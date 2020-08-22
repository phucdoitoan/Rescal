

from scipy.io import loadmat, savemat
import numpy as np
import pickle

def split(mat_source_file, filtered_mat_file, filtered_test_pkl, key='Rs'):

    try:
        tensor = loadmat(mat_source_file)[key]
    except:
        tensor = loadmat(mat_source_file)['R']
        tensor = np.nan_to_num(tensor)

    n, k = tensor.shape[0], tensor.shape[2]

    SZ = n*n*k
    print('Size: ', SZ)

    fold_sz = int(SZ/10)

    idx = np.arange(SZ)
    np.random.shuffle(idx)

    test_idx = idx[:fold_sz]

    heads, tails, rels = np.unravel_index(test_idx, shape=(n, n, k))

    labels = tensor[heads, tails, rels]

    for i in range(len(heads)):
        tensor[heads[i], tails[i], rels[i]] = 0

    savemat(filtered_mat_file, {'Rs': tensor})

    with open(filtered_test_pkl, 'wb') as file:
        pickle.dump({'head': heads, 'tail': tails, 'rel': rels, 'label': labels}, file)

    print('Done: filtered total %s facts' %(labels.sum()))

if __name__ == '__main__':
    split('uml.mat', 'filtered_uml.mat', 'filtered_uml_test.pkl')





