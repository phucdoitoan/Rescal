

from scipy.io import loadmat
import csv


def tensor2triplet(mat_file, triplet_file, key='Rs'):
    """
    convert relational data in the form of a tensor to triplet form of (head, rel, tail)
    Args:
        mat_file: mat_file containing dict of data
        key: containg key of the dict whose valuse is tensor of size n x n x k: modeling relationship of k relations between n entities
    Returns:
        a csv file containing the triplets
    """

    tensor = loadmat(mat_file)[key]

    print('tensor: ', tensor.shape)

    n_facts = 0
    with open(triplet_file, 'w') as file:
        for head in range(tensor.shape[0]):
            for tail in range(tensor.shape[1]):
                for rel in range(tensor.shape[2]):
                    if tensor[head, tail, rel] == 1:
                        file.write('%s\t%s\t%s\n' %(head, rel, tail))
                        n_facts += 1

    print('Finished converting! Total: %s entities, %s relations, %s facts' %(tensor.shape[0], tensor.shape[2], n_facts))

if __name__ == '__main__':

    tensor2triplet('filtered_uml.mat', 'filtered_uml.txt')


