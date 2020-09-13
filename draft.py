

from wasserstein_ot.sinkhorn import SinkhornDistance
import torch


def main():

    n = 6
    pts1 = torch.rand((n, 2)) * 10
    p1 = torch.ones(n)
    p1 /= p1.sum()

    replacement = torch.tensor([2,2])

    pts2 = pts1 - replacement

    p2 = p1.clone()

    device = torch.device('cpu')
    sinkhorn = SinkhornDistance(eps=1e-5, max_iter=100, device=device)

    cost, P, C = sinkhorn(pts1, pts2, p1, p2)

    P_matching = torch.argmax(P, dim=1)
    C_matching = torch.argmin(C, dim=1)

    print('P_matching: ', P_matching)
    print('C_matching: ', C_matching)

    print('P: ', P)


if __name__ == '__main__':
    main()

