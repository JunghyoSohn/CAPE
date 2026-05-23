import torch
import numpy as np

import torch

from utils import (
    full_DAG,
    pre_pruning_with_parents_score,
    get_parents_score,
    add_edge_with_parents_score,
)


def _resolve_device(device):
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _stein_hess(X, device, eta_G, eta_H, s=None):
    """Estimate diagonal Hessian entries via Stein identities."""
    n, _ = X.shape
    target_device = _resolve_device(device)
    X = X.to(target_device)
    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s**2)) / s

    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(target_device)), nablaK)

    nabla2K = torch.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
    hessian = -G**2 + torch.matmul(
        torch.inverse(K + eta_H * torch.eye(n).to(target_device)), nabla2K
    )
    return hessian.to("cpu")


def compute_top_order(X, device=None, eta_G=0.001, eta_H=0.001, dispersion="mean"):
    """Compute node ordering and parents score following CaPS."""
    target_device = _resolve_device(device)
    n, d = X.shape
    full_X = X
    order = []
    active_nodes = list(range(d))
    for _ in range(d - 1):
        H = _stein_hess(X, target_device, eta_G, eta_H)
        if dispersion == "mean":
            idx = int(H.mean(axis=0).argmax())
        else:
            raise Exception("Unknown dispersion criterion")

        order.append(active_nodes[idx])
        active_nodes.pop(idx)
        X = torch.hstack([X[:, 0:idx], X[:, idx + 1 :]])
    order.append(active_nodes[0])
    order.reverse()

    # compute parents score
    full_H = _stein_hess(full_X, target_device, eta_G, eta_H).mean(axis=0)
    parents_score = np.zeros((d, d))
    for i in range(d):
        curr_X = torch.hstack([full_X[:, 0:i], full_X[:, i + 1 :]])
        curr_H = _stein_hess(curr_X, target_device, eta_G, eta_H).mean(axis=0)
        parents_score[i] = get_parents_score(curr_H, full_H, i)
    # print(parents_score)

    return order, parents_score


def prepare_caps(train_set, device=None, pre_pruning=True, lambda1=50.0):
    """Prepare tensors, order, parents score, and init DAG for CaPS."""
    train_tensor = torch.tensor(train_set, dtype=torch.float32)
    order, parents_score = compute_top_order(train_tensor, device)

    if pre_pruning:
        init_dag = pre_pruning_with_parents_score(
            full_DAG(order), parents_score, lambda1
        )
    else:
        init_dag = full_DAG(order)
    return train_tensor, order, parents_score, init_dag


def finalize_caps(dag, parents_score, lambda2=50.0, add_edge=True):
    """Optionally add edges back using parents score heuristics."""
    if add_edge:
        dag = add_edge_with_parents_score(dag, parents_score, lambda2)
    return dag
