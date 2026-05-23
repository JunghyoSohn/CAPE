import numpy as np
import torch

from utils import full_DAG


EPS = 1e-12


def _resolve_device(device):
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _kernel_width(X):
    X_diff = X.unsqueeze(1) - X
    dist = torch.norm(X_diff, dim=2)
    dist_flat = dist.flatten()
    nonzero = dist_flat[dist_flat > 0]
    if nonzero.numel() == 0:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    return nonzero.median()


def _evaluate_kernel(X, s):
    X_diff = X.unsqueeze(1) - X
    dist_sq = torch.sum(X_diff * X_diff, dim=2)
    return torch.exp(-dist_sq / (2 * s**2)) / s


def _evaluate_nablaK(K, X, s):
    X_diff = X.unsqueeze(1) - X
    return -torch.einsum("kij,ik->kj", X_diff, K) / (s**2)


def _stein_gradient(K, nablaK, eta_G):
    n = K.shape[0]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    return torch.linalg.solve(K + eta_G * eye, nablaK)


def _stein_hessian_diagonal(X, eta_G, eta_H):
    n = X.shape[0]
    s = _kernel_width(X)
    K = _evaluate_kernel(X, s)
    nablaK = _evaluate_nablaK(K, X, s)
    G = _stein_gradient(K, nablaK, eta_G)

    X_diff = X.unsqueeze(1) - X
    second_term = -1.0 / (s**2) + (X_diff**2) / (s**4)
    nabla2K = torch.einsum("kij,ik->kj", second_term, K)
    nabla2K = nabla2K.to(K.dtype)
    eye = torch.eye(n, device=X.device, dtype=K.dtype)
    return -(G**2) + torch.linalg.solve(K + eta_H * eye, nabla2K)


def compute_score_order(
    data, device=None, eta_G=0.001, eta_H=0.001, estimate_variance=False
):
    target_device = _resolve_device(device)
    X = torch.as_tensor(data, dtype=torch.float32, device=target_device)
    _, d = X.shape
    active_nodes = list(range(d))
    order = []
    noise_var = np.ones(d, dtype=np.float64)

    for _ in range(d - 1):
        H_diag = _stein_hessian_diagonal(X, eta_G, eta_H)
        var_cols = torch.var(H_diag, dim=0, unbiased=False)
        leaf_pos = int(torch.argmin(var_cols).item())
        leaf_node = active_nodes[leaf_pos]
        if estimate_variance:
            curr_var = torch.var(H_diag[:, leaf_pos], unbiased=False)
            noise_var[leaf_node] = 1.0 / max(float(curr_var.item()), EPS)
        order.append(leaf_node)
        active_nodes.pop(leaf_pos)
        X = torch.hstack([X[:, :leaf_pos], X[:, leaf_pos + 1 :]])

    # last node
    last_node = active_nodes[0]
    if estimate_variance:
        H_last = _stein_hessian_diagonal(X, eta_G, eta_H)
        col = H_last[:, 0]
        col_norm = col / max(float(torch.mean(col).item()), EPS)
        curr_var = torch.var(col_norm, unbiased=False)
        noise_var[last_node] = 1.0 / max(float(curr_var.item()), EPS)
    order.append(last_node)
    order.reverse()
    return order, noise_var


def score_ordering(train_set, device, ordering_config):
    eta_G = ordering_config.get("eta_G", 0.001)
    eta_H = ordering_config.get("eta_H", 0.001)
    estimate_variance = bool(ordering_config.get("estimate_variance", False))
    target_device = _resolve_device(device)
    order, noise_var = compute_score_order(
        train_set,
        device=target_device,
        eta_G=eta_G,
        eta_H=eta_H,
        estimate_variance=estimate_variance,
    )
    pre_dag = full_DAG(order)
    data = np.asarray(train_set, dtype=np.float32)
    extras = {"noise_var": noise_var} if estimate_variance else {}
    return order, pre_dag, data, extras
