import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.pairwise import rbf_kernel

from utils import full_DAG


def _kernel_width(X):
    X_diff = np.expand_dims(X, axis=1) - X
    dist = np.linalg.norm(X_diff, axis=2).flatten()
    nonzero = dist[dist > 0]
    return float(np.median(nonzero)) if nonzero.size else 1.0


def _evaluate_kernel(X, s):
    return rbf_kernel(X, gamma=1 / (2 * s**2)) / s


def _evaluate_nablaK(K, X, s):
    X_diff = np.expand_dims(X, axis=1) - X
    return -np.einsum("kij,ik->kj", X_diff, K) / (s**2)


def _stein_score(X, eta_G):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    s = _kernel_width(X)
    K = _evaluate_kernel(X, s)
    nablaK = _evaluate_nablaK(K, X, s)
    return np.linalg.solve(K + eta_G * np.eye(n), nablaK)


def _create_kernel_ridge(ridge_alpha):
    return KernelRidge(kernel="rbf", gamma=ridge_alpha, alpha=ridge_alpha)


def _estimate_residuals(X, n_crossval, ridge_alpha):
    R = [
        X[:, i]
        - cross_val_predict(
            _create_kernel_ridge(ridge_alpha),
            np.hstack([X[:, 0:i], X[:, i + 1 :]]),
            X[:, i],
            cv=n_crossval,
        )
        for i in range(X.shape[1])
    ]
    return np.vstack(R).transpose()


def _mse(residuals, scores, n_crossval, ridge_alpha):
    errs = []
    for col in range(scores.shape[1]):
        preds = cross_val_predict(
            _create_kernel_ridge(ridge_alpha),
            residuals[:, col].reshape(-1, 1),
            scores[:, col],
            cv=n_crossval,
        )
        errs.append(float(np.mean((scores[:, col] - preds) ** 2)))
    return errs


def nogam_ordering_impl(data, n_crossval=5, ridge_alpha=0.01, eta_G=0.001):
    X = np.asarray(data, dtype=np.float32)
    _, d = X.shape
    active_nodes = list(range(d))
    order = []

    while len(active_nodes) > 1:
        scores = _stein_score(X[:, active_nodes], eta_G=eta_G)
        residuals = _estimate_residuals(X[:, active_nodes], n_crossval, ridge_alpha)
        errs = _mse(residuals, scores, n_crossval, ridge_alpha)
        leaf_pos = int(np.argmin(errs))
        order.append(active_nodes[leaf_pos])
        active_nodes.pop(leaf_pos)

    order.append(active_nodes[0])
    order.reverse()
    return order


def nogam_ordering(train_set, device, ordering_config):
    data = np.asarray(train_set, dtype=np.float32)
    n_crossval = ordering_config.get("n_crossval", 5)
    ridge_alpha = ordering_config.get("ridge_alpha", 0.01)
    eta_G = ordering_config.get("eta_G", 0.001)
    order = nogam_ordering_impl(
        data, n_crossval=n_crossval, ridge_alpha=ridge_alpha, eta_G=eta_G
    )
    pre_dag = full_DAG(order)
    return order, pre_dag, data
