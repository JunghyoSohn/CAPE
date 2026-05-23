import warnings
import numpy as np

from utils import fullAdj2Order, full_DAG

try:  # optional, falls back to linear regression if missing
    from pygam import LinearGAM, s
    from pygam.terms import TermList
except ImportError:  # pragma: no cover
    LinearGAM = None
    TermList = None
    s = None

try:  # optional for Preliminary Neighbour Search (PNS)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
except ImportError:  # pragma: no cover
    ExtraTreesRegressor = None
    SelectFromModel = None


NEG_INF = np.finfo(np.float32).min
EPS = 1e-12


class _LinearModel:
    """Lightweight linear regression fallback with a predict API."""

    def __init__(self, coef):
        self.coef = coef

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_aug = np.column_stack([X, np.ones(X.shape[0], dtype=np.float32)])
        return X_aug.dot(self.coef)


def _compute_n_splines(n, d, n_splines, degree):
    n_splines_adj = n_splines
    if n / max(d, 1) < 3 * n_splines:
        n_splines_adj = max(int(np.ceil(n / (3 * max(n_splines, 1)))), degree + 1)
        if n_splines_adj <= degree:
            warnings.warn(
                f"n_splines must be > spline_order. found: n_splines={n_splines_adj}, spline_order={degree}. "
                f"Using {degree + 1}."
            )
            n_splines_adj = degree + 1
    return n_splines_adj


def _make_formula(d, n_splines, degree):
    if s is None:
        return None
    terms = TermList() if TermList is not None else None
    for i in range(d):
        term = s(i, n_splines=n_splines, spline_order=degree)
        if terms is None:
            terms = term
        else:
            terms += term
    return terms


def _fit_gam_model(X, y, n_splines=10, degree=3):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    if LinearGAM is None or s is None:
        X_aug = np.column_stack([X, np.ones(X.shape[0], dtype=np.float32)])
        coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        return _LinearModel(coef)

    n, d = X.shape
    n_splines_adj = _compute_n_splines(n, d, n_splines, degree)
    formula = _make_formula(d, n_splines_adj, degree)
    lambda_grid = {1: [0.1, 0.5, 1], 20: [5, 10, 20], 100: [50, 80, 100]}
    lam_keys = list(lambda_grid.keys())
    gam = LinearGAM(formula, fit_intercept=False).gridsearch(
        X, y, lam=lam_keys, progress=False, objective="GCV"
    )
    lambdas = np.squeeze([spline.get_params()["lam"] for spline in gam.terms], axis=1)
    lam = np.squeeze([lambda_grid[val] for val in lambdas])
    gam = LinearGAM(formula, fit_intercept=False).gridsearch(
        X, y, lam=lam.transpose(), progress=False, objective="GCV"
    )
    return gam


def _update_directed_paths(parent, child, directed_paths):
    directed_paths[parent, child] = 1
    child_descendants = np.argwhere(directed_paths[child, :]).ravel()
    parent_ancestors = np.argwhere(directed_paths[:, parent]).ravel()
    for p in parent_ancestors:
        for c in child_descendants:
            directed_paths[p, c] = 1


def _update_acyclicity_constraints(parent, child, score_gains, directed_paths):
    score_gains[parent, child] = NEG_INF
    score_gains[child, parent] = NEG_INF
    _update_directed_paths(parent, child, directed_paths)
    score_gains[np.transpose(directed_paths == 1, (1, 0))] = NEG_INF


def _pns_mask(X, threshold, num_neighbors):
    """Optional Preliminary Neighbour Search pruning mask."""
    d = X.shape[1]
    if ExtraTreesRegressor is None or SelectFromModel is None:
        warnings.warn("sklearn is not available; skipping PNS pruning.")
        return np.ones((d, d), dtype=np.int8)

    A = np.ones((d, d), dtype=np.int8)
    for node in range(d):
        X_copy = np.copy(X)
        X_copy[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(X_copy, X[:, node])
        selector = SelectFromModel(
            reg,
            threshold=f"{threshold}*mean",
            prefit=True,
            max_features=num_neighbors,
        )
        mask_selected = selector.get_support(indices=False).astype(np.int8)
        A[:, node] *= mask_selected

    np.fill_diagonal(A, 0)
    return A


def _initialize_score(X, directed_paths, n_splines, degree, min_gain, use_pns, pns_threshold, pns_num_neighbors):
    _, d = X.shape
    pns_mask = np.ones((d, d), dtype=np.int8)
    if use_pns:
        pns_mask = _pns_mask(X, pns_threshold, pns_num_neighbors)

    score_gains = np.zeros((d, d), dtype=np.float32)
    score_gains[np.transpose(directed_paths == 1, (1, 0))] = NEG_INF
    nodes_variance = np.var(X, axis=0, dtype=np.float64)
    nodes_variance[nodes_variance <= 0] = EPS
    init_score = -np.log(nodes_variance)

    for i in range(d):
        for j in range(d):
            if pns_mask[i, j] == 0:
                score_gains[i, j] = NEG_INF
                continue
            if score_gains[i, j] == NEG_INF:
                continue
            model = _fit_gam_model(X[:, [i]], X[:, j], n_splines=n_splines, degree=degree)
            residuals = X[:, j] - model.predict(X[:, [i]]).reshape(-1)
            var = float(np.var(residuals))
            if var <= EPS:
                var = EPS
            gain = -np.log(var) - init_score[j]
            if min_gain is not None and gain < min_gain:
                gain = NEG_INF
            score_gains[i, j] = gain
    return score_gains, init_score


def _update_score(X, A, c, score_gains, score_c, n_splines, degree, min_gain):
    d = A.shape[0]
    current_parents = np.flatnonzero(A[:, c])
    for pot_parent in range(d):
        if pot_parent == c or pot_parent in current_parents:
            continue
        if score_gains[pot_parent, c] == NEG_INF:
            continue
        predictors = np.append(current_parents, [pot_parent])
        model = _fit_gam_model(X[:, predictors], X[:, c], n_splines=n_splines, degree=degree)
        residuals = X[:, c] - model.predict(X[:, predictors]).reshape(-1)
        var = float(np.var(residuals))
        if var <= EPS:
            var = EPS
        gain = -np.log(var) - score_c
        if min_gain is not None and gain < min_gain:
            gain = NEG_INF
        score_gains[pot_parent, c] = gain


def cam_ordering_impl(
    data,
    n_splines=10,
    degree=3,
    use_pns=False,
    pns_threshold=1.0,
    pns_num_neighbors=None,
    min_gain=None,
):
    X = np.asarray(data, dtype=np.float32)
    _, d = X.shape
    A = np.zeros((d, d), dtype=np.int8)
    directed_paths = np.eye(d, dtype=np.int8)

    score_gains, score = _initialize_score(
        X,
        directed_paths,
        n_splines=n_splines,
        degree=degree,
        min_gain=min_gain,
        use_pns=use_pns,
        pns_threshold=pns_threshold,
        pns_num_neighbors=pns_num_neighbors,
    )

    while np.sum(score_gains > NEG_INF) > 0:
        parent, child = np.unravel_index(np.argmax(score_gains, axis=None), score_gains.shape)
        best_gain = score_gains[parent, child]
        if best_gain == NEG_INF:
            break
        A[parent, child] = 1
        score[child] += best_gain
        _update_acyclicity_constraints(parent, child, score_gains, directed_paths)
        _update_score(
            X,
            A,
            child,
            score_gains,
            score_c=score[child],
            n_splines=n_splines,
            degree=degree,
            min_gain=min_gain,
        )

    order = fullAdj2Order(A)
    return order


def cam_ordering(train_set, device, ordering_config):
    data = np.asarray(train_set, dtype=np.float32)
    cfg = ordering_config or {}
    n_splines = cfg.get("n_splines", 10)
    degree = cfg.get("splines_degree", cfg.get("degree", 3))
    use_pns = bool(cfg.get("pns", False))
    pns_threshold = float(cfg.get("pns_threshold", 1.0))
    pns_num_neighbors = cfg.get("pns_num_neighbors", None)
    min_gain = cfg.get("min_gain", None)

    order = cam_ordering_impl(
        data,
        n_splines=n_splines,
        degree=degree,
        use_pns=use_pns,
        pns_threshold=pns_threshold,
        pns_num_neighbors=pns_num_neighbors,
        min_gain=min_gain,
    )
    pre_dag = full_DAG(order)
    return order, pre_dag, data
