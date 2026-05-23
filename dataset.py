import os
import os.path
import copy
import importlib
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import pickle as pk
import networkx as nx

from dag_simulation import DAG, IIDSimulation
from pgmpy.utils import get_example_model


def _mlp_transform_noise(noise, width=100, weight_scale=1.5):
    """Simple 1-hidden-layer MLP transform to induce non-Gaussian noise."""
    noise = noise.reshape(-1, 1)
    w1 = np.random.uniform(-weight_scale, weight_scale, size=(1, width))
    w2 = np.random.uniform(-weight_scale, weight_scale, size=(width, 1))
    hidden = 1 / (1 + np.exp(-noise @ w1))
    transformed = hidden @ w2
    transformed = transformed.flatten()
    std = transformed.std() + 1e-8
    return transformed / std


def _sample_non_gaussian_noise(n, std, noise_type="mlp", weight_scale=1.5):
    base = np.random.normal(0.0, std, size=n)
    if noise_type == "mlp":
        return _mlp_transform_noise(base, weight_scale=weight_scale)
    if noise_type == "laplace":
        return np.random.laplace(0.0, std / np.sqrt(2), size=n)
    if noise_type == "gumbel":
        return np.random.gumbel(0.0, std * np.sqrt(6) / np.pi, size=n)
    if noise_type == "exp":
        return np.random.exponential(std, size=n)
    if noise_type == "uniform":
        return np.random.uniform(-std, std, size=n)
    return base


def _simulate_linear_non_gaussian(weighted_random_dag, num_samples, p_linear, noise_type="mlp", weight_scale=1.5):
    """LiNGAM-like generator: linear mechanisms, non-Gaussian noise."""
    d = weighted_random_dag.shape[0]
    G = nx.from_numpy_array((weighted_random_dag != 0).astype(float), create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Weighted DAG must be acyclic.")
    order = list(nx.topological_sort(G))
    X = np.zeros((num_samples, d), dtype=np.float32)
    noise_std = np.random.uniform(0.5, 1.0, size=d)
    is_linear = np.random.rand(d) < np.clip(p_linear, 0.0, 1.0)
    for j in order:
        parents = np.nonzero(weighted_random_dag[:, j])[0]
        noise = _sample_non_gaussian_noise(num_samples, noise_std[j], noise_type=noise_type, weight_scale=weight_scale)
        if len(parents) == 0:
            X[:, j] = noise
            continue
        parent_sum = X[:, parents] @ weighted_random_dag[parents, j]
        if is_linear[j]:
            X[:, j] = parent_sum + noise
        else:
            X[:, j] = np.tanh(parent_sum) + noise
    return X, (weighted_random_dag != 0).astype(int)


def _ensure_numpy_pickle_compat() -> None:
    """
    Newer NumPy (>=2.0) pickles arrays that reference the private
    ``numpy._core`` namespace. Older NumPy releases (<2.0) only ship
    ``numpy.core``, so unpickling fails with ModuleNotFoundError.

    When ``numpy._core`` is missing we alias it (and a few common submodules)
    to their ``numpy.core`` equivalents so that pickled arrays saved with
    NumPy 2.x remain readable on legacy versions.
    """
    try:
        importlib.import_module("numpy._core.numeric")
        return
    except ModuleNotFoundError:
        pass

    try:
        core_module = importlib.import_module("numpy.core")
    except ModuleNotFoundError:
        return

    sys.modules.setdefault("numpy._core", core_module)
    for sub_name in ("numeric", "multiarray", "_multiarray_umath", "umath"):
        try:
            submodule = importlib.import_module(f"numpy.core.{sub_name}")
        except ModuleNotFoundError:
            submodule = getattr(core_module, sub_name, None)
        if submodule is not None:
            sys.modules.setdefault(f"numpy._core.{sub_name}", submodule)


_ensure_numpy_pickle_compat()


def load_data(dataset, n=7466, norm=True, seed=42, simulation_seed=42, num_nodes=10, num_samples=1000, method='nonlinear', linear_sem_type='gauss', nonlinear_sem_type='gp', linear_rate=0.5, runs=-1, scenario=None):
    
    np.set_printoptions(suppress=True)
    
    root = 'Datasets'
    np.random.seed(seed)
    scenario = scenario or {}
    scenario_name = str(scenario.get('name', 'vanilla')).lower()
    scenario_params = scenario.get('params', {}) or {}
    
    if dataset in ['sachs', 'sachs_nocycle']:
        import cdt

        data_df, solution = cdt.data.load_dataset('sachs')
        node_names = list(data_df.columns)
        true_causal_matrix = nx.to_numpy_array(solution).T  # align with original code

        if dataset == "sachs_nocycle":
            remove_vars = ["praf", "plcg", "PIP2"]
            remove_indices = [data_df.columns.get_loc(var) for var in remove_vars]
            true_causal_matrix = np.delete(true_causal_matrix, remove_indices, axis=0)
            true_causal_matrix = np.delete(true_causal_matrix, remove_indices, axis=1)
            node_names = [name for i, name in enumerate(node_names) if i not in remove_indices]
            data_df = data_df.drop(columns=remove_vars)

        data = data_df.values
        GT_DAG = true_causal_matrix.astype(int)
        ori_data_ls = None

        if norm:
            mu = np.mean(data, axis=0)
            sigma = np.std(data, axis=0)
            sigma[sigma == 0] = 1
            data = (data - mu) / sigma
        np.random.shuffle(data)

    elif dataset in ['syntren']:
        GT_DAG = np.load(os.path.join(root, dataset, "DAG{}.npy".format(runs)))
        data = np.load(os.path.join(root, dataset, "data{}.npy".format(runs)))

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        if norm:
            data = (data - mu) / sigma
        
        np.random.shuffle(data)  # return None, inplace
        ori_data_ls = None

    elif dataset in ['ecoli70_nl', 'magic-irri_nl', 'magic-niab_nl', 'arth150_nl'] or dataset.startswith('bnlearn_'):
        if dataset.startswith('bnlearn_'):
            dataset_name = dataset.split('_', 1)[1]
        else:
            dataset_name = dataset

        dataset_dir = os.path.join(root, 'bnlearn', dataset_name)
        data_path = os.path.join(dataset_dir, "X.npy")
        dag_path = os.path.join(dataset_dir, "true_causal_matrix.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"BNLearn data not found at {data_path}")
        if not os.path.exists(dag_path):
            raise FileNotFoundError(f"BNLearn DAG not found at {dag_path}")

        data = np.load(data_path)
        GT_DAG = np.load(dag_path)

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        sigma[sigma == 0] = 1
        if norm:
            data = (data - mu) / sigma
        ori_data_ls = None

    elif dataset in ['ecoli70', 'magic-irri', 'magic-niab', 'arth150']:
        dataset_name = dataset

        try:
            model = get_example_model(dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset_name} via pgmpy.get_example_model: {e}")

        if not hasattr(model, "simulate"):
            raise RuntimeError(f"Loaded model for {dataset_name} does not support simulate()")

        # Use positional args for broader pgmpy version compatibility; fallback to size kw.
        try:
            df = model.simulate(num_samples, seed=seed)
        except TypeError:
            df = model.simulate(size=num_samples, seed=seed)

        top_order = list(nx.topological_sort(model))
        data = df[top_order].values

        idx_map = {node: i for i, node in enumerate(top_order)}
        GT_DAG = np.zeros((len(top_order), len(top_order)), dtype=int)
        for parent, child in model.edges():
            GT_DAG[idx_map[parent], idx_map[child]] = 1

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        sigma[sigma == 0] = 1
        if norm:
            data = (data - mu) / sigma
        ori_data_ls = None

    elif dataset.startswith('physics'):
        physics_root = os.path.join(root, 'physics_generation')
        truth_root = os.path.join(physics_root, 'physics_truth')

        if dataset == 'physics':
            data_filename = 'physics_7nodes_5000.csv'
            truth_filename = 'physics_7node_truth.csv'
        else:
            data_filename = dataset + ('.csv' if not dataset.endswith('.csv') else '')
            tokens = dataset.split('_')
            if len(tokens) >= 2:
                nodes_token = tokens[1]
                if nodes_token.endswith('s'):
                    nodes_token = nodes_token[:-1]
                truth_filename = f"{tokens[0]}_{nodes_token}_truth.csv"
            else:
                truth_filename = 'physics_7node_truth.csv'

        data_path = os.path.join(physics_root, data_filename)
        if not os.path.exists(data_path):
            # default back to provided dataset if custom file missing
            data_path = os.path.join(physics_root, 'physics_7nodes_5000.csv')
        truth_path = os.path.join(truth_root, truth_filename)
        if not os.path.exists(truth_path):
            truth_path = os.path.join(truth_root, 'physics_7node_truth.csv')

        data_df = pd.read_csv(data_path)
        data = data_df.values
        GT_DAG = pd.read_csv(truth_path, index_col=0).values

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        sigma[sigma == 0] = 1
        if norm:
            data = (data - mu) / sigma
        ori_data_ls = None

    elif dataset.startswith('Syn'):
        gen_graph = dataset[3:5]
        edge_sparsity = int(dataset[-1])
        # print('\033[31m' + f'Syn{gen_graph}{edge_sparsity}(d={num_nodes}, n={num_samples}, linear_sem_type={linear_sem_type}, linear_rate={linear_rate}),' + '\033[0m', end=' ')
        if gen_graph=='ER':
            weighted_random_dag = DAG.erdos_renyi(n_nodes=num_nodes, n_edges=edge_sparsity*num_nodes, seed=simulation_seed)
            # print('weighted_random_dag:\n', weighted_random_dag)
        elif gen_graph=='SF':
            weighted_random_dag = DAG.scale_free(n_nodes=num_nodes, n_edges=edge_sparsity*num_nodes, seed=simulation_seed)
            # print('weighted_random_dag:\n', weighted_random_dag)

        low_weight_scale = 0.1
        high_weight_scale = 1.0
        # noise_scale = np.random.uniform(0.4, 0.8, (num_nodes))
        noise_scale = 1.0 

        # method: str, (linear or nonlinear), default='linear'
        #     Distribution for standard trainning dataset.
        # sem_type: str
        #     gauss, exp, gumbel, uniform, logistic (linear); 
        #     mlp, mim, gp, gp-add, quadratic (nonlinear).

        weighted_random_dag = weighted_random_dag * np.random.uniform(low=low_weight_scale, high=high_weight_scale, size=(num_nodes, num_nodes)) * np.random.choice([-1, 1], size=(num_nodes, num_nodes))
        local_method = method
        local_linear_rate = linear_rate
        local_linear_sem_type = linear_sem_type
        if scenario_name == 'lingam':
            # LiNGAM-like: non-Gaussian noise with controllable linear share
            p_linear = float(np.clip(scenario_params.get('p_linear', 1.0), 0.0, 1.0))
            noise_type = scenario_params.get('noise_type') or scenario_params.get('linear_sem_type') or 'mlp'
            weight_scale = float(scenario_params.get('noise_weight_scale', 1.5))
            data, GT_DAG = _simulate_linear_non_gaussian(
                weighted_random_dag,
                num_samples,
                p_linear=p_linear,
                noise_type=str(noise_type),
                weight_scale=weight_scale,
            )
        else:
            dataset = IIDSimulation(
                W=weighted_random_dag,
                n=num_samples,
                method=local_method,
                linear_sem_type=local_linear_sem_type,
                nonlinear_sem_type=nonlinear_sem_type,
                noise_scale=noise_scale,
                linear_rate=local_linear_rate,
            )
            GT_DAG, data = dataset.B, dataset.X

            if scenario_name not in ('vanilla', 'lingam'):
                data, GT_DAG = _apply_synthetic_scenario(
                    data,
                    weighted_random_dag,
                    GT_DAG,
                    scenario_name,
                    scenario_params,
                    num_samples,
                    local_method,
                    linear_sem_type,
                    nonlinear_sem_type,
                    noise_scale,
                    local_linear_rate,
                )

        if norm:
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            data = (data-mu) / sigma
        ori_data_ls = None

    else:
        raise Exception('Dataset not recognized.')

    return data, GT_DAG, ori_data_ls

# if __name__=='__main__':
#     data, GT_DAG, ori_data_ls = load_data('sachs', n=853, norm=False)
#     print(data.shape, (len(ori_data_ls), len(ori_data_ls[0]), len(ori_data_ls[0][0])), '\n', GT_DAG)
#     data, GT_DAG, ori_data_ls = load_data('SynER1', num_nodes=10, num_samples=1000)
#     print(data.shape, '\n', GT_DAG)


def _apply_synthetic_scenario(data, weighted_random_dag, GT_DAG, scenario_name, params, num_samples,
                              method, linear_sem_type, nonlinear_sem_type, noise_scale, linear_rate):
    scenario_name = scenario_name.lower()
    if scenario_name == 'confounded':
        rho = float(params.get('rho', 0.3))
        data = _simulate_confounded(weighted_random_dag, rho, num_samples, method, linear_sem_type, nonlinear_sem_type, noise_scale, linear_rate)
    elif scenario_name == 'measure_err':
        gamma = float(params.get('gamma', 0.5))
        data = _apply_measurement_error(data, gamma)
    elif scenario_name == 'noniid':
        alpha = _resolve_ar_alpha(params)
        data = _apply_autoregressive(data, alpha)
    elif scenario_name == 'unfaithful':
        prob = float(params.get('p_unfaithful', 0.3))
        data = _apply_unfaithful_distribution(data, GT_DAG, prob)
    elif scenario_name == 'pnl':
        exponent = float(params.get('exponent', 3.0))
        data = _apply_pnl(data, exponent)
    return data, GT_DAG


def _simulate_confounded(weighted_random_dag, rho, num_samples, method, linear_sem_type,
                         nonlinear_sem_type, noise_scale, linear_rate):
    n = weighted_random_dag.shape[0]
    W_ext = np.zeros((2 * n, 2 * n))
    W_ext[n:, n:] = weighted_random_dag
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < rho:
                conf = np.random.randint(0, n)
                w_i = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
                w_j = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
                W_ext[conf, n + i] += w_i
                W_ext[conf, n + j] += w_j
    sim = IIDSimulation(
        W=W_ext,
        n=num_samples,
        method=method,
        linear_sem_type=linear_sem_type,
        nonlinear_sem_type=nonlinear_sem_type,
        noise_scale=noise_scale,
        linear_rate=linear_rate,
    )
    data_ext = sim.X
    return data_ext[:, n:]


def _apply_measurement_error(data, gamma):
    data = np.asarray(data, dtype=np.float32)
    std = data.std(axis=0, keepdims=True) + 1e-6
    noise = np.random.normal(0, np.sqrt(max(gamma, 0.0)) * std, size=data.shape)
    return data + noise


def _resolve_ar_alpha(params):
    if "alpha" in params:
        try:
            return float(params["alpha"])
        except (TypeError, ValueError):
            pass
    coef_range = params.get('coef_range', None)
    if coef_range is not None:
        low = float(coef_range[0]) if isinstance(coef_range, (list, tuple)) else float(coef_range)
        high = float(coef_range[1]) if isinstance(coef_range, (list, tuple)) else float(coef_range)
        return np.random.uniform(low, high)
    return 0.5


def _apply_autoregressive(data, alpha):
    data = np.asarray(data, dtype=np.float32).copy()
    n, d = data.shape
    for t in range(1, n):
        data[t] += alpha * data[t - 1]
    return data


def _apply_unfaithful_distribution(data, GT_DAG, prob):
    """Induce path-canceling on moralized colliders to violate faithfulness."""
    data = np.asarray(data, dtype=np.float32).copy()
    d = GT_DAG.shape[0]
    for child in range(d):
        parents = np.where(GT_DAG[:, child] == 1)[0]
        if len(parents) < 2:
            continue
        for p1, p2 in combinations(parents, 2):
            # consider moralized collider (parents connected by an edge)
            if GT_DAG[p1, p2] + GT_DAG[p2, p1] != 1:
                continue
            if np.random.rand() < prob:
                coeff = np.random.uniform(0.5, 1.5)
                # subtract weighted parent signal to attenuate/cancel effect
                data[:, child] -= coeff * data[:, p2]
    return data


def _apply_pnl(data, exponent):
    data = np.asarray(data, dtype=np.float32)
    return np.sign(data) * (np.abs(data) ** exponent)
