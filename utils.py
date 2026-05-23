import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import uuid
import yaml
from torch.utils.data import DataLoader, Dataset
from collections import Counter

def _normalize_scenario_cfg(raw):
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        raw = {"name": raw, "params": {}}
    name = str(raw.get("name", "vanilla")).lower()
    params = raw.get("params") or {}
    return {"name": name, "params": params}


def is_acyclic(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return nx.is_directed_acyclic_graph(G)

def adj2order(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return list(nx.topological_sort(G))

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def np_to_csv(array, save_path):
        """
        Convert np array to .csv
        array: numpy array
            the numpy array to convert to csv
        save_path: str
            where to temporarily save the csv
        Return the path to the csv file
        """
        id = str(uuid.uuid4())
        #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
        output = os.path.join(save_path, 'tmp_' + id + '.csv')

        df = pd.DataFrame(array)
        df.to_csv(output, header=False, index=False)

        return output

def normlize_data(data):
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    data = (data-mu) / sigma
    return data

def get_parents_score(curr_H, full_H, i):
    full_H = torch.hstack([full_H[0:i], full_H[i+1:]])
    parents_score = np.abs(curr_H - full_H)
    parents_score = torch.cat([parents_score[:i], torch.tensor([0.0]), parents_score[i:]])
    return parents_score

def add_edge_with_parents_score(dag, parents_score, lambda2):
    avg_parent_score = (dag * parents_score.T).mean()
    add_edge = (parents_score.T >= lambda2 * avg_parent_score).astype(np.int32)
    idx = np.transpose(np.nonzero(add_edge))
    val = np.array([parents_score.T[i[0]][i[1]] for i in idx])
    sorted_idx = np.argsort(-val)
    idx = idx[sorted_idx]
    for i in idx:
        if dag[i[0], i[1]] == 1:
            continue
        else:
            dag[i[0], i[1]] = 1
            if is_acyclic(dag):
                continue
            else:
                dag[i[0], i[1]] = 0
    return dag

def pre_pruning_with_parents_score(init_dag, parents_score, lambda1):
    d = init_dag.shape[0]
    threshold = (np.max(parents_score, axis=1)/lambda1).reshape(d, 1)
    threshold = np.tile(threshold, (1, d))
    mask = (parents_score >= threshold).T.astype(np.int32)
    return init_dag * mask


def load_run_config(config_path):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(cfg_path, "r") as f:
        config_values = yaml.safe_load(f) or {}

    general_cfg = config_values.get("general")
    if general_cfg is None:
        raise ValueError("'general' section missing in config file")
    general_cfg["scenarios"] = _normalize_scenario_cfg(general_cfg.get("scenarios"))
    args = SimpleNamespace(**general_cfg)

    ordering_device = general_cfg.get("ordering_device")
    pruning_device = general_cfg.get("pruning_device")

    ordering_section = config_values.get("ordering")
    if ordering_section is None:
        raise ValueError("'ordering' section missing in config file")
    ordering_model = ordering_section.get("model")
    if ordering_model is None:
        raise ValueError("Ordering model must be specified under 'ordering.model'")
    ordering_device = ordering_section.get("device")
    ordering_config = None
    for key, value in ordering_section.items():
        if key.lower() == ordering_model.lower() and isinstance(value, dict):
            ordering_config = value
            break
    if ordering_config is None:
        raise ValueError(f"Config for ordering model '{ordering_model}' not found in YAML")

    pruning_section = config_values.get("pruning")
    if pruning_section is None:
        raise ValueError("'pruning' section missing in config file")
    pruning_method = pruning_section.get("method")
    if pruning_method is None:
        raise ValueError("Pruning method must be specified under 'pruning.method'")
    pruning_device = pruning_section.get("device")
    pruning_config = None
    for key, value in pruning_section.items():
        if key.lower() == pruning_method.lower() and isinstance(value, dict):
            pruning_config = value
            break
    if pruning_config is None:
        raise ValueError(f"Config for pruning method '{pruning_method}' not found in YAML")

    args.model = ordering_model
    args.pruning_method = pruning_method
    args.ordering_device = ordering_device
    args.pruning_device = pruning_device
    args.config_path = str(cfg_path)
    return args, ordering_config, pruning_config
