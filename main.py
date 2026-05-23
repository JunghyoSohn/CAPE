import argparse
import csv
import copy
import logging
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
import yaml

import dataset
from castle.metrics import MetricsDAG
import warnings
warnings.filterwarnings("ignore", message="Detecting .* CUDA device")
from cdt.metrics import SID

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
logging.getLogger("posthog").setLevel(logging.ERROR)

from model_runner import ORDERING_REGISTRY, PRUNING_REGISTRY, OrderingOutput
from utils import load_run_config, full_DAG



def blue(x):
    return "\033[94m" + x + "\033[0m"


def red(x):
    return "\033[31m" + x + "\033[0m"


torch.set_printoptions(linewidth=1000)
np.set_printoptions(linewidth=1000)


class Tee:
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee requires at least one stream")
        self.streams = streams

    def write(self, data):
        for idx, stream in enumerate(self.streams):
            if idx == 0:
                stream.write(data)
            else:
                stream.write(self.ANSI_RE.sub("", data))

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parsed = parser.parse_args()
    return load_run_config(parsed.config)


def evaluate(args, dag, GT_DAG):
    mt = MetricsDAG(dag, GT_DAG)
    sid = SID(GT_DAG, dag)
    metrics = mt.metrics
    metrics["sid"] = sid.item()
    ordered_keys = ["shd", "sid", "F1", "nnz", "precision", "recall", "fdr", "tpr", "fpr", "gscore"]
    ordered_metrics = {k: metrics[k] for k in ordered_keys if k in metrics}
    print(str(ordered_metrics))
    return metrics




def _normalize_device_value(raw):
    if raw is None:
        return None
    if isinstance(raw, torch.device):
        text = str(raw)
    else:
        text = str(raw).strip()
    if text == "" or text.lower() in {"global", "default", "inherit", "none"}:
        return None
    return text


def _select_device(preferred, fallback=None):
    candidate = _normalize_device_value(preferred)
    fallback_value = _normalize_device_value(fallback)
    choice = candidate or fallback_value
    if choice is None:
        choice = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(choice)
    except (TypeError, ValueError):
        return torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device
def train_test(args, train_set, GT_DAG, data_ls, runs, ordering_config, pruning_config):
    ordering_device = _select_device(getattr(args, "ordering_device", None), args.device)
    pruning_device = _select_device(getattr(args, "pruning_device", None), args.device)

    ordering_fn = ORDERING_REGISTRY.get(args.model.lower())
    if ordering_fn is None:
        raise ValueError(f"Ordering model '{args.model}' is not supported.")
    ordering_start = time.time()
    ordering_output = ordering_fn(train_set, ordering_device, ordering_config)
    ordering_time = time.time() - ordering_start

    pruning_fn = PRUNING_REGISTRY.get(args.pruning_method.lower())
    if pruning_fn is None:
        raise ValueError(f"Pruning method '{args.pruning_method}' is not supported.")
    pruning_config_local = copy.deepcopy(pruning_config)
    pruning_config_local.setdefault("device", str(pruning_device))
    pruning_start = time.time()
    dag = pruning_fn(ordering_output, pruning_config_local)
    pruning_time = time.time() - pruning_start

    if args.model.lower() == "caps":
        finalize_fn = ordering_output.extras.get("finalize_caps")
        if finalize_fn:
            dag = finalize_fn(
                dag,
                ordering_output.extras["parents_score"],
                lambda2=ordering_config.get("lambda2", 50.0),
                add_edge=ordering_config.get("add_edge", True),
            )

    print(f"Timing=({ordering_time:.2f}s, {pruning_time:.2f}s)", end=' ') # (ordering, pruning)
    return evaluate(args, dag, GT_DAG), ordering_output.order


def main():
    args, ordering_config, pruning_config = get_args()
    metrics_list_dict = {
        "shd": [],
        "sid": [],
        "F1": [],
        "nnz": [],
        "precision": [],
        "recall": [],
        "fdr": [],
        "tpr": [],
        "fpr": [],
        "gscore": [],
    }
    metrics_res_dict = {}
    run_records = []
    simulation_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_stats_printed = False
    real_datasets = ["physics", "sachs", "magic-niab", "magic-irri"]
    results_root = Path("results")
    result_dir = None
    log_file = None
    original_stdout = sys.stdout
    config_printed = False
    try:
        for i in range(args.runs):
            if args.dataset in real_datasets:
                train_set, GT_DAG, data_ls = dataset.load_data(
                    args.dataset,
                    norm=args.norm,
                    simulation_seed=42,
                    num_nodes=args.num_nodes,
                    num_samples=args.num_samples,
                    method=args.method,
                    linear_sem_type=args.linear_sem_type,
                    nonlinear_sem_type=args.nonlinear_sem_type,
                    linear_rate=args.linear_rate,
                    runs=i + 1,
                    scenario=getattr(args, "scenarios", None),
                )
            elif args.dataset.startswith("Syn"):
                train_set, GT_DAG, data_ls = dataset.load_data(
                    args.dataset,
                    norm=args.norm,
                    simulation_seed=simulation_seeds[i],
                    num_nodes=args.num_nodes,
                    num_samples=args.num_samples,
                    method=args.method,
                    linear_sem_type=args.linear_sem_type,
                    nonlinear_sem_type=args.nonlinear_sem_type,
                    linear_rate=args.linear_rate,
                    runs=i + 1,
                    scenario=getattr(args, "scenarios", None),
                )
            else:
                raise Exception("Dataset not recognized.")

            if not data_stats_printed:
                num_samples = train_set.shape[0]
                num_nodes = train_set.shape[1]
                edge_count = int(np.sum(GT_DAG)) if GT_DAG is not None else 0
                if result_dir is None:
                    timestamp = datetime.now().strftime("%m%d%H_%M%S")
                    scenario_cfg = getattr(args, "scenarios", None)
                    scenario_name = scenario_cfg.get("name", "none") if scenario_cfg else "none"
                    predictor_suffix = ""
                    if args.pruning_method.lower() in {"cape-atomic", "cape"}:
                        predictor = pruning_config.get("predictor")
                        if predictor:
                            predictor_suffix = f"_{str(predictor).upper()}"
                    folder_name = f"{timestamp}_{args.dataset}_{num_nodes}_{edge_count}_{num_samples}_{scenario_name}_{args.model}_{args.pruning_method}{predictor_suffix}"
                    result_dir = results_root / folder_name
                    result_dir.mkdir(parents=True, exist_ok=True)
                    log_file = open(result_dir / "output.txt", "w")
                    sys.stdout = Tee(original_stdout, log_file)
                    cfg_path = Path(args.config_path)
                    shutil.copy(cfg_path, result_dir / cfg_path.name)
                if not config_printed:
                    print(blue(f"Ordering: {args.model} {ordering_config}"))
                    print(blue(f"Pruning : {args.pruning_method} {pruning_config}"))
                    config_printed = True
                if args.dataset in real_datasets:
                    print(blue(f"Dataset: {args.dataset} [norm={args.norm}, nodes={num_nodes}, edges={edge_count}, samples={num_samples}]"))
                else:
                    scenario_cfg = getattr(args, "scenarios", None)
                    scenario_text = ""
                    if scenario_cfg:
                        scenario_text = f", scenario={scenario_cfg.get('name')}"
                    print(blue(f"Dataset: {args.dataset} [norm={args.norm}, nodes={num_nodes}, edges={edge_count}, samples={num_samples}{scenario_text}]"))
                data_stats_printed = True

            print(red(f"runs {i}: "), end=' ')

            if args.manual_seed:
                Seed = args.random_seed
            else:
                Seed = random.randint(1, 10000)
            print(f"Random Seed={Seed}, ", end='')
            random.seed(Seed)
            torch.manual_seed(Seed)
            np.random.seed(Seed)

            metrics_dict, order = train_test(
                args, train_set, GT_DAG, data_ls, runs=i, ordering_config=ordering_config, pruning_config=pruning_config
            )
            for k in metrics_list_dict:
                value = metrics_dict.get(k, 0.0)
                if np.isnan(value):
                    value = 0.0
                metrics_list_dict[k].append(value)
            run_records.append({k: metrics_list_dict[k][-1] for k in metrics_list_dict})

        print(red("Avg Results - "), end=' ')
        for k in metrics_list_dict:
            metrics_list_dict[k] = np.array(metrics_list_dict[k])
            metrics_res_dict[k] = "{}±{}".format(
                np.around(np.mean(metrics_list_dict[k]), 5),
                np.around(np.std(metrics_list_dict[k]), 5),
            )
            print(red(str(k).upper() + ":" + metrics_res_dict[k]), end='  ')
        print()

        if result_dir is not None:
            scenario_cfg = getattr(args, "scenarios", None)
            scenario_name = scenario_cfg.get("name", "none") if scenario_cfg else "none"
            csv_name = f"{args.dataset}_{num_nodes}_{edge_count}_{num_samples}_{scenario_name}_{args.model}_{args.pruning_method}.csv"
            csv_path = result_dir / csv_name
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                columns = list(metrics_list_dict.keys())
                writer.writerow(columns)
                for record in run_records:
                    writer.writerow([f"{record[k]:.2f}" for k in columns])
                writer.writerow([])
                writer.writerow(["Mean & std"])
                writer.writerow([f"{np.mean(metrics_list_dict[k]):.2f}" for k in columns])
                writer.writerow([f"{np.std(metrics_list_dict[k]):.2f}" for k in columns])
    finally:
        if log_file:
            log_file.flush()
            sys.stdout = original_stdout
            log_file.close()

if __name__ == "__main__":
    main()
