import numpy as np
from dataclasses import dataclass

from utils import full_DAG


@dataclass
class OrderingOutput:
    data: np.ndarray
    pre_dag: np.ndarray
    order: list
    extras: dict


def caps_ordering(train_set, device, ordering_config):
    from ordering.caps import finalize_caps, prepare_caps

    caps_device_cfg = ordering_config.get("device")
    if caps_device_cfg is None:
        caps_device = device
    else:
        if isinstance(caps_device_cfg, str) and caps_device_cfg.lower() == "global":
            caps_device = device
        else:
            caps_device = caps_device_cfg

    train_tensor, order, parents_score, init_dag = prepare_caps(
        train_set,
        caps_device,
        pre_pruning=ordering_config.get("pre_pruning", True),
        lambda1=ordering_config.get("lambda1", 50.0),
    )
    data = train_tensor.detach().cpu().numpy()
    extras = {"parents_score": parents_score, "finalize_caps": finalize_caps}
    return OrderingOutput(data=data, pre_dag=init_dag, order=order, extras=extras)


def diffan_ordering(train_set, device, ordering_config):
    from ordering.diffan import DiffANOrdering

    feature_matrix = train_set.astype(np.float32)
    diffan_model = DiffANOrdering(
        n_nodes=feature_matrix.shape[1],
        masking=ordering_config.get("masking", True),
        residue=ordering_config.get("residue", True),
        epochs=ordering_config.get("epochs", 3000),
        batch_size=ordering_config.get("batch_size", 1024),
        eval_batch_size=ordering_config.get("eval_batch_size"),
        learning_rate=ordering_config.get("learning_rate", 0.001),
        n_votes=ordering_config.get("n_votes", 3),
        early_stopping_wait=ordering_config.get("early_stopping_wait", 300),
    )
    order, normalized_data = diffan_model.fit(feature_matrix.astype(np.float32), apply_pruning=False)
    pre_dag = full_DAG(order)
    return OrderingOutput(data=normalized_data, pre_dag=pre_dag, order=order, extras={})


def score_ordering(train_set, device, ordering_config):
    from ordering.score import score_ordering as score_impl

    result = score_impl(train_set, device, ordering_config)
    if len(result) == 4:
        order, pre_dag, data, extras = result
    else:
        order, pre_dag, data = result
        extras = {}
    return OrderingOutput(data=data, pre_dag=pre_dag, order=order, extras=extras)


def cam_ordering_runner(train_set, device, ordering_config):
    from ordering.cam import cam_ordering as cam_impl

    order, pre_dag, data = cam_impl(train_set, device, ordering_config)
    return OrderingOutput(data=data, pre_dag=pre_dag, order=order, extras={})


def nogam_ordering_runner(train_set, device, ordering_config):
    from ordering.nogam import nogam_ordering as nogam_impl

    order, pre_dag, data = nogam_impl(train_set, device, ordering_config)
    return OrderingOutput(data=data, pre_dag=pre_dag, order=order, extras={})


def scino_ordering_runner(train_set, device, ordering_config):
    from ordering.scino import SciNOOrdering

    model = SciNOOrdering(
        n_nodes=train_set.shape[1],
        masking=ordering_config.get("masking", True),
        residue=ordering_config.get("residue", True),
        epochs=ordering_config.get("epochs", 3000),
        batch_size=ordering_config.get("batch_size", 1024),
        eval_batch_size=ordering_config.get("eval_batch_size"),
        learning_rate=ordering_config.get("learning_rate", 0.001),
        n_fourier_layers=ordering_config.get("n_fourier_layers", 1),
        norm_type=ordering_config.get("norm_type", "batch"),
        gamma=ordering_config.get("gamma", 5.0),
        n_votes=ordering_config.get("n_votes", 3),
        early_stopping_wait=ordering_config.get("early_stopping_wait", 300),
    )
    order, data = model.fit(train_set, apply_pruning=False)
    pre_dag = full_DAG(order)
    return OrderingOutput(data=data, pre_dag=pre_dag, order=order, extras={})

def cam_pruning_runner(ordering_output, pruning_config):
    from pruning.cam_pruning import cam_pruning

    cutoff = pruning_config.get("cutoff", 0.001)
    dag, _ = cam_pruning(ordering_output.pre_dag, ordering_output.data, cutoff)
    return dag

def cape_atomic_pruning_runner(ordering_output, pruning_config):
    from pruning.cape_atomic import cape_atomic_pruning

    return cape_atomic_pruning(ordering_output, pruning_config)


def cape_pruning_runner(ordering_output, pruning_config):
    from pruning.cape import cape_pruning

    return cape_pruning(ordering_output, pruning_config)


ORDERING_REGISTRY = {
    "caps": caps_ordering,
    "diffan": diffan_ordering,
    "score": score_ordering,
    "cam": cam_ordering_runner,
    "nogam": nogam_ordering_runner,
    "scino": scino_ordering_runner,
}

PRUNING_REGISTRY = {
    "cam": cam_pruning_runner,
    "cape-atomic": cape_atomic_pruning_runner,
    "cape": cape_pruning_runner,
}
