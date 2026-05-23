from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .cape_atomic import (
    TabPFNCAPEAtomicPruner,
    AutoGluonCAPEAtomicPruner,
    _normalize_predictor_name,
)


class HypergraphMixin:
    """
    Binary group-testing variant of CAPE that uses the Adaptive Structural MDL gate.
    The algorithm recursively bisects the active parent set, prunes whole halves
    if their evidence gain is below the adaptive threshold, and drills down only
    on promising groups.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = getattr(self, "config", {}) or {}
        self.min_group_size = max(1, int(cfg.get("min_group_size", 2)))
        strategy = str(cfg.get("affinity_strategy", "correlation")).lower()
        allowed = {"correlation", "gradient_similarity", "random"}
        if strategy not in allowed:
            raise ValueError(f"affinity_strategy must be one of {allowed}")
        self.affinity_strategy = strategy
        self._rng = np.random.default_rng(self.seed if hasattr(self, "seed") else None)

    def run(self):
        if self.n_samples == 0:
            return self.pre_dag.copy()
        dag = self.pre_dag.copy()
        initial_parent_counts = [np.flatnonzero(dag[:, j]).size for j in range(self.d)]
        self._maybe_print_thresholds(initial_parent_counts)
        for node in self.order:
            active_parents = [int(idx) for idx in np.flatnonzero(dag[:, node])]
            if not active_parents:
                continue
            total_candidates = max(initial_parent_counts[node], len(active_parents))
            self._recursive_group_pruning(
                child=node,
                current_parents=active_parents,
                group=list(active_parents),
                dag=dag,
                total_candidates=total_candidates,
            )
        return dag

    # ------------------------------------------------------------------ #
    # Recursive binary group testing
    # ------------------------------------------------------------------ #
    def _recursive_group_pruning(
        self,
        child: int,
        current_parents: List[int],
        group: List[int],
        dag: np.ndarray,
        total_candidates: int,
    ):
        """Recursively test a candidate group using binary splitting."""
        if len(group) == 0:
            return
        if len(group) <= self.min_group_size:
            # Too small to bother with group tests; fall back to per-parent checks.
            for target in list(group):
                if target not in current_parents:
                    continue
                context = tuple(sorted(current_parents))
                context_minus = tuple(p for p in current_parents if p != target)
                gain = self._delta(child, context, context_minus)
                tau = self._adaptive_tau(total_candidates, len(context_minus))
                if gain <= tau:
                    dag[target, child] = 0
                    current_parents.remove(target)
            return

        # Sort by affinity to the child (descending), then split in half
        sorted_candidates = self._sort_by_affinity(child, group)
        mid = len(sorted_candidates) // 2
        halves = [sorted_candidates[:mid], sorted_candidates[mid:]]

        for half in halves:
            if not half:
                continue
            # Evidence loss when removing the whole group
            context = tuple(sorted(current_parents))
            remainder = tuple(sorted(p for p in current_parents if p not in half))
            gain = self._delta(child, context, remainder)
            tau = self._adaptive_tau(total_candidates, len(remainder))

            if gain <= tau:
                # Prune entire group
                for parent in half:
                    if parent in current_parents:
                        dag[parent, child] = 0
                        current_parents.remove(parent)
            else:
                # Keep and drill down
                preserved = [p for p in half if p in current_parents]
                if preserved:
                    self._recursive_group_pruning(
                        child=child,
                        current_parents=current_parents,
                        group=preserved,
                        dag=dag,
                        total_candidates=total_candidates,
                    )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _adaptive_tau(self, total_candidates: int, remaining: int) -> float:
        """Adaptive Structural MDL gate."""
        if self.n_samples <= 0:
            return np.inf
        identity_term = np.log(max(total_candidates - remaining + 1, 1))
        sparsity_term = np.log(max(remaining, 1))
        log_n = np.log(max(self.n_samples, 1))
        log_d_sq = np.log(max(self.d, 1) ** 2)
        structural_term = self.omega_eta * (log_n * log_d_sq)

        return (identity_term + sparsity_term + structural_term) / float(self.n_samples)

    def _sort_by_affinity(self, child: int, candidates: Sequence[int]) -> List[int]:
        """Rank candidates by absolute correlation with the child (descending)."""
        if len(candidates) <= 1:
            return list(candidates)
        if self.affinity_strategy == "random":
            shuffled = list(candidates)
            self._rng.shuffle(shuffled)
            return shuffled
        child_vals = self.data[:, child].astype(np.float64, copy=False)
        scores = []
        child_std = float(np.std(child_vals))
        for pid in candidates:
            parent_vals = self.data[:, pid].astype(np.float64, copy=False)
            parent_std = float(np.std(parent_vals))
            if child_std < 1e-12 or parent_std < 1e-12:
                metric = 0.0
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    cov = float(np.mean((child_vals - child_vals.mean()) * (parent_vals - parent_vals.mean())))
                if self.affinity_strategy == "gradient_similarity":
                    var_parent = float(np.var(parent_vals))
                    if not np.isfinite(var_parent) or var_parent < 1e-12:
                        metric = 0.0
                    else:
                        metric = abs(cov / var_parent)  # magnitude of linear slope
                else:  # correlation
                    corr = cov / (child_std * parent_std)
                    metric = abs(corr) if np.isfinite(corr) else 0.0
            scores.append((metric, int(pid)))
        scores.sort(key=lambda t: t[0], reverse=True)
        return [pid for _, pid in scores]


class HypergraphTabPFNCAPEPruner(HypergraphMixin, TabPFNCAPEAtomicPruner):
    """Hypergraph pruning with TabPFN backend."""


class HypergraphAutoGluonCAPEPruner(HypergraphMixin, AutoGluonCAPEAtomicPruner):
    """Hypergraph pruning with AutoGluon backend."""


def cape_pruning(ordering_output, config=None):
    """Convenience entry point mirroring the other pruning APIs."""
    cfg = config or {}
    predictor = _normalize_predictor_name(cfg.get("predictor", "tabpfn"))
    if predictor == "tabpfn":
        pruner_cls = HypergraphTabPFNCAPEPruner
    else:
        pruner_cls = HypergraphAutoGluonCAPEPruner
    pruner = pruner_cls(ordering_output, cfg)
    return pruner.run()
