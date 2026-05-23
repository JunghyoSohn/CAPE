import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
import pandas as pd
import copy

try:  # pragma: no cover
    from autogluon.tabular import TabularPredictor
except ImportError:  # pragma: no cover
    TabularPredictor = None

TABPFN_ROOT = Path(__file__).resolve().parent / "TabPFN"
TABPFN_REPO_PRESENT = TABPFN_ROOT.exists()
tabpfn_src = TABPFN_ROOT / "src"
if TABPFN_REPO_PRESENT and tabpfn_src.exists() and str(tabpfn_src) not in sys.path:
    sys.path.append(str(tabpfn_src))
DEFAULT_TABPFN_CKPT = TABPFN_ROOT / "checkpoints" / "tabpfn-v2.5-regressor-v2.5_default.ckpt"

try:  # pragma: no cover
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
except ImportError:  # pragma: no cover
    TabPFNRegressor = None
    ModelVersion = None


def _normalize_predictor_name(raw: str) -> str:
    key = str(raw).strip().lower()
    alias_map = {
        "xgb": "xgb",
        "xgboost": "xgb",
        "gbm": "gbm",
        "lightgbm": "gbm",
        "cat": "cat",
        "catboost": "cat",
        "rf": "rf",
        "randomforest": "rf",
        "random_forest": "rf",
        "mitra": "mitra",
        "tabpfn": "tabpfn",
    }
    return alias_map.get(key, key)


class BaseCAPEAtomicPruner:
    def __init__(self, ordering_output, config):
        config = config or {}
        self.config = config
        data = np.asarray(ordering_output.data)
        if data.ndim != 2:
            raise ValueError("Ordering output data must be 2D array-like")
        self.data = data.astype(np.float32, copy=False)
        if np.isnan(self.data).any():
            col_means = np.nanmean(self.data, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            inds = np.where(np.isnan(self.data))
            self.data[inds] = np.take(col_means, inds[1])
        self.order = list(ordering_output.order)
        self.pre_dag = np.array(ordering_output.pre_dag, dtype=np.int8)
        self.n_samples, self.d = self.data.shape
        self.feature_columns = [f"f{i}" for i in range(self.d)]
        self.omega_n_coef = float(config.get("omega_n_coef", 1.0))
        self.omega_d_coef = float(config.get("omega_d_coef", 1.0))
        self.omega_bias = float(config.get("omega_bias", 0.0))
        self.omega_eta = float(config.get("omega_eta", 1.0))
        self.mdl_gate = True
        self.seed = int(config.get("seed", 0))
        self.calibration = self._resolve_calibration(config.get("calibration", "none"))
        desired_folds = int(config.get("folds", 1))
        if desired_folds < 1:
            raise ValueError("'folds' must be >= 1")
        self.folds = self._make_folds(desired_folds)
        self.min_variance = 1e-6
        self.log_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
        self._thresholds_printed = False

    def run(self):
        if self.n_samples == 0:
            return self.pre_dag.copy()
        dag = self.pre_dag.copy()
        initial_parent_counts = [np.flatnonzero(self.pre_dag[:, j]).size for j in range(self.d)]
        self._maybe_print_thresholds(initial_parent_counts)
        for node in self.order:
            current_parents = [int(idx) for idx in np.flatnonzero(dag[:, node])]
            if not current_parents:
                continue
            total_candidates = max(initial_parent_counts[node], len(current_parents))
            for parent in list(current_parents):
                if parent not in current_parents:
                    continue
                context = tuple(sorted(current_parents))
                context_minus = tuple(sorted(p for p in current_parents if p != parent))
                gain = self._delta(node, context, context_minus)
                tau = self._tau(total_candidates, len(context_minus))
                if gain <= tau:
                    dag[parent, node] = 0
                    current_parents.remove(parent)
        return dag

    def _ensure_tabpfn_ready(self):
        if not TABPFN_REPO_PRESENT:
            raise ImportError(
                "TabPFN repository not found inside pruning/TabPFN. Please clone TabPFN v2.5 inside pruning/TabPFN."
            )
        if TabPFNRegressor is None or ModelVersion is None:
            raise ImportError(
                "TabPFN is not installed. Install it via `pip install -e pruning/TabPFN` or add it to PYTHONPATH."
            )

    def _configure_autogluon(self, config, predictor_key: str):
        predictor = predictor_key
        if predictor == "tabpfnv2":
            raise ValueError(
                "Predictor 'TABPFNV2' is no longer supported through the AutoGluon backend. "
                "Please remove it from the configuration or switch to another predictor."
            )
        model_map = {
            "xgb": {"XGB": {
                "min_child_weight": 10,
                "learning_rate": 0.05,
                "n_estimators": 500,
            }},
            "gbm": {"GBM": {}},
            "cat": {"CAT": {}},
            "rf": {"RF": {}},
            "mitra": {"MITRA": {}},
        }
        if predictor not in model_map:
            raise ValueError(f"Unsupported AutoGluon predictor '{predictor}'")
        model_spec = copy.deepcopy(model_map[predictor])
        if predictor == "mitra":
            mitra_hparams = model_spec["MITRA"]
            mitra_hparams.setdefault(
                "fine_tune", bool(config.get("autogluon_mitra_fine_tune", False))
            )
        user_hparams = config.get("autogluon_hyperparameters")
        self.autogluon_hparams = copy.deepcopy(user_hparams) if user_hparams else model_spec
        foundation_like = predictor in {"mitra"}
        default_presets = config.get("autogluon_presets")
        if default_presets is None:
            default_presets = None if foundation_like else "medium_quality_faster_train"
        self.autogluon_presets = default_presets
        self.autogluon_time_limit = config.get("autogluon_time_limit")
        self.autogluon_eval_metric = config.get("autogluon_eval_metric", "root_mean_squared_error")
        self.autogluon_problem_type = "regression"
        self.autogluon_label = "__cape_atomic_target__"
        num_gpus = config.get("autogluon_num_gpus")
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.autogluon_num_gpus = int(num_gpus)

    def _tau(self, total_candidates: int, k: int) -> float:
        if self.n_samples <= 0:
            return np.inf
        identity_term = np.log(max(total_candidates - k + 1, 1))
        sparsity_term = np.log(max(k, 1))
        log_n = np.log(max(self.n_samples, 1))
        log_d_sq = np.log(max(self.d, 1) ** 2)
        structural_term = self.omega_eta * (log_n * log_d_sq)
        return (identity_term + sparsity_term + structural_term) / float(self.n_samples)

    def _delta(self, child: int, context: Tuple[int, ...], context_minus: Tuple[int, ...]) -> float:
        logs_with = self._log_likelihood(child, context)
        logs_without = self._log_likelihood(child, context_minus)
        return float(np.mean(logs_with - logs_without))

    def _log_likelihood(self, child: int, parents: Tuple[int, ...]) -> np.ndarray:
        key = (child, parents)
        if key not in self.log_cache:
            if len(parents) == 0:
                y = self.data[:, child].astype(np.float32, copy=False)
                self.log_cache[key] = self._baseline_from_targets(y)
            else:
                self.log_cache[key] = self._evaluate_context(child, parents)
        return self.log_cache[key]

    def _evaluate_context(self, child: int, parents: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def _cross_fit_tabpfn(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        logs = np.zeros(self.n_samples, dtype=np.float64)
        for train_idx, test_idx in self.folds:
            X_train = X[train_idx]
            if self._all_features_constant(X_train):
                logs[test_idx] = self._baseline_fold_logs(y, train_idx, test_idx)
                continue
            model = self._make_tabpfn_model()
            try:
                model.fit(X_train, y[train_idx])
            except ValueError as exc:
                if "All features are constant" in str(exc):
                    logs[test_idx] = self._baseline_fold_logs(y, train_idx, test_idx)
                    continue
                raise
            output = model.predict(X[test_idx], output_type="full")
            logits = output["logits"]  # torch.Tensor
            criterion = output["criterion"]
            y_tensor = torch.tensor(
                y[test_idx], device=logits.device, dtype=logits.dtype
            )
            bucket_log_probs = criterion.compute_scaled_log_probs(logits)
            target_idx = criterion.map_to_bucket_idx(y_tensor.clone()).to(torch.long)
            sample_logs = (
                bucket_log_probs.gather(-1, target_idx.unsqueeze(-1))
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            logs[test_idx] = sample_logs
        return logs

    def _cross_fit_autogluon(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        logs = np.zeros(self.n_samples, dtype=np.float64)
        for train_idx, test_idx in self.folds:
            X_train = X[train_idx]
            if self._all_features_constant(X_train):
                logs[test_idx] = self._baseline_fold_logs(y, train_idx, test_idx)
                continue
            predictor = self._train_autogluon_model(X_train, y[train_idx])
            train_pred = predictor.predict(self._autogluon_dataframe(X_train)).to_numpy()
            calibrator = self._fit_calibrator(train_pred, y[train_idx])
            if calibrator is not None:
                train_pred = np.asarray(calibrator(train_pred))
            residuals = y[train_idx] - train_pred
            var = float(np.var(residuals))
            if not np.isfinite(var) or var < self.min_variance:
                var = self.min_variance
            test_df = self._autogluon_dataframe(X[test_idx])
            preds = predictor.predict(test_df).to_numpy()
            if calibrator is not None:
                preds = np.asarray(calibrator(preds))
            diff = y[test_idx] - preds
            logs[test_idx] = -0.5 * (np.log(2.0 * np.pi * var) + (diff**2) / var)
        return logs

    def _train_autogluon_model(self, X_train: np.ndarray, y_train: np.ndarray):
        train_df = self._autogluon_dataframe(X_train, y_train)
        predictor = TabularPredictor(
            label=self.autogluon_label,
            problem_type=self.autogluon_problem_type,
            eval_metric=self.autogluon_eval_metric,
            verbosity=0,
        )
        fit_kwargs = dict(
            train_data=train_df,
            hyperparameters=copy.deepcopy(self.autogluon_hparams),
            time_limit=self.autogluon_time_limit,
            ag_args_fit=self._autogluon_fit_args(),
            verbosity=0,
        )
        if self.autogluon_presets:
            fit_kwargs["presets"] = self.autogluon_presets
        predictor.fit(**fit_kwargs)
        return predictor

    def _autogluon_fit_args(self):
        if self.autogluon_num_gpus and self.autogluon_num_gpus > 0:
            return {"num_gpus": self.autogluon_num_gpus}
        return {}

    def _feature_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.feature_columns[: X.shape[1]])

    def _autogluon_dataframe(self, X: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
        df = self._feature_dataframe(X)
        if y is not None:
            df[self.autogluon_label] = y
        return df

    def _features(self, parents: Tuple[int, ...]) -> np.ndarray:
        if len(parents) == 0:
            return np.ones((self.n_samples, 1), dtype=np.float32)
        return self.data[:, parents].astype(np.float32, copy=False)

    def _make_folds(self, desired_folds: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(self.n_samples)
        if self.n_samples < 2 or desired_folds == 1:
            return [(idx, idx)]
        n_splits = max(2, min(desired_folds, self.n_samples))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        return list(kf.split(idx))

    def _all_features_constant(self, X: np.ndarray) -> bool:
        if X.size == 0:
            return True
        reference = X[0:1]
        return bool(np.allclose(X, reference, atol=1e-12))

    def _baseline_from_targets(self, y: np.ndarray) -> np.ndarray:
        logs = np.zeros(self.n_samples, dtype=np.float64)
        for train_idx, test_idx in self.folds:
            logs[test_idx] = self._baseline_fold_logs(y, train_idx, test_idx)
        return logs

    def _baseline_fold_logs(
        self, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> np.ndarray:
        y_train = y[train_idx]
        mean = float(np.mean(y_train))
        var = float(np.var(y_train))
        if not np.isfinite(var) or var < self.min_variance:
            var = self.min_variance
        diff = y[test_idx] - mean
        return -0.5 * (np.log(2.0 * np.pi * var) + (diff**2) / var)

    def _maybe_print_thresholds(self, parent_counts):
        if self._thresholds_printed:
            return
        self._thresholds_printed = True
        positive_counts = sorted({cnt for cnt in parent_counts if cnt > 0})
        tau_values = []
        for total in positive_counts:
            for parents_after in range(total):
                tau = self._tau(total, parents_after)
                tau_values.append(tau)
        if tau_values:
            avg_tau = float(np.mean(tau_values))
            combos = len(tau_values)
            print(
                f"[CAPE-ATOMIC] avg tau ≈ {avg_tau:.6f} over {combos} contexts "
                f"(omega_eta={self.omega_eta}, d={self.d}, samples={self.n_samples}, gate=ln(n)*ln(d^2))"
            )
        else:
            approx_tau = self._tau(1, 0)
            print(
                f"[CAPE-ATOMIC] tau ≈ {approx_tau:.6f} (omega_eta={self.omega_eta}, d={self.d}, samples={self.n_samples}, gate=ln(n)*ln(d^2))"
            )

    def _fit_calibrator(self, preds: np.ndarray, targets: np.ndarray):
        if self.calibration == "none":
            return None
        if self.calibration == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install scikit-learn to use isotonic calibration") from exc
            calib = IsotonicRegression(out_of_bounds="clip")
            calib.fit(preds, targets)
            return calib.predict
        if self.calibration == "platt":
            calib = _PlattCalibrator().fit(preds, targets)
            return calib
        return None

    def _resolve_calibration(self, raw_setting):
        choice = str(raw_setting).lower()
        if choice not in {"none", "isotonic", "platt"}:
            raise ValueError("calibration must be one of {'none', 'isotonic', 'platt'}")
        return choice


class _PlattCalibrator:
    def __init__(self):
        self.A = 0.0
        self.B = 0.0
        self.y_min = 0.0
        self.y_span = 1.0

    def fit(self, preds: np.ndarray, targets: np.ndarray):
        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        self.y_min = float(np.min(targets))
        self.y_span = float(np.max(targets) - self.y_min)
        if self.y_span < 1e-8:
            self.y_span = 1.0
        y_norm = (targets - self.y_min) / self.y_span
        eps = 1e-6
        y_norm = np.clip(y_norm, eps, 1 - eps)
        logit = np.log(y_norm / (1 - y_norm))
        A, B = np.polyfit(preds, logit, 1)
        self.A = float(A)
        self.B = float(B)
        return self

    def __call__(self, preds: np.ndarray):
        preds = np.asarray(preds, dtype=np.float64)
        logits = self.A * preds + self.B
        y_norm = 1.0 / (1.0 + np.exp(-logits))
        return y_norm * self.y_span + self.y_min


class TabPFNCAPEAtomicPruner(BaseCAPEAtomicPruner):
    def __init__(self, ordering_output, config):
        super().__init__(ordering_output, config)
        self.tabpfn_kwargs = dict(self.config.get("tabpfn_kwargs", {}))
        if "model_path" not in self.tabpfn_kwargs:
            self.tabpfn_kwargs["model_path"] = str(DEFAULT_TABPFN_CKPT)
        device_override = self.config.get("device")
        if device_override:
            device_text = str(device_override).strip()
            if device_text and device_text.lower() not in {"auto", "global"}:
                self.tabpfn_kwargs.setdefault("device", device_text)
        if self.calibration != "none":
            # print("[CAPE-ATOMIC] calibration ignored for TabPFN predictor.")
            self.calibration = "none"
        self._ensure_tabpfn_ready()

    def _evaluate_context(self, child: int, parents: Tuple[int, ...]) -> np.ndarray:
        y = self.data[:, child].astype(np.float32, copy=False)
        X = self._features(parents)
        if self._all_features_constant(X):
            return self._baseline_from_targets(y)
        return self._cross_fit_tabpfn(y, X)

    def _make_tabpfn_model(self):
        kwargs = dict(self.tabpfn_kwargs)
        if "model_path" in kwargs:
            return TabPFNRegressor.create_default_for_version(
                ModelVersion.V2_5,
                **kwargs,
            )
        return TabPFNRegressor(**kwargs)


class AutoGluonCAPEAtomicPruner(BaseCAPEAtomicPruner):
    def __init__(self, ordering_output, config):
        if TabularPredictor is None:
            raise ImportError(
                "AutoGluon is not installed. Install via `pip install autogluon.tabular` to use this predictor."
            )
        super().__init__(ordering_output, config)
        predictor_raw = self.config.get("predictor", "xgb")
        self.autogluon_key = _normalize_predictor_name(predictor_raw)
        # if self.autogluon_key == "mitra":
        #     self.calibration = "none"
        self._configure_autogluon(self.config, self.autogluon_key)

    def _evaluate_context(self, child: int, parents: Tuple[int, ...]) -> np.ndarray:
        y = self.data[:, child].astype(np.float32, copy=False)
        X = self._features(parents)
        if self._all_features_constant(X):
            return self._baseline_from_targets(y)
        return self._cross_fit_autogluon(y, X)


def cape_atomic_pruning(ordering_output, config):
    cfg = config or {}
    predictor = _normalize_predictor_name(cfg.get("predictor", "tabpfn"))
    if predictor == "tabpfn":
        pruner_cls = TabPFNCAPEAtomicPruner
    else:
        pruner_cls = AutoGluonCAPEAtomicPruner
    pruner = pruner_cls(ordering_output, cfg)
    return pruner.run()
