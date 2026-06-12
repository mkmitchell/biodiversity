"""MaxEnt species modeling pipeline extracted from maxent_model.ipynb."""

from __future__ import annotations

import glob
import json
import os
import re
from contextlib import contextmanager

import elapid
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rasterio
import seaborn as sns
from elapid import GeographicKFold
from joblib import Parallel, delayed
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Worker-process cache for batch runs (populated via init_worker_cache).
_BG_CACHE: dict[str, pl.DataFrame] = {}
_AOI_CACHE: gpd.GeoDataFrame | None = None

LABEL_MAP = {
    "0": "water",
    "1": "trees",
    "2": "grass",
    "3": "flooded_vegetation",
    "4": "crops",
    "5": "shrub_and_scrub",
    "6": "built",
    "7": "bare",
    "8": "snow_and_ice",
}

_RENAME_PATTERN = re.compile(r"^dw_class_(\d)_pct_(\d+)m$")

METADATA_DROP_COLS = [
    "basisofrecord",
    "coordinateuncertaintyinmeters",
    "species",
    "rand",
    "date",
    "day",
    "obs_date",
    "observation_date",
    "protocol_name",
    "scientific_name",
    "year",
    "label",
    "srad_mean_mean",
    "swe_mean_mean",
    "vp_mean_mean",
]

STATIC_PREDICTOR_DROPS = [
    "dayl_10000m",
    "dw_class_8_pct_10000m",
    "dw_class_8_pct_500m",
]


def rename_col(column_name: str) -> str:
    match = _RENAME_PATTERN.match(column_name)
    if not match:
        return column_name
    class_id, scale = match.group(1), match.group(2)
    label = LABEL_MAP.get(class_id)
    if label is None:
        return column_name
    return f"{label}_{scale}m"


def prepare_predictors(
    data: pd.DataFrame, excel_group: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build model matrix with group-specific temporal covariates."""
    df = data.copy()
    group = excel_group.strip().lower()

    if group == "birds":
        if "month" not in df.columns:
            raise ValueError("month column required for birds")
        if "season" not in df.columns:
            raise ValueError("season column required for birds")
        df["month"] = df["month"].astype(int)
        df["season"] = df["season"].astype(int)
        categorical_features = ["month", "season"]
    else:
        df = df.drop(columns=["month", "season"], errors="ignore")
        categorical_features = []

    X = df.drop(columns=METADATA_DROP_COLS, errors="ignore").copy()
    X = X.fillna(0)
    X.columns = [col.replace("_mean_mean", "_10000m") for col in X.columns]
    X = X.drop(columns=STATIC_PREDICTOR_DROPS, errors="ignore")
    return X, categorical_features


def load_model_config(outputdir: str, sp: str) -> dict:
    """Load saved model config, falling back to predictors list for legacy models."""
    config_path = os.path.join(outputdir, f"model_config_{sp}.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            return json.load(handle)

    predictors_path = os.path.join(outputdir, f"predictors_{sp}.txt")
    with open(predictors_path, encoding="utf-8") as handle:
        predictor_columns = [line.strip() for line in handle if line.strip()]
    return {
        "scientific_name": sp,
        "excel_group": "birds",
        "spgroup": "avian",
        "categorical_features": [],
        "predictor_columns": predictor_columns,
    }


def init_worker_cache(parambasedir: str, aoi_filename: str) -> None:
    """Load shared AOI/background data once per worker process."""
    global _AOI_CACHE
    _AOI_CACHE = gpd.read_parquet(os.path.join(parambasedir, aoi_filename))
    _BG_CACHE.clear()


def _get_aoi(parambasedir: str, aoi_filename: str) -> gpd.GeoDataFrame:
    if _AOI_CACHE is not None:
        return _AOI_CACHE
    return gpd.read_parquet(os.path.join(parambasedir, aoi_filename))


def _load_background(parambasedir: str, spgroup: str) -> pl.DataFrame:
    if spgroup in _BG_CACHE:
        return _BG_CACHE[spgroup].clone()

    csv_files = glob.glob(os.path.join(parambasedir, f"background*{spgroup}*.csv"))
    bgs = []
    for file_path in csv_files:
        print(f"Reading: {file_path}")
        bgs.append(pl.read_csv(file_path))

    combined_bg = pl.concat(bgs, how="diagonal")
    combined_bg = combined_bg.with_columns(pl.lit(0).alias("label"))
    _BG_CACHE[spgroup] = combined_bg
    return combined_bg.clone()


def _extract_lon_lat(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col(".geo")
        .str.json_path_match("$['coordinates'][0]")
        .cast(pl.Float64)
        .alias("longitude"),
        pl.col(".geo")
        .str.json_path_match("$['coordinates'][1]")
        .cast(pl.Float64)
        .alias("latitude"),
    ]).drop(["system:index", ".geo"])


def _load_presence_points(parambasedir: str, sp: str) -> pl.DataFrame:
    csv_files = glob.glob(os.path.join(parambasedir, f"*{sp}*.csv"))
    dfs = []
    for file_path in csv_files:
        print(f"Reading: {file_path}")
        df = pl.read_csv(file_path)
        df = _extract_lon_lat(df)
        dfs.append(df)

    combined_df = pl.concat(dfs, how="diagonal")
    return combined_df.with_columns(pl.lit(1).alias("label"))


@contextmanager
def _batch_plot_context(batch_mode: bool):
    if batch_mode:
        plt.switch_backend("Agg")
    try:
        yield
    finally:
        if batch_mode:
            plt.close("all")


def _show_or_close(batch_mode: bool) -> None:
    if batch_mode:
        plt.close()
    else:
        plt.show()


def plot_elapid_responses(
    model,
    training_data: pd.DataFrame,
    target_var: str,
    title: str = "Response Curve",
    *,
    outputdir: str | None = None,
    sp: str | None = None,
    batch_mode: bool = False,
) -> None:
    v_min, v_max = training_data[target_var].min(), training_data[target_var].max()
    v_range = np.linspace(v_min, v_max, 100)

    mean_data = training_data.mean(axis=0).to_frame().T
    eval_df = pd.concat([mean_data] * 100, ignore_index=True)
    eval_df[target_var] = v_range

    preds = model.predict(eval_df)
    display_name = rename_col(target_var)

    plt.figure(figsize=(6, 4))
    plt.plot(v_range, preds)
    plt.xlabel(f"{display_name} value")
    plt.ylabel("Suitability")
    plt.title(f"{title}: {display_name}")
    plt.grid(True, alpha=0.3)

    if outputdir and sp:
        safe_name = target_var.replace("/", "_")
        plt.savefig(os.path.join(outputdir, f"Response_{sp}_{safe_name}.png"))
    _show_or_close(batch_mode)


def _build_maxent_model(
    feature_types,
    base_tau: float,
    reg: float,
    n_cpus: int,
    categorical_features: list[str] | None = None,
):
    return elapid.MaxentModel(
        feature_types=feature_types,
        tau=base_tau,
        clamp=True,
        scorer="roc_auc",
        beta_multiplier=reg,
        beta_lqp=1.0,
        beta_hinge=1.0,
        beta_threshold=1.0,
        beta_categorical=1.0,
        n_hinge_features=10,
        n_threshold_features=10,
        convergence_tolerance=1e-4,
        use_lambdas="best",
        n_cpus=n_cpus,
        categorical_features=categorical_features or [],
    )


def _eval_reg(
    reg: float,
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    feature_types,
    base_tau: float,
    n_cpus: int,
    categorical_features: list[str] | None = None,
) -> dict:
    model_alpha = _build_maxent_model(
        feature_types, base_tau, reg, n_cpus, categorical_features=categorical_features
    )

    all_y_true_alpha, all_y_pred_prob_alpha = [], []
    aucs_alpha = []

    for train_idx, test_idx in folds:
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        model_alpha.fit(X_train, y_train)
        y_pred_prob_alpha = model_alpha.predict(X_test)

        all_y_true_alpha.extend(y_test)
        all_y_pred_prob_alpha.extend(y_pred_prob_alpha)

        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob_alpha)
            aucs_alpha.append(auc(fpr, tpr))

    all_y_true_alpha = np.array(all_y_true_alpha)
    all_y_pred_prob_alpha = np.array(all_y_pred_prob_alpha)

    precisions_alpha, recalls_alpha, _ = precision_recall_curve(
        all_y_true_alpha, all_y_pred_prob_alpha
    )
    f1_scores_alpha = (
        2 * (precisions_alpha * recalls_alpha) / (precisions_alpha + recalls_alpha + 1e-9)
    )
    best_f1_alpha = float(np.max(f1_scores_alpha))
    mean_auc_alpha = float(np.mean(aucs_alpha)) if len(aucs_alpha) else float("nan")

    return {
        "beta_multiplier": reg,
        "best_f1": best_f1_alpha,
        "mean_auc": mean_auc_alpha,
    }


def fit_maxent_with_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    coords: np.ndarray,
    outputdir: str,
    sp: str,
    selection_metric: str = "best_f1",
    target_vars=None,
    base_tau: float = 0.5,
    n_splits: int = 5,
    random_state: int = 42,
    reg_values: list[float] | None = None,
    feature_types: list[str] | None = None,
    categorical_features: list[str] | None = None,
    excel_group: str = "birds",
    spgroup: str = "avian",
    n_cpus: int = 8,
    inner_parallel: bool = False,
    batch_mode: bool = False,
) -> dict:
    os.makedirs(outputdir, exist_ok=True)

    X = X.copy()
    y = y.astype(int).copy()
    n_pres = int(y.sum())

    if feature_types is None:
        feature_types = (
            ["linear", "quadratic"] if n_pres < 10 else ["linear", "quadratic", "hinge"]
        )
    feature_types = ["linear", "quadratic", "product"]

    if reg_values is None:
        reg_values = (
            [1.5, 2.0, 3.0, 4.0]
            if n_pres < 10
            else [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        )
    reg_values = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    try:
        gkf = GeographicKFold(n_splits=n_splits)
        geometry = gpd.points_from_xy(X["longitude"], X["latitude"])
        gdf_coords = gpd.GeoDataFrame(
            X[["longitude", "latitude"]], geometry=geometry, crs="EPSG:4326"
        )
        folds = list(gkf.split(gdf_coords))
        cv_name = "GeographicKFold"
    except Exception as exc:
        print("Not using GeographicKFold:", exc)
        folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
                X, y
            )
        )
        cv_name = "StratifiedKFold"
    print("Fold type:", cv_name)

    X = X.drop(columns=["longitude", "latitude"])

    if inner_parallel and len(reg_values) > 1:
        reg_jobs = min(len(reg_values), n_cpus)
        reg_results = Parallel(n_jobs=reg_jobs)(
            delayed(_eval_reg)(
                reg, X, y, folds, feature_types, base_tau, max(1, n_cpus // reg_jobs), categorical_features
            )
            for reg in reg_values
        )
    else:
        reg_results = [
            _eval_reg(reg, X, y, folds, feature_types, base_tau, n_cpus, categorical_features)
            for reg in reg_values
        ]

    reg_df = pd.DataFrame(reg_results)
    reg_df.to_csv(os.path.join(outputdir, f"Regularization_{sp}.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(reg_df["beta_multiplier"], reg_df["best_f1"], marker="o", label="Best F1")
    plt.xscale("log")
    plt.xlabel("beta_multiplier (log scale)")
    plt.ylabel("Best F1")
    plt.title(f"Effect of Regularization on F1 ({cv_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outputdir, f"Regularization_F1_{sp}.png"))
    _show_or_close(batch_mode)

    plt.figure(figsize=(10, 5))
    plt.plot(reg_df["beta_multiplier"], reg_df["mean_auc"], marker="o", color="orange", label="Mean AUC")
    plt.xscale("log")
    plt.xlabel("beta_multiplier (log scale)")
    plt.ylabel("Mean AUC")
    plt.title(f"Effect of Regularization on AUC ({cv_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outputdir, f"Regularization_AUC_{sp}.png"))
    _show_or_close(batch_mode)

    sel = selection_metric if selection_metric in {"best_f1", "mean_auc"} else "best_f1"
    if sel == "mean_auc" and np.all(np.isnan(reg_df["mean_auc"].values)):
        sel = "best_f1"

    best_entry = max(reg_results, key=lambda r: r[sel] if not np.isnan(r[sel]) else -np.inf)
    best_beta = best_entry["beta_multiplier"]
    best_score = best_entry[sel]
    print(f"Selected beta_multiplier={best_beta} (CV {sel}={best_score:.4f})")

    with open(os.path.join(outputdir, f"best_beta_{sp}.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(best_beta))

    final_model = _build_maxent_model(
        feature_types, base_tau, best_beta, n_cpus, categorical_features=categorical_features
    )
    final_model.fit(X, y)
    y_pred_prob_full = final_model.predict(X)

    precisions_f, recalls_f, thresholds_f = precision_recall_curve(y, y_pred_prob_full)
    f1_scores_f = 2 * (precisions_f * recalls_f) / (precisions_f + recalls_f + 1e-9)
    idx_f = int(np.argmax(f1_scores_f))
    final_threshold = thresholds_f[idx_f]
    final_f1 = float(f1_scores_f[idx_f])

    if len(np.unique(y)) > 1:
        fpr_f, tpr_f, _ = roc_curve(y, y_pred_prob_full)
        final_auc = float(auc(fpr_f, tpr_f))
    else:
        final_auc = float("nan")

    final_precision = precision_score(y, (y_pred_prob_full >= final_threshold).astype(int))
    final_recall = recall_score(y, (y_pred_prob_full >= final_threshold).astype(int))
    final_logloss = log_loss(y, y_pred_prob_full, labels=[0, 1])
    prevalence = y.mean()

    print(f"Final tuned model (beta_multiplier={best_beta})")
    print(
        f"AUC={final_auc:.4f}  Precision={final_precision:.4f}  Recall={final_recall:.4f}  "
        f"F1={final_f1:.4f}"
    )
    print(
        f"Best threshold={final_threshold:.4f}  Log-loss={final_logloss:.4f}  "
        f"Prevalence={prevalence:.4f}"
    )

    pd.Series(X.columns).to_csv(
        os.path.join(outputdir, f"predictors_{sp}.txt"), index=False, header=False
    )

    config = {
        "scientific_name": sp,
        "excel_group": excel_group,
        "spgroup": spgroup,
        "categorical_features": categorical_features or [],
        "predictor_columns": list(X.columns),
    }
    with open(os.path.join(outputdir, f"model_config_{sp}.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    joblib.dump(final_model, os.path.join(outputdir, f"elapid_maxent_model_tuned_{sp}.pkl"))
    np.savetxt(
        os.path.join(outputdir, f"accuracy_tuned_{sp}.csv"),
        [
            ["beta_multiplier", best_beta],
            ["SelectionMetric", sel],
            ["AUC", final_auc],
            ["Precision", final_precision],
            ["Recall", final_recall],
            ["F1", final_f1],
            ["BestThreshold", final_threshold],
            ["LogLoss", final_logloss],
            ["Prevalence", prevalence],
        ],
        delimiter=",",
        fmt="%s",
    )

    if target_vars is not None and len(target_vars) > 0:
        skip_categorical = set(categorical_features or [])
        for var in target_vars:
            if var in skip_categorical:
                continue
            plot_elapid_responses(
                final_model,
                X,
                var,
                title=f"Beta={best_beta}",
                outputdir=outputdir,
                sp=sp,
                batch_mode=batch_mode,
            )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recalls_f, precisions_f, label="PR (final tuned)", color="navy")
    plt.scatter(recalls_f[idx_f], precisions_f[idx_f], color="red", s=80, label=f"Best F1={final_f1:.2f}")
    plt.title(f"PR Curve (beta={best_beta})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(1, 2, 2)
    if not np.isnan(final_auc):
        plt.plot(*roc_curve(y, y_pred_prob_full)[:2], label=f"ROC (AUC={final_auc:.2f})", color="darkorange")
    else:
        plt.text(0.5, 0.5, "AUC undefined (single class in y)", ha="center", va="center")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (beta={best_beta})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f"Curves_tuned_{sp}.png"))
    _show_or_close(batch_mode)

    lambdas = getattr(final_model, "lambdas_", None)
    names = getattr(final_model, "feature_names_", None)

    if lambdas is not None:
        lambdas = np.asarray(lambdas).ravel()
        if names is None or len(names) != len(lambdas):
            names = [f"feature_{i}" for i in range(len(lambdas))]
        df_imp = pd.DataFrame(
            {
                "feature": names,
                "lambda_weight": lambdas,
                "abs_importance": np.abs(lambdas),
            }
        ).sort_values("abs_importance", ascending=True)
        df_imp.to_csv(os.path.join(outputdir, f"FeatureImportance_Lambdas_{sp}.csv"), index=False)
        plt.figure(figsize=(10, max(6, 0.3 * len(names))))
        plt.barh(df_imp["feature"], df_imp["lambda_weight"], color="skyblue")
        plt.axvline(0, color="gray", linewidth=1)
        plt.title("MaxEnt Feature Importance (lambda weights)")
        plt.xlabel("Lambda weight (positive ↑ suitability)")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"FeatureImportance_Lambdas_{sp}.png"))
        _show_or_close(batch_mode)
    else:
        model_pi = _build_maxent_model(
            feature_types, base_tau, best_beta, n_cpus, categorical_features=categorical_features
        )
        try:
            X_train_pi, X_holdout_pi, y_train_pi, y_holdout_pi = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
        except ValueError:
            split = int(0.8 * len(X))
            X_train_pi, X_holdout_pi = X.iloc[:split].copy(), X.iloc[split:].copy()
            y_train_pi, y_holdout_pi = y.iloc[:split].copy(), y.iloc[split:].copy()

        model_pi.fit(X_train_pi, y_train_pi)

        def baseline_score(y_true, y_prob):
            if len(np.unique(y_true)) > 1:
                return roc_auc_score(y_true, y_prob)
            return -log_loss(y_true, y_prob, labels=[0, 1])

        base_score = baseline_score(y_holdout_pi, model_pi.predict(X_holdout_pi))

        def robust_prob_scorer(est, X_eval, y_eval):
            y_prob = est.predict(X_eval)
            if len(np.unique(y_eval)) > 1:
                return roc_auc_score(y_eval, y_prob)
            return -log_loss(y_eval, y_prob, labels=[0, 1])

        perm = permutation_importance(
            model_pi,
            X_holdout_pi,
            y_holdout_pi,
            scoring=robust_prob_scorer,
            n_repeats=10,
        )

        imp = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=True)

        imp.to_csv(os.path.join(outputdir, f"PermutationImportance_{sp}.csv"), index=False)
        plt.figure(figsize=(10, max(6, 0.3 * len(imp))))
        plt.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color="steelblue")
        plt.axvline(0, color="gray", linewidth=1)
        metric_label = "AUC" if len(np.unique(y_holdout_pi)) > 1 else "−LogLoss"
        plt.title(f"Permutation Importance ({metric_label} change). Baseline={base_score:.3f}")
        plt.xlabel(f"Mean importance (Δ{metric_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"PermutationImportance_{sp}.png"))
        _show_or_close(batch_mode)

    print("Tuned final MaxEnt (PPP-equivalent) model saved, with diagnostics & importance.")
    return {
        "best_beta": best_beta,
        "selection_metric": sel,
        "final_auc": final_auc,
        "final_f1": final_f1,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_threshold": final_threshold,
        "final_logloss": final_logloss,
        "prevalence": prevalence,
        "cv_name": cv_name,
        "reg_results": reg_results,
    }


def run_maxent_species(
    sp: str,
    spgroup: str,
    parambasedir: str,
    baseoutputdir: str,
    aoi_filename: str,
    *,
    n_cpus: int = 4,
    batch_mode: bool = True,
    inner_parallel: bool = False,
    selection_metric: str = "mean_auc",
    excel_group: str = "birds",
) -> dict:
    """Run the full MaxEnt modeling pipeline for one species."""
    sp = sp.lower().replace(" ", "_")

    if "HUGE" in baseoutputdir:
        outputdir = baseoutputdir
    else:
        outputdir = os.path.join(baseoutputdir, "ppp_paramsoutput")
        os.makedirs(outputdir, exist_ok=True)
        outputdir = os.path.join(outputdir, sp)
        os.makedirs(outputdir, exist_ok=True)

    print(outputdir)
    aoi = _get_aoi(parambasedir, aoi_filename)

    with _batch_plot_context(batch_mode):
        combined_df = _load_presence_points(parambasedir, sp)

        gdf = gpd.GeoDataFrame(
            combined_df.to_pandas(),
            geometry=gpd.points_from_xy(combined_df["longitude"], combined_df["latitude"]),
            crs="EPSG:4326",
        )
        convex_hull_geom = gdf.geometry.union_all().convex_hull
        hull_gdf = gpd.GeoDataFrame({"id": [1], "geometry": [convex_hull_geom]}, crs="EPSG:4326")
        hull_gdf.to_file(os.path.join(outputdir, f"convex_hull_{sp}.json"), driver="GeoJSON")

        if "month" not in combined_df.columns:
            combined_df = combined_df.with_columns(
                pl.col("obs_date")
                .str.strptime(pl.Date, format="%Y-%m-%d")
                .dt.month()
                .alias("month")
            )

        month_counts = combined_df.group_by("month").len().sort("month")
        month_counts_pd = month_counts.to_pandas()
        plt.figure(figsize=(8, 5))
        plt.bar(month_counts_pd["month"], month_counts_pd["len"], color="skyblue")
        plt.title(f"Row Count by Month for {sp}")
        plt.xlabel("Month")
        plt.ylabel("Row Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"CountByMonth_{sp}.png"))
        _show_or_close(batch_mode)

        combined_bg = _load_background(parambasedir, spgroup)

        x = combined_df["longitude"]
        y = combined_df["latitude"]
        xy = np.vstack([x, y])
        kde = stats.gaussian_kde(xy)
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        X_grid, Y_grid = np.meshgrid(
            np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        )
        Z = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)

        plt.figure(figsize=(8, 6))
        plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="viridis")
        plt.scatter(x, y, s=5, c="red")
        plt.title(f"Presence KDE for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"KDensity_{sp}.png"))
        _show_or_close(batch_mode)

        xres = (xmax - xmin) / Z.shape[1]
        yres = (ymax - ymin) / Z.shape[0]
        transform = from_origin(xmin, ymax, xres, yres)
        Z_clean = np.nan_to_num(Z, nan=0.0)
        Z_clean[Z_clean < 0] = 0.0
        Z_raster = np.flipud(Z_clean)

        with rasterio.Env():
            with MemoryFile() as memfile:
                with memfile.open(
                    driver="GTiff",
                    height=Z_raster.shape[0],
                    width=Z_raster.shape[1],
                    count=1,
                    dtype=Z_raster.dtype,
                    crs="EPSG:4326",
                    transform=transform,
                ) as dataset:
                    dataset.write(Z_raster, 1)
                vsi_path = memfile.name
                pseudoabsence_bias = elapid.sample_bias_file(vsi_path, 10000)

        fig, ax = plt.subplots(figsize=(8, 6))
        pseudoabsence_bias.plot(ax=ax, markersize=0.5)
        plt.title(f"Bias surface for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"BiasSurface_{sp}.png"))
        _show_or_close(batch_mode)

        geometry = gpd.points_from_xy(combined_bg["longitude"], combined_bg["latitude"])
        bgdf = gpd.GeoDataFrame(combined_bg.to_pandas().copy(), geometry=geometry, crs="EPSG:4326")
        pseudo = gpd.GeoDataFrame(geometry=pseudoabsence_bias, crs="EPSG:4326")

        A_with_nearest_B = gpd.sjoin_nearest(
            pseudo,
            bgdf,
            how="left",
            distance_col="dist_m",
        )
        unique_b_idxs = A_with_nearest_B["index_right"].dropna().astype(int).unique()
        B_for_each_A = bgdf.loc[unique_b_idxs].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        aoi.plot(ax=ax, color="red")
        B_for_each_A.plot(ax=ax, markersize=0.5)
        plt.title(f"Background points for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"BackgroundPoints_{sp}.png"))
        _show_or_close(batch_mode)

        print("Total presence points:", len(combined_df))
        print("Total background points:", len(B_for_each_A))
        pd.Series(
            [
                f"Total presence points: {len(combined_df)}",
                f"Total background points: {len(B_for_each_A)}",
            ]
        ).to_csv(os.path.join(outputdir, f"PointNums_{sp}.txt"), index=False, header=False)

        combined_bg = pl.from_pandas(B_for_each_A.drop(columns="geometry"))
        combined_bg = combined_bg.with_columns(
            pl.col("obs_date").str.to_datetime(strict=False).dt.month().alias("month")
        )
        month_counts = combined_bg.group_by("month").len().sort("month")
        month_counts_pd = month_counts.to_pandas()
        plt.figure(figsize=(8, 5))
        plt.bar(month_counts_pd["month"], month_counts_pd["len"], color="skyblue")
        plt.title("Background count by month")
        plt.xlabel("Month")
        plt.ylabel("Row Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"CountByMonth_bg_{sp}.png"))
        _show_or_close(batch_mode)

        data = pd.concat([combined_df.to_pandas(), combined_bg.to_pandas()], ignore_index=True)
        coords = data[["longitude", "latitude"]].to_numpy()
        X, categorical_features = prepare_predictors(data, excel_group)
        y = data["label"].astype(int)

        target_vars = [
            col
            for col in X.drop(columns=["latitude", "longitude"]).columns
            if col not in categorical_features
        ]
        if len(combined_df) <= len(target_vars):
            raise ValueError("Not enough presence records")

        results = fit_maxent_with_tuning(
            X=X,
            y=y,
            coords=coords,
            outputdir=outputdir,
            sp=sp,
            selection_metric=selection_metric,
            target_vars=target_vars,
            categorical_features=categorical_features,
            excel_group=excel_group,
            spgroup=spgroup,
            n_cpus=n_cpus,
            inner_parallel=inner_parallel,
            batch_mode=batch_mode,
        )

        dfX = X.drop(columns=["longitude", "latitude"]).rename(columns=rename_col)
        num_df = dfX.select_dtypes(include="number")
        corr = num_df.corr(method="spearman")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=False, square=True)
        plt.title("Variable Correlations (Spearman)")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"Correlation_{sp}.png"))
        _show_or_close(batch_mode)

    return results
