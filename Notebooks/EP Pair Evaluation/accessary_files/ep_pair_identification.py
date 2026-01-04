import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def EP_pair_identification_cv(
    df,
    feature_sets,
    distance_col="distanceToTSS_abs",
    n_splits=5,
    pca_components=50,
    random_state=0,
):
    labels = list(feature_sets.keys())
    cmap = plt.get_cmap("tab20")  # good for categorical colors
    colors = [cmap(i % cmap.N) for i in range(len(labels))]
    results = []

    y = df["Significant"].astype(int).to_numpy()

    for (label, feats_orig), color in zip(feature_sets.items(), colors):
        feats = list(feats_orig)
        feat_mat = []
        feature_names = []

        embedding_cols = []
        scalar_cols = []

        if "embeddings" in feats:
            E = np.vstack(df["enhancer_embedding_mean"].to_numpy())
            P = np.vstack(df["promoter_embedding_mean"].to_numpy())
            EP = np.hstack([E, P]).astype(np.float32)

            dE, dP = E.shape[1], P.shape[1]
            emb_names = [f"enh_emb_{j}" for j in range(dE)] + [f"prom_emb_{j}" for j in range(dP)]
            embedding_cols = emb_names

            feat_mat.append(EP)
            feature_names += emb_names

            feats.remove("embeddings")

        for feat in feats:
            if feat == "distanceToTSS_abs":
                col = df[distance_col].to_numpy().reshape(-1, 1)
                name = "distance"
            else:
                col = df[feat].to_numpy().reshape(-1, 1)
                name = feat

            feat_mat.append(col.astype(np.float32))
            feature_names.append(name)
            scalar_cols.append(name)

        X = np.hstack(feat_mat).astype(np.float32)
        X_df = pd.DataFrame(X, columns=feature_names)

        poly_scale_pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scale", StandardScaler()),
        ])

        emb_pca_pipeline = Pipeline([
            ("pca", PCA(n_components=min(pca_components, len(embedding_cols)), random_state=random_state)),
        ])

        transformers = []

        if embedding_cols:
            transformers.append(("emb_pca", emb_pca_pipeline, embedding_cols))

        poly_cols = [c for c in ["distance", "enhancer_dnase_avg", "promoter_dnase_avg"] if c in X_df.columns]
        if poly_cols:
            transformers.append(("poly_scale", poly_scale_pipeline, poly_cols))

        used_cols = set()
        for _, _, cols in transformers:
            used_cols.update(cols)

        remainder_cols = [c for c in X_df.columns if c not in used_cols]

        remainder_num = X_df[remainder_cols].select_dtypes(include="number").columns.tolist()
        remainder_other = [c for c in remainder_cols if c not in remainder_num]

        if remainder_num:
            transformers.append(("remainder_scale", StandardScaler(), remainder_num))

        if remainder_other:
            transformers.append(("remainder_passthrough", "passthrough", remainder_other))

        preprocess = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",)),
        ])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        y_pred_proba = cross_val_predict(pipe, X_df, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }

        results.append({
            "label": label,
            "color": color,
            "metrics": metrics,
            "y_true": y,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        })

    return results

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def plot_distance_stratified_metrics(
    df,
    results,
    model_class,
    distance_col="distanceToTSS_abs",
    n_bins=10,
    metrics=("f1", "precision", "recall", "accuracy"),
    use_proba_threshold=0.5,  # used if a result only has y_pred_proba
):
    """
    Plots distance-stratified metrics for each model in `results` returned by EP_pair_identification_cv.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `distance_col`.
    results : list[dict]
        Each dict should contain:
          - "label"
          - "color" (optional)
          - "y_true"
          - either "y_pred" or "y_pred_proba"
          - "metrics" (optional)
    distance_col : str
        Column used for stratification.
    n_bins : int
        Number of quantile bins.
    metrics : tuple[str]
        Any of: "f1", "precision", "recall", "accuracy".
    use_proba_threshold : float
        Threshold to convert probabilities to labels if needed.
    """
    dist = df[distance_col].to_numpy()
    y_true_global = df["Significant"].astype(int).to_numpy()

    # quantile bin edges (robust to repeats)
    edges = np.nanquantile(dist, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough unique distance values to create bins. Try smaller n_bins.")

    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(dist, edges[1:-1], right=True)  # 0..(B-1)
    B = len(edges) - 1

    metric_fns = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    }
    for m in metrics:
        if m not in metric_fns:
            raise ValueError(f"Unknown metric '{m}'. Choose from {list(metric_fns.keys())}.")

    # one figure per metric (cleaner than subplots)
    for metric_name in metrics:
        plt.figure(figsize=(7, 4))

        for r in results:
            label = r.get("label", "model")
            color = r.get("color", None)

            y_true = r.get("y_true", y_true_global)
            if "y_pred" in r and r["y_pred"] is not None:
                y_pred = np.asarray(r["y_pred"]).astype(int)
            else:
                proba = np.asarray(r["y_pred_proba"])
                y_pred = (proba >= use_proba_threshold).astype(int)

            vals = np.full(B, np.nan, dtype=float)
            counts = np.zeros(B, dtype=int)

            for b in range(B):
                idx = np.where(bin_ids == b)[0]
                counts[b] = len(idx)
                if counts[b] == 0:
                    continue
                vals[b] = metric_fns[metric_name](y_true[idx], y_pred[idx])

            # plot only bins with data
            mask = ~np.isnan(vals)
            plt.plot(centers[mask], vals[mask], marker="o", label=label, color=color)
            # optionally show bin counts as faint points near x-axis (comment out if you don't want)
            # plt.scatter(centers[mask], np.zeros_like(centers[mask]), s=counts[mask], alpha=0.15, color=color)

        plt.xlabel(distance_col)
        plt.ylabel(metric_name)
        plt.title(f"Distance-stratified {metric_name} (quantile bins={B}), {model_class}")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curves(results, title="Precision–Recall (out-of-fold)"):
    """
    results: list of dicts from EP_pair_identification_cv, each containing
      - 'label', 'color', 'y_true', 'y_pred_proba'
    """
    plt.figure(figsize=(13, 10))

    for r in results:
        y_true = np.asarray(r["y_true"]).astype(int)
        y_score = np.asarray(r["y_pred_proba"]).astype(float)

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.plot(rec, prec, label=f'{r["label"]} (AP={ap:.3f})', color=r.get("color", None))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.xlim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


import numpy as np

def _get_model_by_label(results, label):
    matches = [r for r in results if r.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 model with label='{label}', found {len(matches)}.")
    return matches[0]

def _per_sample_logloss(y_true, p, eps=1e-15):
    y = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def _per_sample_brier(y_true, p):
    y = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    return (p - y) ** 2

def paired_swap_test(
    results,
    label0,
    label1,
    metric="logloss",
    B=20000,
    seed=0,
    two_sided=True,
    return_null=False,
):
    m0 = _get_model_by_label(results, label0)
    m1 = _get_model_by_label(results, label1)

    y0 = np.asarray(m0["y_true"])
    y1 = np.asarray(m1["y_true"])
    if y0.shape != y1.shape or not np.array_equal(y0, y1):
        raise ValueError("y_true arrays differ between the two models. "
                         "Swap test requires aligned pairs (same samples in same order).")

    p0 = np.asarray(m0["y_pred_proba"], dtype=float)
    p1 = np.asarray(m1["y_pred_proba"], dtype=float)

    if metric == "logloss":
        l0 = _per_sample_logloss(y0, p0)
        l1 = _per_sample_logloss(y0, p1)
    elif metric == "brier":
        l0 = _per_sample_brier(y0, p0)
        l1 = _per_sample_brier(y0, p1)
    else:
        raise ValueError("metric must be 'logloss' or 'brier'.")

    # Paired differences in loss: positive means model1 has lower loss => better
    d = l0 - l1
    delta_obs = float(d.mean())

    rng = np.random.default_rng(seed)

    signs = rng.choice([-1.0, 1.0], size=(B, d.size), replace=True)
    null_deltas = (signs * d[None, :]).mean(axis=1)

    if two_sided:
        p_value = (1.0 + np.sum(np.abs(null_deltas) >= abs(delta_obs))) / (B + 1.0)
    else:
        p_value = (1.0 + np.sum(null_deltas >= delta_obs)) / (B + 1.0)

    out = {
        "delta_obs": delta_obs,
        "p_value": float(p_value),
        "n": int(d.size),
        "metric": metric,
        "labels": (label0, label1),
    }
    if return_null:
        out["null_deltas"] = null_deltas
    return out

