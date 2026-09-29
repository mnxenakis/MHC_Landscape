"""Core numerical routines ported from the legacy Bregman clustering script.

This module implements distance computations, clustering algorithms, and a few
small plotting/diagnostics helpers. To make behavior easier to tune and to
avoid magic numbers, commonly tweaked numeric and styling defaults are exposed
as module-level constants below. Public APIs remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .mpl_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .parameters import get_parameter_defaults

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - optional dependency
    DBSCAN = None

_PARAM_DEFAULTS = get_parameter_defaults()


def _tuple_from_defaults(key: str) -> tuple[float, float]:
    raw = _PARAM_DEFAULTS[key]
    if isinstance(raw, (tuple, list)):
        return tuple(float(v) for v in raw)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Parameter {key} must define two comma-separated values.")
        return float(parts[0]), float(parts[1])
    raise TypeError(f"Parameter {key} must be a tuple/list or 'a,b' string, got {type(raw)}")


_EPS = float(_PARAM_DEFAULTS["algo_eps"])  # Numerical floor to avoid log/div-by-zero

# ----- Algorithmic defaults (sourced from parameters.log) -----
EUCLIDEAN_KMEANS_MAX_ITER = int(_PARAM_DEFAULTS["algo_euclidean_kmeans_max_iter"])
SPECTRAL_MAX_ATTEMPTS = int(_PARAM_DEFAULTS["algo_spectral_max_attempts"])
SPECTRAL_KNN_STEP = int(_PARAM_DEFAULTS["algo_spectral_knn_step"])
DBSCAN_EPS_QUANTILE = float(_PARAM_DEFAULTS["algo_dbscan_eps_quantile"])
DBSCAN_FALLBACK_EPS = float(_PARAM_DEFAULTS["algo_dbscan_fallback_eps"])
PERMUTATION_ITERS_DEFAULT = int(_PARAM_DEFAULTS["cli_permutations"])

# ----- Plotting defaults (tunable, cosmetic) -----
SCATTER_POINT_SIZE = float(_PARAM_DEFAULTS["algo_scatter_point_size"])
SCATTER_ALPHA = float(_PARAM_DEFAULTS["algo_scatter_alpha"])
GRID_ALPHA = float(_PARAM_DEFAULTS["algo_grid_alpha"])
LABEL_FONTSIZE = float(_PARAM_DEFAULTS["algo_label_fontsize"])
LABEL_FONTSIZE_CLASS1 = float(_PARAM_DEFAULTS["algo_label_fontsize_class1"])
LABEL_COLOR_CLASS1 = str(_PARAM_DEFAULTS["algo_label_color_class1"])
CENTROID_LABEL_FONTSIZE = float(_PARAM_DEFAULTS["algo_centroid_label_fontsize"])
FIGSIZE_CLUSTER_SCATTER = _tuple_from_defaults("algo_figsize_cluster_scatter")
FIGSIZE_CONFUSION_MARGINS = _tuple_from_defaults("algo_figsize_confusion_margins")

rcParams.update(
    {
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "font.family": "serif",
        "text.usetex": False,
    }
)


def format_class_value(value: object | None) -> str:
    """Convert class values (including numpy scalars) to readable strings."""

    if value is None:
        return "N/A"
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_simplex(arr: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Return a copy of arr with each row projected onto the probability simplex."""

    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    elif data.ndim == 2:
        data = data.copy()
        squeeze = False
    else:
        raise ValueError("Input must be 1D or 2D to normalise onto the simplex.")

    data = np.clip(data, eps, None)
    row_sums = data.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("Non-positive row sum encountered during simplex normalisation.")
    data /= row_sums
    return data[0] if squeeze else data


def _kl_distance_matrix(
    X: np.ndarray,
    centroids: np.ndarray,
    data_entropy: np.ndarray | None = None,
    eps: float = _EPS,
) -> np.ndarray:
    """Return the KL divergence matrix D(X_i || C_j).

    This is the Bregman divergence for multinomial/simplex vectors.
    Assumes rows of `X` and `centroids` represent (approximately) probability
    vectors; values are clipped by `eps` to avoid log/div-by-zero.

    If `data_entropy` is supplied it must be the vector H(X_i)=∑ X_i log X_i
    (note: negative Shannon entropy), which saves recomputation inside k-means.
    """

    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if data_entropy is None:
        safe_X = np.clip(X, eps, None)
        data_entropy = np.sum(safe_X * np.log(safe_X), axis=1)
    safe_centroids = np.clip(centroids, eps, None)
    log_centroids = np.log(safe_centroids)
    return data_entropy[:, None] - X @ log_centroids.T


def _js_distance_matrix(X: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Return the Jensen–Shannon distance matrix for rows of X."""

    X = _normalize_simplex(X, eps)
    safe_X = np.clip(X, eps, None)
    entropies = -np.sum(safe_X * np.log(safe_X), axis=1)

    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        Pi = safe_X[i]
        for j in range(i + 1, n_samples):
            Pj = safe_X[j]
            M = 0.5 * (Pi + Pj)
            entropy_m = -np.sum(M * np.log(np.clip(M, eps, None)))
            # JS divergence = H(M) - 1/2(H(P)+H(Q)), with M=(P+Q)/2.
            # We clamp at zero to guard against tiny negative values from
            # floating-point error before taking sqrt.
            divergence = entropy_m - 0.5 * (entropies[i] + entropies[j])
            divergence = max(divergence, 0.0)
            distance = float(np.sqrt(divergence))
            distances[i, j] = distances[j, i] = distance
    return distances


def _kmeanspp_init_kl(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    eps: float = _EPS,
) -> np.ndarray:
    """Seed centroids using k-means++ style initialisation under KL divergence."""
    # NOTE: All randomness flows through the provided `rng`; fixing the seed makes this deterministic.

    X = _normalize_simplex(X, eps)
    n_samples = X.shape[0]
    if k > n_samples:
        raise ValueError("Number of clusters cannot exceed number of samples for initialisation.")

    safe_X = np.clip(X, eps, None)
    data_entropy = np.sum(safe_X * np.log(safe_X), axis=1)

    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    first = rng.integers(0, n_samples)
    centroids[0] = X[first]
    min_distances = _kl_distance_matrix(X, centroids[0:1], data_entropy, eps)[:, 0]

    for idx in range(1, k):
        weights = np.clip(min_distances, 0.0, None)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            next_index = rng.integers(0, n_samples)
        else:
            probabilities = weights / total
            next_index = rng.choice(n_samples, p=probabilities)
        centroids[idx] = X[next_index]
        distances = _kl_distance_matrix(X, centroids[idx:idx + 1], data_entropy, eps)[:, 0]
        min_distances = np.minimum(min_distances, distances)

    return centroids


def _knn_affinity(
    distance_matrix: np.ndarray,
    k: int,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Build a symmetric kNN affinity matrix using an RBF kernel."""

    D = np.asarray(distance_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")

    n = D.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    k = max(1, min(int(k), n - 1))

    positives = D[D > 0]
    if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
        if positives.size == 0:
            sigma_val = 1.0
        else:
            sigma_val = float(np.median(positives))
            if not np.isfinite(sigma_val) or sigma_val <= 0.0:
                sigma_val = 1.0
    else:
        sigma_val = max(float(sigma), _EPS)
    # Convert distances to similarities. Larger affinity means closer.
    affinity_full = np.exp(-np.square(D) / (sigma_val ** 2))

    np.fill_diagonal(affinity_full, 0.0)

    A = np.zeros_like(affinity_full)
    if k >= n - 1:
        A[:] = affinity_full
    else:
        # Keep the k *largest* affinities per row (i.e., k nearest neighbours).
        idx = np.argpartition(affinity_full, -k, axis=1)[:, -k:]
        row_selector = np.zeros_like(affinity_full, dtype=bool)
        row_selector[np.arange(n)[:, None], idx] = True
        A[row_selector] = affinity_full[row_selector]

    A = np.maximum(A, A.T)
    return A


def _spectral_embedding(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return the spectral embedding (row-normalised eigenvectors) for affinity graph."""

    A = np.asarray(affinity, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Affinity matrix must be square.")
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive for spectral embedding.")

    degrees = A.sum(axis=1)
    if np.allclose(degrees, 0.0):
        raise ValueError("Affinity matrix has zero degree for all nodes.")

    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    # Symmetric normalisation: S = D^{-1/2} A D^{-1/2}.
    # We use the top-eigenvector subspace as the embedding.
    S = (inv_sqrt[:, None] * A) * inv_sqrt[None, :]

    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[-n_clusters:]
    embedding = vecs[:, order]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return embedding / norms


def _kmeanspp_euclidean(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation under Euclidean distance."""

    n_samples, n_features = X.shape
    centroids = np.empty((n_clusters, n_features), dtype=np.float64)
    first = rng.integers(0, n_samples)
    centroids[0] = X[first]
    dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

    for idx in range(1, n_clusters):
        total = dist_sq.sum()
        if not np.isfinite(total) or total <= 0.0:
            candidate = rng.integers(0, n_samples)
        else:
            probs = dist_sq / total
            candidate = rng.choice(n_samples, p=probs)
        centroids[idx] = X[candidate]
        dist_sq = np.minimum(dist_sq, np.sum((X - centroids[idx]) ** 2, axis=1))
    return centroids


def _euclidean_kmeans(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int = EUCLIDEAN_KMEANS_MAX_ITER,
) -> np.ndarray:
    """Simple Euclidean k-means for spectral embeddings."""

    n_samples = X.shape[0]
    centroids = _kmeanspp_euclidean(X, n_clusters, rng)
    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cid in range(n_clusters):
            mask = labels == cid
            if not np.any(mask):
                # Empty-cluster fix: steal the point farthest from this centroid.
                farthest = np.argmax(distances[:, cid])
                labels[farthest] = cid
                mask = labels == cid
            centroids[cid] = X[mask].mean(axis=0)
    return labels


def _spectral_kmeans_labels(
    js_distances: np.ndarray,
    n_clusters: int,
    knn: int,
    sigma: float | None,
    random_state: int | None,
) -> tuple[np.ndarray | None, int]:
    """Return spectral clusters via k-means embedding; on failure returns (None, knn_used)."""

    n_samples = js_distances.shape[0]
    knn_limit = max(1, n_samples - 1)
    working_knn = max(1, min(int(knn), knn_limit))
    # Spectral embedding itself is deterministic; the only randomness is the
    # Euclidean k-means round on the embedding, which is seeded via
    # `random_state` so CLI --seed fully controls spectral mode.
    # k-means++ seeding on the simplex is the only stochastic element of
    # Bregman KL k-means; wiring `random_state` here ensures CLI --seed makes
    # runs repeatable across machines.
    rng = np.random.default_rng(random_state)

    attempt = 0
    # Try a few times, increasing k in the kNN graph to improve connectivity
    while attempt < SPECTRAL_MAX_ATTEMPTS:
        affinity = _knn_affinity(js_distances, k=working_knn, sigma=sigma)
        try:
            embedding = _spectral_embedding(affinity, n_clusters)
        except ValueError:
            embedding = None
        if embedding is not None and np.isfinite(embedding).all():
            labels = _euclidean_kmeans(embedding, n_clusters, rng)
            if np.unique(labels).size == n_clusters:
                return labels, working_knn
        attempt += 1
        next_knn = min(working_knn + SPECTRAL_KNN_STEP, knn_limit)
        if next_knn <= working_knn:
            break
        working_knn = next_knn
    return None, working_knn


def _pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project X onto the first `n_components` principal components using SVD."""

    if n_components <= 0:
        raise ValueError("Number of PCA components must be positive.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input data must be 2-dimensional for PCA.")
    mean = X.mean(axis=0, keepdims=True)
    centered = X - mean
    if centered.size == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    num_components = min(n_components, Vt.shape[0])
    projection = centered @ Vt[:num_components].T
    if num_components < n_components:
        padding = np.zeros((projection.shape[0], n_components - num_components), dtype=np.float64)
        projection = np.hstack((projection, padding))
    return projection


@dataclass(frozen=True)
class FoldxEnergyDataset:
    """Container for a single FoldX energy dataset and its raw rank energies."""

    dqa_name: str
    dqb_name: str
    energies: np.ndarray
    ranks: np.ndarray
    extra_energies: dict[str, np.ndarray]
    class_value: object | None = None

    @property
    def label(self) -> str:
        return self.dqb_name


@dataclass
class ClusterResult:
    """Outcome from clustering a set of distribution vectors."""

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int
    normalized_data: np.ndarray
    js_distances: np.ndarray
    mode: str
    knn_used: int
    sigma_used: float | None
    messages: list[str]


def bregman_kmeans_kl(
    data: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Cluster distribution vectors with Bregman k-means under KL divergence.
    Returns a tuple of (labels, centroids, inertia, iterations).
    """

    if k <= 0:
        raise ValueError("Number of clusters must be positive.")

    X = _normalize_simplex(data, _EPS)
    n_samples, _ = X.shape
    if k > n_samples:
        raise ValueError("Number of clusters cannot exceed number of samples.")

    rng = np.random.default_rng(random_state)
    centroids = _kmeanspp_init_kl(X, k, rng, _EPS)
    centroids = _normalize_simplex(centroids, _EPS)

    safe_X = np.clip(X, _EPS, None)
    data_entropy = np.sum(safe_X * np.log(safe_X), axis=1)

    labels = np.full(n_samples, -1, dtype=np.int32)
    prev_inertia: float | None = None

    for iteration in range(1, max_iter + 1):
        distances = _kl_distance_matrix(X, centroids, data_entropy, _EPS)
        new_labels = distances.argmin(axis=1)
        cluster_distances = distances[np.arange(n_samples), new_labels]

        counts = np.bincount(new_labels, minlength=k)
        if np.any(counts == 0):
            # Empty-cluster fix: re-seed missing clusters with the currently
            # worst-explained points (largest KL to their assigned centroid).
            for cluster_id, count in enumerate(counts):
                if count == 0:
                    farthest_idx = int(np.argmax(cluster_distances))
                    new_labels[farthest_idx] = cluster_id
                    cluster_distances[farthest_idx] = distances[farthest_idx, cluster_id]
            counts = np.bincount(new_labels, minlength=k)

        inertia = float(cluster_distances.sum())

        # Convergence: stable assignments.
        if np.array_equal(new_labels, labels):
            labels = new_labels
            prev_inertia = inertia
            break

        # Convergence: relative inertia improvement below tolerance.
        if prev_inertia is not None and prev_inertia > 0.0:
            if (prev_inertia - inertia) <= tol * max(prev_inertia, _EPS):
                labels = new_labels
                prev_inertia = inertia
                break

        labels = new_labels
        prev_inertia = inertia

        new_centroids = np.empty_like(centroids)
        for cluster_id in range(k):
            mask = labels == cluster_id
            centroid = X[mask].mean(axis=0)
            new_centroids[cluster_id] = _normalize_simplex(centroid, _EPS)
        centroids = new_centroids
    else:
        iteration = max_iter

    distances = _kl_distance_matrix(X, centroids, data_entropy, _EPS)
    labels = distances.argmin(axis=1)
    inertia = float(distances[np.arange(n_samples), labels].sum())
    return labels, centroids, inertia, iteration


def predict_bregman_kl(
    data: np.ndarray,
    centroids: np.ndarray,
    *,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign rows of `data` to the closest centroid under KL divergence.

    Returns a tuple of (labels, pointwise_KL).
    """

    X = _normalize_simplex(data, eps)
    centroids = _normalize_simplex(centroids, eps)
    safe_X = np.clip(X, eps, None)
    data_entropy = np.sum(safe_X * np.log(safe_X), axis=1)
    distances = _kl_distance_matrix(X, centroids, data_entropy, eps)
    labels = distances.argmin(axis=1)
    pointwise = distances[np.arange(X.shape[0]), labels]
    return labels, pointwise


def _render_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    datasets: Sequence[FoldxEnergyDataset],
    output_path: Path,
    axis_labels: tuple[str, str],
    panel_label: str | None = None,
    *,
    print_class1_pc1: bool = False,
) -> None:
    """Create a 2D scatter plot showing clustered points and their centroids."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.shape[1] != 2:
        raise ValueError("Scatter plot expects 2D embeddings.")

    # Detect exact/near duplicate coordinates to help diagnose overplotting.
    rounded = np.round(embeddings, decimals=8)
    coord_groups: dict[tuple[float, float], list[int]] = {}
    for idx, key in enumerate(map(tuple, rounded)):
        coord_groups.setdefault(key, []).append(idx)
    duplicate_coords = [(key, idxs) for key, idxs in coord_groups.items() if len(idxs) > 1]
    if duplicate_coords:
        print("\n[warning] Overlapping 2D coordinates detected in scatter plot:")
        for coord, idxs in duplicate_coords:
            entries = ", ".join(
                f"{datasets[i].label} (class={format_class_value(datasets[i].class_value)})" for i in idxs
            )
            print(f"  {coord}: {entries}")
            for i, j in combinations(idxs, 2):
                print(f"    pair: {datasets[i].label} (idx {i}) == {datasets[j].label} (idx {j})")

    unique_clusters = sorted(np.unique(labels))
    class_strings = np.array([format_class_value(ds.class_value) for ds in datasets])
    natural_mask_all = class_strings == "1"
    if print_class1_pc1:
        class1_idx = np.flatnonzero(natural_mask_all)
        if class1_idx.size:
            order = np.argsort(embeddings[class1_idx, 0])
            sorted_idx = class1_idx[order]
            print("Class-1 labels sorted by PC1 (lowest→highest), first 10:")
            for i in sorted_idx[:10]:
                label = datasets[i].label
                pc1 = float(embeddings[i, 0])
                print(f"  {label}: {pc1:.4f}")
        else:
            print("Class-1 labels sorted by PC1: none.")
    point_size = SCATTER_POINT_SIZE * 1.6
    point_alpha = 0.4
    cluster_marker_map = {0: "s", 1: "o", 2: "^"}  # square, circle, triangle
    fallback_markers = ["D", "P", "X", "*", "v", ">", "<"]

    fig_w, fig_h = FIGSIZE_CLUSTER_SCATTER
    fig, ax = plt.subplots(figsize=(fig_w * 1.2, fig_h * 1.2))
    cluster_handles: list[Line2D] = []
    for idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        marker_shape = cluster_marker_map.get(cluster_id)
        if marker_shape is None:
            marker_shape = fallback_markers[(idx - len(cluster_marker_map)) % len(fallback_markers)]
        contour_color = "black"

        nat_mask = mask & natural_mask_all
        if np.any(nat_mask):
            ax.scatter(
                embeddings[nat_mask, 0],
                embeddings[nat_mask, 1],
                s=point_size,
                color="red",
                alpha=point_alpha,
                edgecolors="black",
                linewidths=1.0,
                marker=marker_shape,
                label=None,
            )

        non_nat_mask = mask & ~natural_mask_all
        if np.any(non_nat_mask):
            ax.scatter(
                embeddings[non_nat_mask, 0],
                embeddings[non_nat_mask, 1],
                s=point_size,
                color="blue",
                alpha=point_alpha,
                edgecolors="black",
                linewidths=1.0,
                marker=marker_shape,
                label=None,
            )
        pts = embeddings[mask]
        if pts.shape[0] >= 3:
            kde = gaussian_kde(pts.T)
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            xs = np.linspace(xmin, xmax, 60)
            ys = np.linspace(ymin, ymax, 60)
            Xg, Yg = np.meshgrid(xs, ys)
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contour(
                Xg,
                Yg,
                Z,
                levels=4,
                colors=contour_color,
                linewidths=1.0,
                alpha=0.35,
            )
        centroid = embeddings[mask].mean(axis=0)
        # Position centroid labels; apply fixed adjustments per cluster to avoid overlap.
        pos_x, pos_y = centroid
        # if cluster_id == 0:
        #     pos_y = 0
        # elif cluster_id == 1:
        #     pos_x = -0.05
        # elif cluster_id == 2:
        #     pos_x = 0.12
        ax.text(
            pos_x,
            pos_y,
            f"Cluster {cluster_id}",
            color="black",
            fontsize=CENTROID_LABEL_FONTSIZE * 4.0,
            fontweight="bold",
            alpha=0.2,
            ha="center",
            va="center",
            zorder=0,
        )
        cluster_handles.append(
            Line2D(
                [],
                [],
                marker=marker_shape,
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=8,
                label=f"Cluster {cluster_id}",
            )
        )

    # Global natural-only contour to visualise overlap of naturally occurring dimers
    natural_points = embeddings[natural_mask_all]
    if natural_points.shape[0] >= 3:
        kde = gaussian_kde(natural_points.T)
        xmin, ymin = natural_points.min(axis=0)
        xmax, ymax = natural_points.max(axis=0)
        xs = np.linspace(xmin, xmax, 80)
        ys = np.linspace(ymin, ymax, 80)
        Xg, Yg = np.meshgrid(xs, ys)
        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        ax.contour(
            Xg,
            Yg,
            Z,
            levels=5,
            colors="red",
            linewidths=1.0,
            linestyles="dashed",
            alpha=0.38,
        )

    ax.set_xlabel(axis_labels[0], fontsize=32)
    ax.set_ylabel(axis_labels[1], fontsize=32)
    ax.tick_params(axis="both", which="both", labelsize=24)
    # ax.set_title(r"PCA (KL-Bregman)", fontsize=28)
    if panel_label:
        ax.text(
            0.02,
            0.96,
            panel_label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=22,
            va="top",
            ha="left",
        )
    # Reference axes
    ax.axhline(0.0, color="lightgray", linestyle="--", linewidth=1.8, alpha=0.85)
    ax.axvline(0.0, color="lightgray", linestyle="--", linewidth=1.8, alpha=0.85)
    cluster_legend = ax.legend(
        handles=cluster_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=24,
        frameon=True,
        ncol=1,
    )
    if cluster_legend:
        cluster_legend.get_frame().set_alpha(0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _symmetrised_kl_matrix(X: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Return the symmetrised KL divergence matrix: ½(KL(P||Q)+KL(Q||P))."""

    safe = np.clip(_normalize_simplex(X, eps), eps, None)
    log_safe = np.log(safe)
    self_terms = np.sum(safe * log_safe, axis=1)
    cross = safe @ log_safe.T
    kl = self_terms[:, None] - cross
    sym = 0.5 * (kl + kl.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def _mds_embedding(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS embedding from a distance matrix."""

    D = np.asarray(distance_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    components = []
    for idx in order:
        if vals[idx] <= 0 or len(components) >= n_components:
            break
        components.append(np.sqrt(vals[idx]) * vecs[:, idx])
    if not components:
        return np.zeros((n, n_components), dtype=np.float64)
    embedding = np.column_stack(components)
    if embedding.shape[1] < n_components:
        padding = np.zeros((n, n_components - embedding.shape[1]), dtype=np.float64)
        embedding = np.hstack((embedding, padding))
    return embedding


def _hierarchical_js_clustering(
    js_distances: np.ndarray,
    n_clusters: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """Return hierarchical clustering labels from a JS distance matrix."""

    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive for hierarchical clustering.")

    condensed = squareform(js_distances, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, n_clusters, criterion="maxclust").astype(np.int32) - 1
    return labels


def _dbscan_js_clustering(
    js_distances: np.ndarray,
    *,
    eps: float | None = None,
    min_samples: int = 2,
) -> tuple[np.ndarray, float]:
    """Run DBSCAN on JS distances and return (labels, eps_used)."""

    if DBSCAN is None:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for DBSCAN diagnostics.")

    if min_samples <= 0:
        raise ValueError("min_samples must be positive for DBSCAN.")

    if eps is None or not np.isfinite(eps) or eps <= 0.0:
        positives = js_distances[js_distances > 0.0]
        if positives.size:
            eps_val = float(np.quantile(positives, DBSCAN_EPS_QUANTILE))
            eps_val = max(eps_val, _EPS)
        else:
            eps_val = DBSCAN_FALLBACK_EPS
    else:
        eps_val = float(eps)

    model = DBSCAN(metric="precomputed", eps=eps_val, min_samples=int(min_samples))
    labels = model.fit_predict(js_distances).astype(np.int32)
    return labels, eps_val


def _map_labels(labels: np.ndarray) -> tuple[np.ndarray, Dict[int, int]]:
    """Map arbitrary cluster IDs to a dense 0..k-1 range.

    DBSCAN uses -1 for noise; we preserve -1 if present.
    """

    labels = np.asarray(labels)
    unique = np.unique(labels)

    mapping: Dict[int, int] = {}
    next_id = 0
    if -1 in unique:
        mapping[-1] = -1
    for old in unique:
        old_int = int(old)
        if old_int == -1:
            continue
        mapping[old_int] = next_id
        next_id += 1

    remapped = np.array([mapping[int(val)] for val in labels], dtype=np.int32)
    return remapped, mapping


def _compute_centroids(labels: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Return simplex-normalised centroids for each cluster label."""

    unique = np.unique(labels)
    centroids = np.empty((unique.size, data.shape[1]), dtype=np.float64)
    for idx, label in enumerate(unique):
        mask = labels == label
        centroid = data[mask].mean(axis=0)
        centroids[idx] = _normalize_simplex(centroid, _EPS)
    return centroids


def _cluster_distributions(
    distributions: Sequence[np.ndarray] | np.ndarray,
    *,
    cluster_mode: str,
    k: int,
    max_iter: int,
    tol: float,
    seed: int | None,
    knn: int,
    sigma: float | None,
    linkage_method: str = "average",
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 2,
) -> ClusterResult:
    """Cluster pre-computed distribution vectors and report useful artefacts."""

    if isinstance(distributions, np.ndarray):
        if distributions.ndim != 2:
            raise ValueError("Distributions array must be 2D.")
        data_matrix = distributions
    else:
        if not distributions:
            raise ValueError("No distribution vectors supplied for clustering.")
        data_matrix = np.vstack(distributions)

    normalized_data = _normalize_simplex(data_matrix, _EPS)
    js_distances = _js_distance_matrix(normalized_data)

    n_samples = normalized_data.shape[0]
    if k <= 0:
        raise ValueError("Number of clusters must be positive.")
    if k > n_samples:
        raise ValueError("Number of clusters cannot exceed number of samples.")

    messages: list[str] = []
    mode = cluster_mode
    knn_limit = max(1, n_samples - 1)
    knn_used = max(1, min(int(knn), knn_limit)) if knn_limit > 0 else 0
    sigma_arg = sigma if (sigma is not None and np.isfinite(sigma) and sigma > 0.0) else None
    seed_value = seed

    labels: np.ndarray | None = None
    centroids: np.ndarray | None = None
    inertia = float("nan")
    iterations = 0
    sigma_used_report: float | None = sigma_arg
    knn_used_report = 0

    while True:
        if mode == "spectral":
            if np.allclose(js_distances, 0.0):
                messages.append("JS distances are degenerate; falling back to bregman clustering.")
                mode = "bregman"
                continue

            spectral_labels, working_knn = _spectral_kmeans_labels(
                js_distances,
                n_clusters=k,
                knn=knn,
                sigma=sigma_arg,
                random_state=seed_value,
            )
            if spectral_labels is None or np.unique(spectral_labels).size < k:
                messages.append("Warning: spectral clustering failed to partition data; falling back to bregman.")
                mode = "bregman"
                continue
            labels = spectral_labels
            centroids = _compute_centroids(labels, normalized_data)
            iterations = 1
            knn_used_report = working_knn
            sigma_used_report = sigma_arg
            break

        if mode == "hierarchical":
            try:
                hier_raw = _hierarchical_js_clustering(js_distances, k, linkage_method=linkage_method)
            except Exception as exc:  # pragma: no cover - defensive
                messages.append(f"Hierarchical clustering failed ({exc}); falling back to bregman.")
                mode = "bregman"
                continue
            labels, _ = _map_labels(hier_raw)
            centroids = _compute_centroids(labels, normalized_data)
            iterations = 1
            inertia = float("nan")
            messages.append(
                f"Hierarchical clustering produced {np.unique(labels).size} cluster(s) using {linkage_method} linkage."
            )
            sigma_used_report = None
            break

        if mode == "dbscan":
            if DBSCAN is None:
                messages.append("DBSCAN unavailable (scikit-learn missing); falling back to bregman.")
                mode = "bregman"
                continue
            try:
                db_labels_raw, eps_used = _dbscan_js_clustering(
                    js_distances,
                    eps=dbscan_eps,
                    min_samples=max(1, dbscan_min_samples),
                )
            except Exception as exc:  # pragma: no cover - defensive
                messages.append(f"DBSCAN failed ({exc}); falling back to bregman.")
                mode = "bregman"
                continue
            if np.all(db_labels_raw == -1):
                messages.append("DBSCAN produced only noise; falling back to bregman.")
                mode = "bregman"
                continue
            noise = int(np.sum(db_labels_raw == -1))
            cluster_ids = sorted(lab for lab in np.unique(db_labels_raw) if lab >= 0)
            if len(cluster_ids) != k:
                messages.append(
                    f"DBSCAN formed {len(cluster_ids)} usable cluster(s); enforcing k={k} via hierarchical linkage."
                )
                mode = "hierarchical"
                continue

            # If DBSCAN produced noise points (-1), reassign them to the nearest
            # non-noise point by JS distance so downstream code still has exactly
            # k clusters.
            if noise:
                non_noise_idx = np.flatnonzero(db_labels_raw >= 0)
                noise_idx = np.flatnonzero(db_labels_raw == -1)
                nearest_pos = np.argmin(js_distances[np.ix_(noise_idx, non_noise_idx)], axis=1)
                nearest_global = non_noise_idx[nearest_pos]
                db_labels_raw[noise_idx] = db_labels_raw[nearest_global]
                noise = 0

            labels, _ = _map_labels(db_labels_raw)
            centroids = _compute_centroids(labels, normalized_data)
            iterations = 1
            inertia = float("nan")
            sigma_used_report = eps_used
            knn_used_report = 0
            messages.append(
                f"DBSCAN formed {len(cluster_ids)} cluster(s); noise points={noise}; eps={eps_used:g}."
            )
            break

        if mode == "bregman":
            labels, centroids, inertia, iterations = bregman_kmeans_kl(
                data_matrix,
                k,
                max_iter=max_iter,
                tol=tol,
                random_state=seed_value,
            )
            sigma_used_report = sigma_arg
            knn_used_report = 0
            break

        raise ValueError(f"Unsupported cluster mode: {cluster_mode}")

    if labels is None or centroids is None:
        raise RuntimeError("Clustering failed; no assignments computed.")

    return ClusterResult(
        labels=labels,
        centroids=centroids,
        inertia=inertia,
        iterations=iterations,
        normalized_data=normalized_data,
        js_distances=js_distances,
        mode=mode,
        knn_used=knn_used_report,
        sigma_used=sigma_used_report,
        messages=messages,
    )


def _contingency(labels: np.ndarray, classes_num: np.ndarray) -> np.ndarray:
    """Return clusters × classes count matrix."""

    k = int(labels.max()) + 1 if labels.size else 0
    c = int(classes_num.max()) + 1 if classes_num.size else 0
    tbl = np.zeros((k, c), dtype=int)
    for y, cl in zip(labels, classes_num, strict=True):
        tbl[int(y), int(cl)] += 1
    return tbl


def _class1_purity(tbl: np.ndarray, class1_col: int = 1) -> tuple[int, int, float]:
    """Return (dominant_count, total_class1, purity) for class-1."""

    if tbl.size == 0 or class1_col >= tbl.shape[1]:
        return 0, 0, 0.0
    col = tbl[:, class1_col]
    total = int(col.sum())
    dominant = int(col.max()) if total else 0
    purity = 0.0 if total == 0 else dominant / total
    return dominant, total, purity


def _safe_sklearn_metrics(labels: np.ndarray, classes_num: np.ndarray) -> dict[str, float]:
    """Adjusted Rand Index and NMI if scikit-learn is available; NaNs otherwise."""

    try:  # pragma: no cover - optional dependency
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        return {
            "ARI": float(adjusted_rand_score(classes_num, labels)),
            "NMI": float(normalized_mutual_info_score(classes_num, labels, average_method="arithmetic")),
        }
    except Exception:  # pragma: no cover - optional dependency
        return {"ARI": float("nan"), "NMI": float("nan")}


def _permutation_pvalue(
    labels: np.ndarray,
    classes_num: np.ndarray,
    iters: int = PERMUTATION_ITERS_DEFAULT,
    seed: int | None = None,
) -> float:
    """One-sided p-value that the observed max class-1 concentration could occur by chance."""

    if labels.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)  # Permutation test randomness (ARI/NMI) is fully driven by --seed.
    k = int(labels.max()) + 1
    is_c1 = (classes_num == 1).astype(np.int8)
    obs = int(np.bincount(labels[is_c1 == 1], minlength=k).max())
    ge = 0
    for _ in range(int(iters)):
        perm = rng.permutation(is_c1)
        stat = int(np.bincount(labels[perm == 1], minlength=k).max())
        ge += (stat >= obs)
    return (ge + 1) / (iters + 1)


def _class1_cohesion_stats(
    labels: np.ndarray,
    classes_num: np.ndarray,
    iters: int,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Estimate how tightly class-1 members cluster together, independent of other classes.
    Returns observed same-cluster pair fraction plus permutation expectations.
    """

    labels = np.asarray(labels, dtype=np.int64)
    classes_num = np.asarray(classes_num, dtype=np.int8)
    is_c1 = classes_num == 1
    idx = np.flatnonzero(is_c1)
    n_pairs = idx.size * (idx.size - 1) // 2
    if n_pairs <= 0:
        return {
            "pairs": float(n_pairs),
            "observed": float("nan"),
            "expected_mean": float("nan"),
            "expected_std": float("nan"),
            "expected_q95": float("nan"),
            "p_value": float("nan"),
        }

    k = int(labels.max()) + 1
    counts_obs = np.bincount(labels[idx], minlength=k)
    same_pairs_obs = float(np.sum(counts_obs * (counts_obs - 1) // 2))
    observed_frac = same_pairs_obs / n_pairs if n_pairs else float("nan")

    if iters <= 0:
        return {
            "pairs": float(n_pairs),
            "observed": observed_frac,
            "expected_mean": float("nan"),
            "expected_std": float("nan"),
            "expected_q95": float("nan"),
            "p_value": float("nan"),
        }

    rng = np.random.default_rng(seed)  # JS clustering random fallback obeys --seed (mostly deterministic otherwise).
    ge = 0
    samples: list[float] = []
    for _ in range(int(iters)):
        permuted_classes = rng.permutation(classes_num)
        perm_mask = permuted_classes == 1
        perm_idx = np.flatnonzero(perm_mask)
        if perm_idx.size < 2:
            frac = 0.0
        else:
            counts = np.bincount(labels[perm_idx], minlength=k)
            same_pairs = float(np.sum(counts * (counts - 1) // 2))
            frac = same_pairs / n_pairs
        samples.append(frac)
        if frac >= observed_frac:
            ge += 1

    expected_mean = float(np.mean(samples))
    expected_std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    expected_q95 = float(np.quantile(samples, 0.95))
    p_value = (ge + 1) / (iters + 1)

    return {
        "pairs": float(n_pairs),
        "observed": observed_frac,
        "expected_mean": expected_mean,
        "expected_std": expected_std,
        "expected_q95": expected_q95,
        "p_value": p_value,
    }


def _kl_margins(X: np.ndarray, centroids: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """ΔKL = KL(other) - KL(own) for each sample (k>=2). Larger ⇒ more decisive assignment."""

    safe_X = np.clip(_normalize_simplex(X, eps), eps, None)
    H = np.sum(safe_X * np.log(safe_X), axis=1)
    D = _kl_distance_matrix(safe_X, _normalize_simplex(centroids, eps), H, eps)
    if D.shape[1] < 2:
        return np.zeros(D.shape[0], dtype=np.float64)
    own = np.take_along_axis(D, D.argmin(axis=1, keepdims=True), axis=1).ravel()
    other = np.partition(D, 1, axis=1)[:, 1]
    return other - own


def _plot_confusion_and_margins(tbl: np.ndarray, margins: np.ndarray, classes_num: np.ndarray, out_path: Path) -> None:
    """Save a PDF with (clusters × classes) counts and ΔKL margins by class."""

    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE_CONFUSION_MARGINS)
    im = ax[0].imshow(tbl, aspect="auto")
    ax[0].set_title("Counts: clusters × classes")
    ax[0].set_xlabel("class")
    ax[0].set_ylabel("cluster")
    for (i, j), v in np.ndenumerate(tbl):
        ax[0].text(j, i, str(int(v)), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    data0 = margins[classes_num == 0]
    data1 = margins[classes_num == 1]
    ax[1].violinplot([data0, data1], showmedians=True)
    ax[1].set_xticks([1, 2], labels=["class 0", "class 1"])
    ax[1].set_title(r"$\Delta$KL = KL(\mathrm{other}) - KL(\mathrm{own})$")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _run_self_test(seed: int | None = None) -> None:
    """Run a lightweight smoke test for clustering routines."""

    defaults = _PARAM_DEFAULTS
    knn_default = int(defaults["cli_knn"])
    sigma_default = defaults.get("cli_sigma")
    if isinstance(sigma_default, str):
        sigma_default = float(sigma_default)
    sigma_value = None
    if sigma_default is not None and not (isinstance(sigma_default, float) and np.isnan(sigma_default)):
        sigma_value = float(sigma_default)
    dbscan_min_samples = int(defaults.get("cli_dbscan_min_samples", 2))

    rng = np.random.default_rng(seed)  # DBSCAN heuristic sampling obeys the CLI seed.
    samples = []
    ts = np.linspace(0.15, 0.85, 60)
    for t in ts:
        base = np.array([0.6 * t + 0.1, 0.6 * (1 - t) + 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        alpha = 60 * base / base.sum()
        samples.append(rng.dirichlet(alpha))
    ts = np.linspace(0.15, 0.85, 60)
    for t in ts:
        base = np.array([0.05, 0.05, 0.6 * t + 0.1, 0.6 * (1 - t) + 0.1, 0.05, 0.05, 0.05, 0.05])
        alpha = 60 * base / base.sum()
        samples.append(rng.dirichlet(alpha))

    data = np.vstack(samples)
    normalized = _normalize_simplex(data, _EPS)
    js_dist = _js_distance_matrix(normalized)
    affinity = _knn_affinity(js_dist, k=knn_default, sigma=sigma_value)
    spectral_labels, _ = _spectral_kmeans_labels(
        js_dist,
        n_clusters=2,
        knn=knn_default,
        sigma=sigma_value,
        random_state=seed,
    )
    if spectral_labels is None or np.unique(spectral_labels).size < 2:
        raise AssertionError("Spectral clustering failed to split synthetic data.")

    labels, _, inertia, _ = bregman_kmeans_kl(normalized, 2, random_state=seed)
    if np.unique(labels).size < 2:
        raise AssertionError("Bregman clustering failed on synthetic data.")
    if not np.isfinite(inertia):
        raise AssertionError("Bregman clustering returned non-finite inertia.")

    hier_labels = _hierarchical_js_clustering(js_dist, 2, linkage_method="average")
    if np.unique(hier_labels).size < 2:
        raise AssertionError("Hierarchical JS clustering failed to split synthetic data.")

    if DBSCAN is not None:
        db_labels, eps_used = _dbscan_js_clustering(js_dist, min_samples=dbscan_min_samples)
        clusters = {lab for lab in np.unique(db_labels) if lab >= 0}
        if not clusters:
            raise AssertionError("DBSCAN produced no clusters on synthetic data.")
        print(f"Self-test DBSCAN epsilon≈{eps_used:.4f}; clusters={len(clusters)}")
    else:  # pragma: no cover - optional dependency
        print("Self-test DBSCAN skipped (scikit-learn missing).")

    print("Self-test passed: spectral, hierarchical, and Bregman clustering behaving as expected.")


__all__ = [
    "_EPS",
    "format_class_value",
    "_normalize_simplex",
    "_kl_distance_matrix",
    "_js_distance_matrix",
    "_kmeanspp_init_kl",
    "_knn_affinity",
    "_spectral_kmeans_labels",
    "_pca_project",
    "_render_cluster_scatter",
    "_symmetrised_kl_matrix",
    "_mds_embedding",
    "_hierarchical_js_clustering",
    "_dbscan_js_clustering",
    "_map_labels",
    "_compute_centroids",
    "_cluster_distributions",
    "FoldxEnergyDataset",
    "ClusterResult",
    "bregman_kmeans_kl",
    "predict_bregman_kl",
    "_contingency",
    "_class1_purity",
    "_safe_sklearn_metrics",
    "_permutation_pvalue",
    "_class1_cohesion_stats",
    "_kl_margins",
    "_plot_confusion_and_margins",
    "_run_self_test",
]
