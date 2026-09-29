"""Diagnostic plotting surfaces used by the publication CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .mpl_config import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, gamma

from .algorithms import (
    _EPS,
    FoldxEnergyDataset,
    _kl_distance_matrix,
    _mds_embedding,
    _render_cluster_scatter,
    _symmetrised_kl_matrix,
)
from .bootstrap import BootstrapSummary
from .parameters import get_parameter_defaults
from .plot_common import (
    HEATMAP_FIG_H_SCALE,
    HEATMAP_FIG_W_SCALE,
    HEATMAP_MIN_H,
    HEATMAP_MIN_W,
    _add_highlight,
    _format_allele_label,
    _get_cmap_copy,
)

PARAM_DEFAULTS = get_parameter_defaults()
CENTROID_FIG_COLS = int(PARAM_DEFAULTS["plot_centroid_fig_cols"])


def _pca_embeddings_and_loadings(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA scores and loadings for centred X using SVD."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input to PCA must be 2D.")
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros((n_samples, min(n_components, n_features))), np.zeros((n_components, n_features))

    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = min(n_components, Vt.shape[0])
    scores = U[:, :comp] * S[:comp]

    loadings = np.zeros((n_components, n_features), dtype=np.float64)
    loadings[:comp] = Vt[:comp]

    if comp < n_components:
        scores = np.hstack([scores, np.zeros((n_samples, n_components - comp))])
    return scores, loadings


def _render_pca_loadings(
    loadings: np.ndarray,
    output_path: Path,
    *,
    bin_edges: np.ndarray | None = None,
) -> None:
    """Save a bar plot of PCA loadings, one subplot per component."""
    loadings = np.asarray(loadings, dtype=np.float64)
    if loadings.ndim != 2 or loadings.size == 0:
        return

    n_components, n_features = loadings.shape
    fig, axes = plt.subplots(n_components, 1, figsize=(14, 4 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
    use_energy = bin_edges is not None and bin_edges.shape[0] == n_features + 1
    if use_energy:
        widths = np.diff(bin_edges)
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_widths = widths * 0.9
    else:
        x = np.arange(n_features)
        bar_widths = 0.9

    for idx, ax in enumerate(axes):
        ax.bar(
            x,
            loadings[idx],
            color="blue",
            alpha=0.5,
            edgecolor="black",
            linewidth=1.2,
            width=bar_widths,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        if use_energy:
            left = x.min() - 0.5 * bar_widths.max()
            right = x.max() + 0.5 * bar_widths.max()
            ax.set_xlim(left, right)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{val:.1f}" for val in x], rotation=90, ha="center", fontsize=18)
        else:
            ax.set_xlim(-0.5, n_features - 0.5)
            ax.set_xticks(np.arange(0, n_features, 2 if n_features >= 4 else 1))
        ax.tick_params(axis="x", labelsize=18)
        signs = np.sign(loadings[idx])
        sign_flips = [i for i in range(1, n_features) if signs[i] != signs[i - 1]]
        if sign_flips:
            for sign_flip in sign_flips:
                if use_energy:
                    span_half = max(np.min(widths) / 5.0, 1e-6)
                    ax.axvspan(x[sign_flip] - span_half, x[sign_flip] + span_half, color="blue", alpha=0.1, zorder=0)
                else:
                    ax.axvspan(sign_flip - 0.25, sign_flip + 0.25, color="blue", alpha=0.2, zorder=0)
            if not use_energy:
                tick_vals = np.array(ax.get_xticks().tolist() + sign_flips, dtype=float)
                tick_vals = np.unique(np.clip(tick_vals, 0, n_features - 1))
                ax.set_xticks(tick_vals)

        if n_features >= 4:
            xs_dense = np.linspace(x.min(), x.max(), 400)
            if use_energy:
                h = max(np.median(widths) if widths.size else 1.0, (x.max() - x.min()) / 40.0)
            else:
                h = max(1.0, n_features / 20.0)
            diffs = xs_dense[:, None] - x[None, :]
            weights = np.exp(-0.5 * (diffs / h) ** 2)
            weights_sum = weights.sum(axis=1, keepdims=True)
            weights_sum = np.where(weights_sum > 0, weights_sum, 1.0)
            smooth = (weights @ loadings[idx][:, None])[:, 0] / weights_sum[:, 0]
            ax.plot(xs_dense, smooth, color="red", linewidth=2.0, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"PC{idx + 1} loading", fontsize=16)
        ax.set_title(f"Principal component ${idx + 1}$ (PC${idx + 1}$)", fontsize=18, pad=10)

    axes[-1].set_xlabel("Energy midpoint [kcal/mol]" if use_energy else "Common support")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


class DiagnosticPlotter:
    """Generate scatter plots, centroid profiles, and bootstrap heatmaps."""

    def __init__(self, output_dir: Path, suffix: str = "") -> None:
        self.output_dir = output_dir
        self.suffix = suffix.strip()

    def plot_scatters(
        self,
        normalized_data: np.ndarray,
        js_distances: np.ndarray,
        labels: np.ndarray,
        datasets: Sequence[FoldxEnergyDataset],
        bin_edges: np.ndarray | None = None,
        pca_panel_label: str | None = None,
    ) -> list[Line2D] | None:
        """Render PCA, JS-MDS, and KL-MDS cluster maps."""
        pca_axis_labels = ("PC1", "PC2")
        if normalized_data.shape[1] > 2:
            pca_embeddings, pca_loadings = _pca_embeddings_and_loadings(normalized_data, 2)
        else:
            pca_embeddings = normalized_data[:, :2]
            pca_embeddings, pca_loadings = _pca_embeddings_and_loadings(normalized_data, 2)

        distance_embeddings = _mds_embedding(js_distances, 2)
        distance_axis_labels = ("MDS-1 (JS)", "MDS-2 (JS)")

        kl_distances = _symmetrised_kl_matrix(normalized_data)
        kl_embeddings = _mds_embedding(kl_distances, 2)
        kl_axis_labels = ("MDS-1 (KL)", "MDS-2 (KL)")

        suffix = f"_{self.suffix}" if self.suffix else ""
        pca_path = self.output_dir / f"2D_mapping_via_PCA{suffix}.pdf"
        _render_cluster_scatter(
            pca_embeddings,
            labels,
            datasets,
            pca_path,
            pca_axis_labels,
            panel_label=pca_panel_label,
            print_class1_pc1=True,
        )
        print(
            f"\nSaved clustering scatter plot (PCA) to {pca_path}"
            " (PCA: top two eigenvectors of the centred probability vectors)."
        )
        loadings_path = self.output_dir / f"PCA_loadings{suffix}.pdf"
        _render_pca_loadings(pca_loadings, loadings_path, bin_edges=bin_edges)
        print(f"Saved PCA loadings bar plot to {loadings_path}")

        js_path = self.output_dir / f"bregman_clustering_js{suffix}.pdf"
        _render_cluster_scatter(distance_embeddings, labels, datasets, js_path, distance_axis_labels, panel_label=None)
        print(
            f"Saved clustering scatter plot (JS-MDS) to {js_path}"
            " (JS classical MDS: top two eigenvectors of -½·J·D²·J from JS distances)."
        )

        kl_path = self.output_dir / f"2D_mapping_via_KL_MDS{suffix}.pdf"
        _render_cluster_scatter(kl_embeddings, labels, datasets, kl_path, kl_axis_labels, panel_label=None)
        print(
            f"Saved clustering scatter plot (KL-MDS) to {kl_path}"
            " (KL classical MDS: top two eigenvectors of -½·J·D²·J from symmetrised KL distances)."
        )

    def plot_centroid_profiles(
        self,
        centroids: np.ndarray,
        bin_edges: np.ndarray | None,
        energy_min: float,
        energy_max: float,
        labels: np.ndarray,
        classes_num: np.ndarray,
        mode_name: str,
        filename: str = "{mode}_centroid_energy_profiles.pdf",
    ) -> None:
        """Plot centroid energy density profiles with KDE/Gamma overlays."""
        k, n_bins = centroids.shape
        if bin_edges is not None:
            x_edges = bin_edges[:-1]
            widths = np.diff(bin_edges)
            x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bar_kwargs = {"align": "edge", "width": widths}
            x_label = r"$\alpha\beta$-interaction energy [kcal/mol]"
        else:
            x = np.linspace(
                energy_min,
                energy_max if energy_max > energy_min else energy_min + 1.0,
                n_bins,
                endpoint=False,
            )
            width = (x[1] - x[0]) if n_bins > 1 else 1.0
            x_edges = x
            bar_kwargs = {"align": "edge", "width": width}
            x_label = "Bin index"

        rows = int(np.ceil(k / CENTROID_FIG_COLS))
        cols = CENTROID_FIG_COLS if k > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 6.0 * rows), sharex=False, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        legend_handle = None
        legend_ax = None
        gamma_handle = None
        gamma_ax = None
        class1_mask = classes_num == 1
        class1_summary: list[tuple[int, int, int, float]] = []
        gamma_fits: list[tuple[int, float, float, float, float]] = []

        for cid in range(k):
            ax = axes[cid]
            profile_mass = centroids[cid]
            if profile_mass.size == 0:
                continue
            ax.set_title(f"Cluster {cid}", fontsize=15)
            if bin_edges is not None:
                density = np.divide(profile_mass, widths, out=np.zeros_like(profile_mass), where=widths > 0)
                ax.bar(x_edges, density, color="tab:blue", alpha=0.35, edgecolor="black", **bar_kwargs)
                ax.set_xlim(bin_edges[0], bin_edges[-1])
            else:
                width_val = bar_kwargs["width"]
                density = profile_mass / width_val if width_val > 0 else profile_mass
                ax.bar(x_edges, density, color="tab:blue", edgecolor="black", **bar_kwargs)
                if np.isscalar(width_val):
                    ax.set_xlim(x_edges[0], x_edges[-1] + width_val)
                else:
                    ax.set_xlim(x_edges[0], x_edges[-1] + width_val[-1])

            ax.set_xlabel(x_label)
            ax.set_ylabel("Probability density" if cid % cols == 0 else "")

        for idx in range(k, axes.size):
            axes[idx].axis("off")

        for cid in range(k):
            ax = axes[cid]
            profile_mass = centroids[cid]
            if profile_mass.size == 0:
                continue
            x_left, x_right = ax.get_xlim()

            if x.size > 0:
                if bin_edges is not None:
                    density = np.divide(profile_mass, widths, out=np.zeros_like(profile_mass), where=widths > 0)
                    mode_idx = int(np.argmax(density))
                else:
                    mode_idx = int(np.argmax(profile_mass))
                mode_x = x[mode_idx]
                ax.axvline(mode_x, color="tab:blue", linewidth=4.0, alpha=0.5, zorder=2)
                max_ticks = 8
                if x.size > max_ticks:
                    tick_idx = np.linspace(0, x.size - 1, num=max_ticks, dtype=int)
                    tick_vals = x[np.unique(tick_idx)]
                else:
                    tick_vals = x
                tick_vals = np.sort(np.append(tick_vals, mode_x))
                ax.set_xticks(tick_vals)
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()])

            positive_mask = profile_mass > 0
            if np.count_nonzero(positive_mask) >= 1:
                sample_points = x[positive_mask]
                weights = profile_mass[positive_mask]
                member_mask = labels == cid
                total = int(np.sum(member_mask))
                class1 = int(np.sum(class1_mask[member_mask]))
                share1 = (class1 / total) * 100.0 if total > 0 else 0.0
                class1_summary.append((cid, total, class1, share1))
                kde_line = None
                try:
                    kde = gaussian_kde(sample_points, weights=weights)
                    grid = np.linspace(x_left, x_right, 400)
                    kde_vals = kde(grid)
                except Exception:
                    try:
                        grid = np.linspace(x_left, x_right, 400)
                        bandwidth = np.std(sample_points) if sample_points.size > 1 else 0.0
                        if not np.isfinite(bandwidth) or bandwidth <= 0:
                            bandwidth = max((x_right - x_left) / 50.0, 1e-3)
                        diffs = (grid[:, None] - sample_points[None, :]) / bandwidth
                        weights_norm = weights / max(weights.sum(), 1e-12)
                        kde_vals = np.exp(-0.5 * diffs ** 2) @ weights_norm
                        kde_vals = kde_vals / (np.sqrt(2 * np.pi) * bandwidth)
                        (kde_line,) = ax.plot(grid, kde_vals, color="red", linewidth=1.6, zorder=3)
                    except Exception:
                        kde_line = None
                if kde_line is not None and legend_handle is None:
                    legend_handle = kde_line
                    legend_ax = ax

                gamma_line = None
                try:
                    shift = np.min(sample_points)
                    shifted = sample_points - shift
                    valid = np.isfinite(shifted)
                    shifted = shifted[valid]
                    weight_valid = weights[valid]
                    if shifted.size >= 2:
                        mean = np.average(shifted, weights=weight_valid)
                        var = np.average((shifted - mean) ** 2, weights=weight_valid)
                        if not np.isfinite(var) or var <= 0:
                            var = max(mean * 1e-3, 1e-6)
                        if mean > 0 and var > 0:
                            shape = (mean ** 2) / var
                            scale = var / mean
                            if np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0:
                                gamma_grid = np.linspace(x_left, x_right, 400)
                                gamma_vals = gamma.pdf(gamma_grid, a=shape, loc=shift, scale=scale)
                                (gamma_line,) = ax.plot(
                                    gamma_grid,
                                    gamma_vals,
                                    color="magenta",
                                    linewidth=4,
                                    alpha=0.5,
                                    linestyle="-",
                                    zorder=3,
                                )
                                gamma_mean = shift + shape * scale
                                gamma_std = np.sqrt(shape) * scale
                                gamma_fits.append((cid, float(shape), float(scale), float(gamma_mean), float(gamma_std)))
                except Exception:
                    gamma_line = None
                if gamma_line is not None and gamma_handle is None:
                    gamma_handle = gamma_line
                    gamma_ax = ax

        handles = []
        labels_list = []
        if legend_handle is not None:
            handles.append(legend_handle)
            labels_list.append("KDE fit")
        if gamma_handle is not None:
            handles.append(gamma_handle)
            labels_list.append("Gamma")
        if handles:
            target_ax = legend_ax if legend_ax is not None else gamma_ax
            if target_ax is not None:
                target_ax.legend(
                    handles=handles,
                    labels=labels_list,
                    loc="upper right",
                    fontsize=15,
                    frameon=True,
                    fancybox=True,
                    framealpha=1.0,
                    edgecolor="black",
                    facecolor="white",
                    borderpad=0.6,
                    borderaxespad=0.4,
                )

        if class1_summary:
            best_cid, best_total, best_c1, best_share = max(class1_summary, key=lambda item: item[3])
            if best_total > 0 and best_c1 > 0:
                print(
                    f"Centroid profiles: cluster {best_cid} attracts the most class-1 samples "
                    f"({best_c1}/{best_total} = {best_share:.2f}%)."
                )
            else:
                print("Centroid profiles: no class-1 samples assigned to any cluster.")
        if gamma_fits:
            print("Centroid profiles: Gamma fits per cluster (shape k, scale theta, mean, std):")
            for cid, shape, scale, mean, std in gamma_fits:
                print(f"  cluster {cid}: k={shape:.4f}, theta={scale:.4f}, mean={mean:.4f}, std={std:.4f}")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_name = filename.format(mode=mode_name.lower())
        if self.suffix:
            if out_name.lower().endswith(".pdf"):
                out_name = out_name[:-4] + f"_{self.suffix}.pdf"
            else:
                out_name = f"{out_name}_{self.suffix}"
        fig.savefig(self.output_dir / out_name, format="pdf")
        plt.close(fig)
        print(f"Saved centroid profiles to {self.output_dir / out_name}")

    def plot_distance_heatmap(
        self,
        normalized_data: np.ndarray,
        centroids: np.ndarray,
        labels: np.ndarray,
        datasets: Sequence[FoldxEnergyDataset],
        classes_num: np.ndarray,
        parent_labels: Sequence[str],
        sub_labels: Sequence[str],
        filename: str = "distance_to_centroid_heatmap.pdf",
    ) -> None:
        """Plot per-sample KL distance to the assigned centroid as a heatmap."""
        if normalized_data.size == 0 or centroids.size == 0:
            return

        safe = np.clip(normalized_data, _EPS, None)
        data_entropy = np.sum(safe * np.log(safe), axis=1)
        all_dists = _kl_distance_matrix(normalized_data, centroids, data_entropy, _EPS)
        per_sample = all_dists[np.arange(labels.size), labels]

        parent_index = {parent: idx for idx, parent in enumerate(parent_labels)}
        sub_index = {sub: idx for idx, sub in enumerate(sub_labels)}
        formatted_parents = [_format_allele_label(lbl) for lbl in parent_labels]
        formatted_subs = [_format_allele_label(lbl) for lbl in sub_labels]

        mat = np.full((len(sub_labels), len(parent_labels)), np.nan, dtype=float)
        highlights: set[tuple[int, int]] = set()
        for idx, dataset in enumerate(datasets):
            label = dataset.label
            if "_" not in label:
                continue
            parent, sub = label.split("_", 1)
            row = sub_index.get(sub)
            col = parent_index.get(parent)
            if row is None or col is None:
                continue
            mat[row, col] = per_sample[idx]
            if idx < classes_num.size and classes_num[idx] == 1:
                highlights.add((row, col))

        finite_vals = mat[np.isfinite(mat)]
        if finite_vals.size == 0:
            return
        vmax = float(np.nanpercentile(finite_vals, 95)) if finite_vals.size else float(np.nanmax(mat))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else float(np.nanmax(finite_vals))

        n_par = max(1, len(parent_labels))
        n_sub = max(1, len(sub_labels))
        fig_w = max(HEATMAP_MIN_W, HEATMAP_FIG_W_SCALE * n_par * 1.2)
        fig_h = max(HEATMAP_MIN_H, HEATMAP_FIG_H_SCALE * n_sub * 0.6)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        cmap = _get_cmap_copy("magma")
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(parent_labels)))
        ax.set_xticklabels(formatted_parents, rotation=90, ha="center", fontsize=16)
        ax.set_yticks(np.arange(len(sub_labels)))
        ax.set_yticklabels(formatted_subs, fontsize=16)
        ax.set_xlabel(r"HLA-DQA1$^*$", fontsize=16)
        ax.set_ylabel(r"HLA-DQB1$^*$", fontsize=16)
        ax.set_title("KL distance to assigned centroid", fontsize=17)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)

        for row, col in highlights:
            _add_highlight(ax, row, col, lw=1.1)

        fig.tight_layout()
        out_name = filename
        if self.suffix:
            if out_name.lower().endswith(".pdf"):
                out_name = out_name[:-4] + f"_{self.suffix}.pdf"
            else:
                out_name = f"{out_name}_{self.suffix}"
        output_path = self.output_dir / out_name
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        print(f"Saved distance-to-centroid heatmap to {output_path}")

    def plot_bootstrap_heatmaps(
        self,
        summary: BootstrapSummary,
        datasets: Sequence[FoldxEnergyDataset],
        classes_num: np.ndarray,
        parent_labels: Sequence[str],
        sub_labels: Sequence[str],
        filename: str = "class1_dominant_frequency_heatmap.pdf",
    ) -> None:
        """Render bootstrap dominance heatmaps by allele pair."""
        n_samples = len(datasets)
        freq_mask = summary.trials > 0
        p_hat = np.full(n_samples, np.nan, dtype=np.float64)
        p_hat[freq_mask] = summary.dominant_hits[freq_mask] / summary.trials[freq_mask]

        posterior_mean = np.full(n_samples, np.nan, dtype=np.float64)
        posterior_std = np.full(n_samples, np.nan, dtype=np.float64)
        dist_mean = np.full(n_samples, np.nan, dtype=np.float64)
        dist_std = np.full(n_samples, np.nan, dtype=np.float64)
        if np.any(freq_mask):
            a = summary.dominant_hits[freq_mask].astype(np.float64) + 1.0
            b = (summary.trials[freq_mask] - summary.dominant_hits[freq_mask]).astype(np.float64) + 1.0
            posterior_mean[freq_mask] = a / (a + b)
            posterior_std[freq_mask] = np.sqrt((a * b) / (((a + b) ** 2) * (a + b + 1.0)))
        dist_mask = summary.dist_trials > 0
        dist_mean[dist_mask] = summary.dist_sum[dist_mask] / summary.dist_trials[dist_mask]
        dist_second = np.full(n_samples, np.nan, dtype=np.float64)
        dist_second[dist_mask] = summary.dist_sum_sq[dist_mask] / summary.dist_trials[dist_mask]
        dist_var = dist_second - np.square(dist_mean)
        dist_var = np.where(np.isfinite(dist_var), np.maximum(dist_var, 0.0), np.nan)
        dist_std[dist_mask] = np.sqrt(dist_var[dist_mask])

        freq_percent = 100.0 * p_hat

        parent_index = {parent: idx for idx, parent in enumerate(parent_labels)}
        sub_index = {sub: idx for idx, sub in enumerate(sub_labels)}
        formatted_parents = [_format_allele_label(lbl) for lbl in parent_labels]
        formatted_subs = [_format_allele_label(lbl) for lbl in sub_labels]
        label_lookup = {dataset.label: idx for idx, dataset in enumerate(datasets)}

        freq_matrix = np.full((len(sub_labels), len(parent_labels)), np.nan, dtype=np.float64)
        mean_matrix = np.full_like(freq_matrix, np.nan)
        std_matrix = np.full_like(freq_matrix, np.nan)
        dist_matrix = np.full_like(freq_matrix, np.nan)
        dist_std_matrix = np.full_like(freq_matrix, np.nan)
        highlights: set[tuple[int, int]] = set()

        for dataset in datasets:
            label = dataset.label
            if "_" not in label:
                continue
            parent, sub = label.split("_", 1)
            if parent not in parent_index or sub not in sub_index:
                continue
            idx = label_lookup[label]
            freq_val = freq_percent[idx]
            mean_val = posterior_mean[idx]
            std_val = posterior_std[idx]
            dist_val = dist_mean[idx]
            dist_std_val = dist_std[idx]
            if np.isnan(freq_val):
                continue
            row = sub_index[sub]
            col = parent_index[parent]
            freq_matrix[row, col] = freq_val
            mean_matrix[row, col] = mean_val
            std_matrix[row, col] = std_val
            dist_matrix[row, col] = dist_val
            dist_std_matrix[row, col] = dist_std_val
            if classes_num[idx] == 1:
                highlights.add((row, col))

        masked_freq = np.ma.masked_invalid(freq_matrix)
        masked_mean = np.ma.masked_invalid(mean_matrix)
        masked_std = np.ma.masked_invalid(std_matrix)
        masked_dist = np.ma.masked_invalid(dist_matrix)
        masked_dist_std = np.ma.masked_invalid(dist_std_matrix)
        try:
            std_max_raw = float(np.nanmax(std_matrix))
        except ValueError:
            std_max_raw = float("nan")
        std_max = std_max_raw if np.isfinite(std_max_raw) else np.nan
        dist_finite = dist_matrix[np.isfinite(dist_matrix)]
        dist_max = float(np.nanpercentile(dist_finite, 95)) if dist_finite.size else np.nan
        dist_std_finite = dist_std_matrix[np.isfinite(dist_std_matrix)]
        dist_std_max = float(np.nanpercentile(dist_std_finite, 95)) if dist_std_finite.size else np.nan
        if not np.isfinite(dist_max) or dist_max <= 0.0:
            try:
                dist_max = float(np.nanmax(dist_matrix))
            except ValueError:
                dist_max = float("nan")
        if not np.isfinite(dist_max) or dist_max <= 0.0:
            dist_max = 1.0
        if not np.isfinite(dist_std_max) or dist_std_max <= 0.0:
            try:
                dist_std_max = float(np.nanmax(dist_std_matrix))
            except ValueError:
                dist_std_max = float("nan")
        if not np.isfinite(dist_std_max) or dist_std_max <= 0.0:
            dist_std_max = 1.0
        if not np.isfinite(std_max) or std_max <= 0.0:
            std_max = 1.0

        n_par = max(1, len(parent_labels))
        n_sub = max(1, len(sub_labels))
        fig_w = max(HEATMAP_MIN_W, HEATMAP_FIG_W_SCALE * n_par * 1.4)
        fig_h = max(HEATMAP_MIN_H, HEATMAP_FIG_H_SCALE * n_sub * 0.5)
        fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), sharey=True)

        matrices = (
            (masked_freq, "Freq (% of replicates)", "Frequency (%)", 0.0, 100.0, "viridis"),
            (
                masked_mean,
                r"Posterior mean (Beta(1,1)) $\frac{n+1}{B+2}$",
                "Mean (prob)",
                0.0,
                1.0,
                "viridis",
            ),
            (
                masked_std,
                r"Posterior std (Beta(1,1)) $\sqrt{\frac{(n+1)(B-n+1)}{(B+2)^2(B+3)}}$",
                "Std",
                0.0,
                float(std_max) if np.isfinite(std_max) else 1.0,
                "magma",
            ),
            (
                masked_dist,
                r"Avg KL distance to centroid (bootstrap mean)",
                "Mean KL distance",
                0.0,
                float(dist_max) if np.isfinite(dist_max) else 1.0,
                "magma",
            ),
            (
                masked_dist_std,
                r"KL distance std (bootstrap)",
                "Std (KL distance)",
                0.0,
                float(dist_std_max) if np.isfinite(dist_std_max) else 1.0,
                "magma",
            ),
        )

        tick_fs = 16
        title_fs = 16
        label_fs = 16
        for ax, (mat, title, _cbar_label, vmin, vmax, cmap_name) in zip(axes, matrices):
            cmap = _get_cmap_copy(cmap_name)
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(parent_labels)))
            ax.set_xticklabels(formatted_parents, rotation=90, ha="center", fontsize=tick_fs)
            ax.set_yticks(np.arange(len(sub_labels)))
            ax.set_yticklabels(formatted_subs, fontsize=tick_fs)
            ax.set_xlabel(r"HLA-DQA1$^*$", fontsize=label_fs)
            ax.set_title(title, fontsize=title_fs)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=tick_fs)
            if ax is axes[0]:
                ax.set_ylabel(r"HLA-DQB1$^*$", fontsize=label_fs)

        for ax in axes:
            for row, col in highlights:
                _add_highlight(ax, row, col, lw=0.9)

        fig.tight_layout()
        out_name = filename
        if self.suffix:
            if out_name.lower().endswith(".pdf"):
                out_name = out_name[:-4] + f"_{self.suffix}.pdf"
            else:
                out_name = f"{out_name}_{self.suffix}"
        heatmap_path = self.output_dir / out_name
        fig.savefig(heatmap_path, format="pdf")
        plt.close(fig)
        print("Saved class-1-dominant heatmaps to", heatmap_path)


def plot_centroid_profiles_from_arrays(
    centroids: np.ndarray,
    bin_edges: np.ndarray | None,
    energy_min: float,
    energy_max: float,
    labels: np.ndarray,
    classes_num: np.ndarray,
    mode_name: str,
    output_dir: Path | str,
    *,
    suffix: str = "",
    filename: str = "{mode}_centroid_energy_profiles.pdf",
) -> Path:
    """Standalone helper to plot centroid profiles without building a plotter manually."""
    out_dir = Path(output_dir)
    plotter = DiagnosticPlotter(out_dir, suffix=suffix)
    plotter.plot_centroid_profiles(
        centroids,
        bin_edges,
        energy_min,
        energy_max,
        labels,
        classes_num,
        mode_name,
        filename=filename,
    )
    out_name = filename.format(mode=mode_name.lower())
    if suffix:
        if out_name.lower().endswith(".pdf"):
            out_name = out_name[:-4] + f"_{suffix}.pdf"
        else:
            out_name = f"{out_name}_{suffix}"
    return out_dir / out_name


__all__ = ["DiagnosticPlotter", "plot_centroid_profiles_from_arrays"]
