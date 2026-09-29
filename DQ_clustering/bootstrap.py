"""Bootstrap stability analysis helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .algorithms import (
    _EPS,
    ClusterResult,
    FoldxEnergyDataset,
    _kl_distance_matrix,
    _class1_cohesion_stats,
    _class1_purity,
    _contingency,
    _js_distance_matrix,
    _normalize_simplex,
    _permutation_pvalue,
    _safe_sklearn_metrics,
)

from .engines import ClusteringEngine, PreparedData

R_KCAL_PER_MOL_K = 0.0019872042586408316
ENSEMBLE_T_K = 300.0
ENSEMBLE_RT_KCAL = R_KCAL_PER_MOL_K * ENSEMBLE_T_K


def _ensemble_free_energy_kcal(energies: np.ndarray, rt: float = ENSEMBLE_RT_KCAL) -> float:
    """Normalized ensemble free energy in kcal/mol for one rank-energy ensemble."""
    values = np.asarray(energies, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or rt <= 0.0:
        return float("nan")
    e_min = float(np.min(values))
    log_mean_z = np.log(np.mean(np.exp(-(values - e_min) / rt))) - (e_min / rt)
    return float(-rt * log_mean_z)


@dataclass
class BootstrapSummary:
    """Container for aggregated bootstrap results."""

    stats: List[dict[str, object]]
    dominant_hits: np.ndarray
    dominant_trials: np.ndarray
    trials: np.ndarray
    energy_dom_hits: np.ndarray
    energy_dom_trials: np.ndarray
    class1_dom_hits: np.ndarray
    class1_dom_trials: np.ndarray
    class0_same_hits: np.ndarray
    class0_same_trials: np.ndarray
    dist_sum: np.ndarray
    dist_sum_sq: np.ndarray
    dist_trials: np.ndarray
    class1_share_samples: List[float]
    class0_share_samples: List[float]
    dominant_cluster_ids: List[int]
    log_lines: List[str]


class BootstrapAnalyzer:
    """Runs bootstrap replicates using a preconfigured clustering engine."""

    def __init__(
        self,
        engine: ClusteringEngine,
        datasets: List[FoldxEnergyDataset],
        classes_num: np.ndarray,
    ) -> None:
        self.engine = engine
        self.datasets = datasets
        self.classes_num = classes_num
        self.class_values = np.unique(classes_num)
        self.class_index = {int(val): idx for idx, val in enumerate(self.class_values)}

    # ------------------------------------------------------------------
    def _resample_distributions(self, prepared: PreparedData, rng: np.random.Generator) -> np.ndarray:
        if prepared.bin_edges is None:
            return np.tile(
                np.full(self.engine.bins, 1.0 / self.engine.bins, dtype=np.float64),
                (len(self.datasets), 1),
            )

        resampled = []
        for dataset in self.datasets:
            sample = rng.choice(dataset.energies, size=dataset.energies.size, replace=True)
            # Histogram bin edges are shared with the main clustering pass so we
            # do not rebuild them here; this keeps bootstrap resamples both
            # consistent and cheap even for large B.
            counts, _ = np.histogram(sample, bins=prepared.bin_edges, density=False)
            counts = counts.astype(np.float64)
            if self.engine.smoothing > 0.0:
                counts += self.engine.smoothing
            total = counts.sum()
            if total <= 0.0 or not np.isfinite(total):
                raise ValueError(f"Invalid bootstrap histogram for dataset {dataset.label}.")
            resampled.append(counts / total)
        return np.vstack(resampled)

    # ------------------------------------------------------------------
    def run(
        self,
        prepared: PreparedData,
        *,
        mode: str,
        k: int,
        n_bootstrap: int,
        max_iter: int,
        tol: float,
        seed: Optional[int],
        knn: int,
        sigma: Optional[float],
        linkage: str,
        dbscan_eps: Optional[float],
        dbscan_min_samples: int,
        log_path: Optional[Path] = None,
    ) -> BootstrapSummary:
        rng = np.random.default_rng(seed)  # All bootstrap randomness flows through this generator (seeded once).
        n_samples = len(self.datasets)
        class1_mask = self.classes_num == 1
        class0_mask = self.classes_num == 0

        dominant_hits = np.zeros(n_samples, dtype=np.int64)
        dominant_trials = np.zeros(n_samples, dtype=np.int64)
        trials = np.full(n_samples, int(n_bootstrap), dtype=np.int64)
        energy_dom_hits = np.zeros(n_samples, dtype=np.int64)
        energy_dom_trials = np.zeros(n_samples, dtype=np.int64)
        class1_dom_hits = np.zeros(n_samples, dtype=np.int64)
        class1_dom_trials = np.zeros(n_samples, dtype=np.int64)
        class0_same_hits = np.zeros(n_samples, dtype=np.int64)
        class0_same_trials = np.zeros(n_samples, dtype=np.int64)
        # Track per-sample KL distance to its assigned centroid across bootstraps
        dist_sum = np.zeros(n_samples, dtype=np.float64)
        dist_sum_sq = np.zeros(n_samples, dtype=np.float64)
        dist_trials = np.zeros(n_samples, dtype=np.int64)
        class1_share_samples: List[float] = []
        class0_share_samples: List[float] = []
        dominant_cluster_ids: List[int] = []
        bootstrap_stats: List[dict[str, object]] = []
        bootstrap_lines: List[str] = []
        dataset_free_energies = np.array(
            [_ensemble_free_energy_kcal(ds.energies) for ds in self.datasets],
            dtype=np.float64,
        )

        for rep in range(int(n_bootstrap)):
            boot_dist = self._resample_distributions(prepared, rng)
            normalized = _normalize_simplex(boot_dist, _EPS)
            js_distances = _js_distance_matrix(normalized)
            boot_prepared = PreparedData(
                distributions=boot_dist,
                normalized=normalized,
                js_distances=js_distances,
                bin_edges=prepared.bin_edges,
                energy_min=prepared.energy_min,
                energy_max=prepared.energy_max,
            )

            boot_seed = int(rng.integers(0, 2**32 - 1)) if seed is None else seed + rep + 1
            boot_result = self.engine.run(
                boot_prepared,
                mode=mode,
                k=k,
                max_iter=max_iter,
                tol=tol,
                seed=boot_seed,
                knn=knn,
                sigma=sigma,
                linkage=linkage,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
            )
            boot_labels = np.asarray(boot_result.labels, dtype=int)
            # Per-sample KL distance to its assigned centroid
            safe_norm = np.clip(boot_result.normalized_data, _EPS, None)
            data_entropy = np.sum(safe_norm * np.log(safe_norm), axis=1)
            dist_matrix = _kl_distance_matrix(boot_result.normalized_data, boot_result.centroids, data_entropy, _EPS)
            own_dist = dist_matrix[np.arange(n_samples), boot_labels]
            dist_sum += own_dist
            dist_sum_sq += own_dist ** 2
            dist_trials += 1

            tbl_boot = _contingency(boot_labels, self.classes_num)
            dom_boot, tot_boot, _ = _class1_purity(tbl_boot, class1_col=1)
            share_boot = float("nan") if tot_boot == 0 else dom_boot / tot_boot
            cohesion_boot = _class1_cohesion_stats(boot_labels, self.classes_num, iters=0, seed=boot_seed)
            coh_value = cohesion_boot["observed"]
            if not np.isnan(share_boot):
                class1_share_samples.append(share_boot)

            unique_clusters = np.unique(boot_labels)
            cluster_index = {int(lab): idx for idx, lab in enumerate(unique_clusters)}
            counts_matrix = np.zeros((len(unique_clusters), len(self.class_values)), dtype=np.float64)
            for class_idx, class_val in enumerate(self.class_values):
                mask = self.classes_num == class_val
                if not np.any(mask):
                    continue
                for lab, ridx in cluster_index.items():
                    counts_matrix[ridx, class_idx] = np.sum(boot_labels[mask] == lab)

            class1_idx = self.class_index.get(1)
            class0_idx = self.class_index.get(0)
            if class1_idx is not None and counts_matrix[:, class1_idx].sum() > 0.0:
                # Tie policy: every replicate with class-1 members counts as a trial.
                # Hits are recorded only when a unique dominant class-1 cluster exists.
                dominant_trials += 1
                class1_dom_trials[class1_mask] += 1
                class0_same_trials[class0_mask] += 1
                class1_counts = counts_matrix[:, class1_idx]
                max_count = float(np.max(class1_counts))
                dominant_mask = class1_counts == max_count
                if np.sum(dominant_mask) == 1:
                    dominant_local_idx = int(np.argmax(class1_counts))
                    dominant_cluster_ids.append(int(unique_clusters[dominant_local_idx]))
                    mask_dom = (boot_labels == unique_clusters[dominant_local_idx])
                    dominant_hits[mask_dom] += 1
                    class1_dom_hits[mask_dom & class1_mask] += 1
                    class0_same_hits[mask_dom & class0_mask] += 1
                    if class0_idx is not None and counts_matrix[:, class0_idx].sum() > 0.0:
                        frac_c0 = (
                            counts_matrix[dominant_local_idx, class0_idx]
                            / counts_matrix[:, class0_idx].sum()
                        )
                        class0_share_samples.append(float(frac_c0))
                else:
                    # Tie for dominant class-1 cluster: treat as undefined for hits.
                    dominant_cluster_ids.append(-1)
            else:
                dominant_cluster_ids.append(-1)

            # Energy-dominant hits: cluster with lowest median per-molecule
            # normalized ensemble free energy.
            cluster_medians = []
            for lab in unique_clusters:
                free_energy_values = dataset_free_energies[boot_labels == lab]
                free_energy_values = free_energy_values[np.isfinite(free_energy_values)]
                if free_energy_values.size == 0:
                    cluster_medians.append(float("nan"))
                    continue
                cluster_medians.append(float(np.median(free_energy_values)))
            cluster_medians = np.asarray(cluster_medians, dtype=np.float64)
            if np.all(np.isfinite(cluster_medians)):
                energy_dom_trials += 1
                min_val = float(np.min(cluster_medians))
                min_mask = np.isclose(cluster_medians, min_val, rtol=1e-8, atol=1e-12)
                if np.sum(min_mask) == 1:
                    energy_local_idx = int(np.argmin(cluster_medians))
                    mask_energy = (boot_labels == unique_clusters[energy_local_idx])
                    energy_dom_hits[mask_energy] += 1

            bootstrap_stats.append(
                {
                    "mode": boot_result.mode,
                    "share": share_boot,
                    "cohesion": coh_value,
                    "dom_count": dom_boot,
                    "tot_count": tot_boot,
                }
            )

            if log_path is not None:
                line = (
                    f"{rep}\t{boot_result.mode}\t{dom_boot}\t{tot_boot}\t"
                    f"{(share_boot if not np.isnan(share_boot) else float('nan')):.6f}\t{coh_value:.6f}"
                )
                bootstrap_lines.append(line)

        if log_path is not None and bootstrap_lines:
            header = "replicate\tmode\tclass1_dominant\tclass1_total\tdominant_share\tcohesion_observed"
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(header + "\n")
                handle.write("\n".join(bootstrap_lines))

        return BootstrapSummary(
            stats=bootstrap_stats,
            dominant_hits=dominant_hits,
            dominant_trials=dominant_trials,
            trials=trials,
            energy_dom_hits=energy_dom_hits,
            energy_dom_trials=energy_dom_trials,
            class1_dom_hits=class1_dom_hits,
            class1_dom_trials=class1_dom_trials,
            class0_same_hits=class0_same_hits,
            class0_same_trials=class0_same_trials,
            dist_sum=dist_sum,
            dist_sum_sq=dist_sum_sq,
            dist_trials=dist_trials,
            class1_share_samples=class1_share_samples,
            class0_share_samples=class0_share_samples,
            dominant_cluster_ids=dominant_cluster_ids,
            log_lines=bootstrap_lines,
        )


__all__ = ["BootstrapAnalyzer", "BootstrapSummary"]
