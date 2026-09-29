"""Clustering engine wrapper built on top of the original implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .algorithms import (
    _EPS,
    ClusterResult,
    _cluster_distributions,
    _js_distance_matrix,
    _normalize_simplex,
)


def _build_histograms(
    datasets,
    bins: int,
    smoothing: float,
) -> tuple[np.ndarray, Optional[np.ndarray], float, float]:
    all_energies = np.concatenate([dataset.energies for dataset in datasets])
    energy_min = float(np.min(all_energies))
    energy_max = float(np.max(all_energies))

    if energy_min == energy_max:
        distributions = [
            np.full(bins, 1.0 / bins, dtype=np.float64)
            for _ in datasets
        ]
        bin_edges = None
    else:
        bin_edges = np.linspace(energy_min, energy_max, bins + 1, dtype=np.float64)
        distributions = []
        for dataset in datasets:
            counts, _ = np.histogram(dataset.energies, bins=bin_edges, density=False)
            counts = counts.astype(np.float64)
            if smoothing > 0.0:
                counts += smoothing
            total = counts.sum()
            if total <= 0.0 or not np.isfinite(total):
                raise ValueError(f"Invalid histogram for dataset {dataset.label}.")
            distributions.append(counts / total)
    return np.vstack(distributions), bin_edges, energy_min, energy_max


@dataclass
class PreparedData:
    distributions: np.ndarray
    normalized: np.ndarray
    js_distances: np.ndarray
    bin_edges: Optional[np.ndarray]
    energy_min: float
    energy_max: float


class ClusteringEngine:
    def __init__(self, bins: int, smoothing: float) -> None:
        self.bins = max(int(bins), 1)
        self.smoothing = max(float(smoothing), 0.0)

    def build_distributions(self, datasets) -> PreparedData:
        distributions, bin_edges, energy_min, energy_max = _build_histograms(
            datasets, self.bins, self.smoothing
        )
        normalized = _normalize_simplex(distributions, _EPS)
        js_distances = _js_distance_matrix(normalized)
        return PreparedData(
            distributions=distributions,
            normalized=normalized,
            js_distances=js_distances,
            bin_edges=bin_edges,
            energy_min=energy_min,
            energy_max=energy_max,
        )

    def run(
        self,
        prepared: PreparedData,
        *,
        mode: str,
        k: int,
        max_iter: int,
        tol: float,
        seed: Optional[int] = None,
        knn: int,
        sigma: Optional[float],
        linkage: str,
        dbscan_eps: Optional[float],
        dbscan_min_samples: int,
    ) -> ClusterResult:
        return _cluster_distributions(
            prepared.distributions,
            cluster_mode=mode,
            k=k,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            knn=knn,
            sigma=sigma,
            linkage_method=linkage,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )


__all__ = ["ClusteringEngine", "PreparedData"]
