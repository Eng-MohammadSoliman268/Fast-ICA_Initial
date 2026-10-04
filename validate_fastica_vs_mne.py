"""
validate_fastica_vs_mne.py
==========================

Validation utilities for comparing FastICAFromScratch against MNE FastICA
on the SAME preprocessed data.

Important
---------
This is an agreement / validation benchmark, not a proof that either
implementation is the ground truth.

Expected input
--------------
X.shape == (n_channels, n_samples)

X should already be preprocessed appropriately for ICA, e.g.:
- bad channels handled/excluded
- severe corrupted segments handled
- suitable high-pass filtering already applied

The benchmark:
1) fits FastICAFromScratch
2) uses the SAME training sample indices for MNE
3) applies both solutions to the complete X
4) computes all pairwise IC time-course correlations
5) performs optimal one-to-one Hungarian matching
6) compares matched scalp/mixing maps
7) reports timing, iterations, and agreement metrics

Dependencies
------------
numpy
scipy
pandas
mne
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import mne

from scipy.optimize import linear_sum_assignment
from mne.preprocessing import ICA

from fastica_core import FastICAFromScratch


def _standardize_rows(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    scale = A.std(axis=1, keepdims=True)
    return A / np.maximum(scale, eps)


def _standardize_columns(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    scale = A.std(axis=0, keepdims=True)
    return A / np.maximum(scale, eps)


def compare_fastica_with_mne(
    X: np.ndarray,
    sfreq: float,
    ch_names: Optional[Sequence[str]] = None,
    *,
    n_components: Optional[int] = None,
    max_train_samples: Optional[int] = 50000,
    max_iter: int = 2000,
    tol: float = 1e-6,
    alpha: float = 1.0,
    random_state: int = 42,
    print_top_n: int = 10,
):
    """
    Compare the custom FastICA core with MNE FastICA.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        PREPROCESSED data to compare on.

    sfreq : float
        Sampling frequency in Hz.

    ch_names : sequence of str or None
        Channel names. If None, generic EEG names are created.

    n_components : int or None
        Number of ICA components for the custom model.
        If None, it uses the estimated numerical rank.
        MNE is then given the same number of components.

    max_train_samples : int or None
        Number of samples used to fit both models.
        Both models use the same indices selected by the custom model.

    Returns
    -------
    result : dict
        Contains fitted models, IC matrices, matching table, and summary.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_channels, n_samples).")

    n_channels, n_samples = X.shape

    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    else:
        ch_names = list(ch_names)

    if len(ch_names) != n_channels:
        raise ValueError(
            f"len(ch_names)={len(ch_names)} does not match "
            f"n_channels={n_channels}."
        )

    # ================================================================
    # 1. OUR FASTICA
    # ================================================================

    print("=" * 70)
    print("1) FASTICA FROM SCRATCH")
    print("=" * 70)

    ours = FastICAFromScratch(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        fun="logcosh",
        alpha=alpha,
        max_train_samples=max_train_samples,
        random_state=random_state,
        verbose=True,
        print_every=25,
    )

    t0 = time.perf_counter()
    ours.fit(X)
    ours_fit_time = time.perf_counter() - t0

    S_ours = ours.transform(X)

    print(f"\nCustom fit time: {ours_fit_time:.3f} s")
    print("Custom source matrix:", S_ours.shape)
    print(
        "Custom reconstruction error:",
        ours.reconstruction_error(X)
    )

    # Use exactly the SAME training samples for MNE.
    train_idx = ours.train_indices_
    X_train = X[:, train_idx]

    # ================================================================
    # 2. MNE FASTICA
    # ================================================================

    print("\n" + "=" * 70)
    print("2) MNE FASTICA")
    print("=" * 70)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=float(sfreq),
        ch_types=["eeg"] * n_channels,
    )

    raw_train = mne.io.RawArray(
        X_train,
        info,
        verbose=False,
    )

    raw_full = mne.io.RawArray(
        X,
        info.copy(),
        verbose=False,
    )

    ica_mne = ICA(
        n_components=ours.n_components_,
        method="fastica",
        random_state=random_state,
        max_iter=max_iter,
        fit_params={
            "algorithm": "parallel",
            "fun": "logcosh",
            "fun_args": {"alpha": alpha},
            "tol": tol,
        },
    )

    t0 = time.perf_counter()
    ica_mne.fit(
        raw_train,
        picks="eeg",
        verbose=False,
    )
    mne_fit_time = time.perf_counter() - t0

    S_mne = ica_mne.get_sources(
        raw_full
    ).get_data()

    print(f"MNE fit time: {mne_fit_time:.3f} s")
    print("MNE iterations:", ica_mne.n_iter_)
    print("MNE source matrix:", S_mne.shape)

    # ================================================================
    # 3. TIME-COURSE CORRELATION
    # ================================================================

    if S_ours.shape[0] != S_mne.shape[0]:
        raise RuntimeError(
            "The two models returned different numbers of components."
        )

    S_ours_std = _standardize_rows(S_ours)
    S_mne_std = _standardize_rows(S_mne)

    R_time = (
        S_ours_std @ S_mne_std.T
    ) / S_ours.shape[1]

    R_time_abs = np.abs(R_time)

    # Optimal one-to-one matching
    ours_idx, mne_idx = linear_sum_assignment(
        -R_time_abs
    )

    matched_time = R_time_abs[
        ours_idx,
        mne_idx
    ]

    signed_time = R_time[
        ours_idx,
        mne_idx
    ]

    # ================================================================
    # 4. SCALP / MIXING-MAP CORRELATION
    # ================================================================

    A_ours = ours.mixing_
    A_mne = ica_mne.get_components()

    if A_ours.shape[0] != A_mne.shape[0]:
        raise RuntimeError(
            "Mixing maps use different channel dimensions."
        )

    A_ours_std = _standardize_columns(A_ours)
    A_mne_std = _standardize_columns(A_mne)

    R_map = (
        A_ours_std.T @ A_mne_std
    ) / A_ours.shape[0]

    R_map_abs = np.abs(R_map)

    matched_map = R_map_abs[
        ours_idx,
        mne_idx
    ]

    signed_map = R_map[
        ours_idx,
        mne_idx
    ]

    # ================================================================
    # 5. MATCHING TABLE
    # ================================================================

    mapping = pd.DataFrame({
        "our_ic": ours_idx,
        "mne_ic": mne_idx,
        "time_corr_signed": signed_time,
        "time_corr_abs": matched_time,
        "map_corr_signed": signed_map,
        "map_corr_abs": matched_map,
    })

    mapping["joint_score"] = (
        mapping["time_corr_abs"]
        * mapping["map_corr_abs"]
    )

    # ================================================================
    # 6. SUMMARY
    # ================================================================

    summary = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "n_train_samples": len(train_idx),
        "n_components": ours.n_components_,
        "custom_converged": ours.converged_,
        "custom_iterations": ours.n_iter_,
        "custom_fit_time_s": ours_fit_time,
        "custom_final_convergence": ours.final_convergence_,
        "custom_whitening_error": ours.whitening_error_,
        "custom_orthogonality_error": ours.orthogonality_error_,
        "custom_reconstruction_error": ours.reconstruction_error(X),
        "mne_iterations": int(ica_mne.n_iter_),
        "mne_fit_time_s": mne_fit_time,
        "mean_time_abs_r": float(matched_time.mean()),
        "median_time_abs_r": float(np.median(matched_time)),
        "min_time_abs_r": float(matched_time.min()),
        "max_time_abs_r": float(matched_time.max()),
        "time_r_ge_095": int(np.sum(matched_time >= 0.95)),
        "time_r_ge_090": int(np.sum(matched_time >= 0.90)),
        "time_r_ge_080": int(np.sum(matched_time >= 0.80)),
        "mean_map_abs_r": float(matched_map.mean()),
        "median_map_abs_r": float(np.median(matched_map)),
        "min_map_abs_r": float(matched_map.min()),
        "max_map_abs_r": float(matched_map.max()),
    }

    # ================================================================
    # 7. REPORT
    # ================================================================

    print("\n" + "=" * 70)
    print("3) AGREEMENT REPORT")
    print("=" * 70)

    print(
        "Mean matched |time correlation|   :",
        summary["mean_time_abs_r"],
    )
    print(
        "Median matched |time correlation| :",
        summary["median_time_abs_r"],
    )
    print(
        "Minimum matched |time correlation|:",
        summary["min_time_abs_r"],
    )
    print(
        "Maximum matched |time correlation|:",
        summary["max_time_abs_r"],
    )

    print(
        f"\n|r_time| >= 0.95: "
        f"{summary['time_r_ge_095']} / {ours.n_components_}"
    )
    print(
        f"|r_time| >= 0.90: "
        f"{summary['time_r_ge_090']} / {ours.n_components_}"
    )
    print(
        f"|r_time| >= 0.80: "
        f"{summary['time_r_ge_080']} / {ours.n_components_}"
    )

    print(
        "\nMean matched |map correlation|    :",
        summary["mean_map_abs_r"],
    )
    print(
        "Median matched |map correlation|  :",
        summary["median_map_abs_r"],
    )

    ordered = mapping.sort_values(
        "time_corr_abs",
        ascending=False,
    )

    n_show = min(print_top_n, len(ordered))

    print(f"\n{n_show} BEST TIME-COURSE MATCHES:")
    print(
        ordered.head(n_show)[
            [
                "our_ic",
                "mne_ic",
                "time_corr_abs",
                "map_corr_abs",
                "joint_score",
            ]
        ].to_string(index=False)
    )

    print(f"\n{n_show} WORST TIME-COURSE MATCHES:")
    print(
        ordered.tail(n_show)[
            [
                "our_ic",
                "mne_ic",
                "time_corr_abs",
                "map_corr_abs",
                "joint_score",
            ]
        ].to_string(index=False)
    )

    print("=" * 70)

    return {
        "ours": ours,
        "mne": ica_mne,
        "S_ours": S_ours,
        "S_mne": S_mne,
        "A_ours": A_ours,
        "A_mne": A_mne,
        "R_time": R_time,
        "R_map": R_map,
        "mapping": mapping,
        "summary": summary,
        "train_indices": train_idx,
    }


if __name__ == "__main__":
    print(
        "Import this module and call compare_fastica_with_mne(...).\n\n"
        "Example:\n"
        "    from validate_fastica_vs_mne import compare_fastica_with_mne\n"
        "    result = compare_fastica_with_mne(\n"
        "        X=X_ica,\n"
        "        sfreq=sfreq,\n"
        "        ch_names=good_channels,\n"
        "        max_train_samples=50000,\n"
        "        max_iter=2000,\n"
        "        tol=1e-6,\n"
        "        random_state=42,\n"
        "    )\n"
    )
