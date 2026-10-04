"""
fastica_core.py
===============

FastICA Core v1.0
-----------------
A reusable NumPy implementation of symmetric/parallel FastICA.

Scope of this module
--------------------
This file intentionally implements ONLY the core ICA responsibilities:

    X -> independent components S
    S -> reconstructed sensor/channel data X_hat

It does NOT:
- detect artifact ICs
- classify ICs
- remove artifact ICs
- filter EEG
- detect bad channels
- interpolate channels
- re-reference EEG

Those tasks should live in separate preprocessing / artifact modules.

Data convention
---------------
X.shape == (n_channels, n_samples)

Mathematical pipeline
---------------------
1. Select training samples (optional)
2. Center:
       Xc = X - mean
3. Covariance:
       Cx = Xc Xc^T / T
4. Eigendecomposition:
       Cx = E D E^T
5. Numerical-rank selection
6. Whitening:
       Z = D^(-1/2) E^T Xc
7. Symmetric FastICA:
       Y = W Z
       W_new = E[g(Y) Z^T] - diag(E[g'(Y)]) W
       W_new <- (W_new W_new^T)^(-1/2) W_new
8. Total unmixing:
       W_total = W_ICA K
9. Mixing / reconstruction:
       A = K_dewhite W_ICA^T
       X_hat = A S + mean

Dependencies
------------
NumPy only.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import numpy as np

__version__ = "1.0.0"

Array = np.ndarray
Nonlinearity = Callable[[Array], Tuple[Array, Array]]


class FastICAFromScratch:
    """
    Symmetric / parallel FastICA implemented from scratch with NumPy.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of ICA components to estimate.
        If None, use the estimated numerical rank of the training data.

    max_iter : int, default=2000
        Maximum number of fixed-point iterations.

    tol : float, default=1e-6
        Convergence tolerance:

            max_i | |w_i(new)^T w_i(old)| - 1 | < tol

    fun : {"logcosh", "cube", "exp"} or callable, default="logcosh"
        Nonlinearity used by FastICA.

        A custom callable must accept Y and return:
            G, G_prime
        with both arrays having the same shape as Y.

    alpha : float, default=1.0
        Scale for logcosh:
            g(u) = tanh(alpha * u)

    max_train_samples : int or None, default=None
        If not None and X has more samples than this value, randomly
        select this many samples for BOTH whitening estimation and
        FastICA optimization.

        The learned transform can then be applied to the complete X
        using transform().

    random_state : int or None, default=42
        Random seed.

    rank_tol : float or None, default=None
        Absolute covariance-eigenvalue threshold for numerical rank.
        If None:

            eps * max(X_train.shape) * largest_eigenvalue

    decorrelation_eps : float, default=1e-12
        Eigenvalue floor used in symmetric decorrelation.

    verbose : bool, default=True
        Print fit progress.

    print_every : int, default=25
        Print convergence every N iterations when verbose=True.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        max_iter: int = 2000,
        tol: float = 1e-6,
        fun: Union[str, Nonlinearity] = "logcosh",
        alpha: float = 1.0,
        max_train_samples: Optional[int] = None,
        random_state: Optional[int] = 42,
        rank_tol: Optional[float] = None,
        decorrelation_eps: float = 1e-12,
        verbose: bool = True,
        print_every: int = 25,
    ):
        self.n_components = n_components
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fun = fun
        self.alpha = float(alpha)
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.rank_tol = rank_tol
        self.decorrelation_eps = float(decorrelation_eps)
        self.verbose = bool(verbose)
        self.print_every = int(print_every)

        self._is_fitted = False

    # ================================================================
    # Validation
    # ================================================================

    @staticmethod
    def _validate_X(X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "X must be 2-D with shape (n_channels, n_samples). "
                f"Received {X.shape}."
            )

        if X.shape[0] < 2:
            raise ValueError("At least two channels/features are required.")

        if X.shape[1] < 3:
            raise ValueError("At least three samples are required.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf.")

        return X

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("The ICA model is not fitted. Call fit() first.")

    # ================================================================
    # Training sample selection
    # ================================================================

    def _select_training_samples(self, X: Array) -> Tuple[Array, Array]:
        n_samples = X.shape[1]

        if self.max_train_samples is None or self.max_train_samples >= n_samples:
            idx = np.arange(n_samples, dtype=np.int64)
            return X, idx

        if int(self.max_train_samples) < 3:
            raise ValueError("max_train_samples must be >= 3.")

        rng = np.random.default_rng(self.random_state)

        idx = rng.choice(
            n_samples,
            size=int(self.max_train_samples),
            replace=False,
        )
        idx.sort()

        return X[:, idx], idx

    # ================================================================
    # Centering
    # ================================================================

    @staticmethod
    def _center_fit(X_train: Array) -> Tuple[Array, Array]:
        mean = X_train.mean(axis=1, keepdims=True)
        Xc = X_train - mean
        return Xc, mean

    # ================================================================
    # Whitening
    # ================================================================

    def _fit_whitening(self, Xc_train: Array) -> Array:
        n_channels, n_samples = Xc_train.shape

        # Cx = Xc Xc^T / T
        covariance = (Xc_train @ Xc_train.T) / n_samples

        # Symmetric eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Largest -> smallest
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        lambda_max = float(eigenvalues[0])

        if lambda_max <= 0:
            raise ValueError(
                "Largest covariance eigenvalue is not positive; "
                "whitening is impossible."
            )

        # Numerical rank
        if self.rank_tol is None:
            eps = np.finfo(np.float64).eps
            rank_tol = eps * max(Xc_train.shape) * lambda_max
        else:
            rank_tol = float(self.rank_tol)

        valid = eigenvalues > rank_tol
        estimated_rank = int(np.sum(valid))

        if estimated_rank < 1:
            raise ValueError("Estimated numerical rank is zero.")

        if self.n_components is None:
            n_components = estimated_rank
        else:
            n_components = int(self.n_components)

            if n_components < 1:
                raise ValueError("n_components must be >= 1.")

            if n_components > estimated_rank:
                raise ValueError(
                    f"n_components={n_components} exceeds "
                    f"estimated rank={estimated_rank}."
                )

        eigvals_keep = eigenvalues[:n_components]
        eigvecs_keep = eigenvectors[:, :n_components]

        # K = D^(-1/2) E^T
        whitening = (
            np.diag(1.0 / np.sqrt(eigvals_keep))
            @ eigvecs_keep.T
        )

        # K_dewhite = E D^(1/2)
        dewhitening = (
            eigvecs_keep
            @ np.diag(np.sqrt(eigvals_keep))
        )

        Z_train = whitening @ Xc_train

        # Diagnostics / fitted attributes
        self.covariance_ = covariance
        self.eigenvalues_all_ = eigenvalues
        self.eigenvectors_all_ = eigenvectors
        self.eigenvalues_ = eigvals_keep
        self.eigenvectors_ = eigvecs_keep
        self.estimated_rank_ = estimated_rank
        self.rank_tol_ = rank_tol
        self.n_components_ = n_components
        self.whitening_ = whitening
        self.dewhitening_ = dewhitening

        Cz = (Z_train @ Z_train.T) / Z_train.shape[1]
        self.whitening_error_ = float(
            np.max(np.abs(Cz - np.eye(n_components)))
        )

        return Z_train

    # ================================================================
    # Nonlinearity
    # ================================================================

    def _nonlinearity(self, Y: Array) -> Tuple[Array, Array]:
        if callable(self.fun):
            G, G_prime = self.fun(Y)

            G = np.asarray(G, dtype=np.float64)
            G_prime = np.asarray(G_prime, dtype=np.float64)

            if G.shape != Y.shape or G_prime.shape != Y.shape:
                raise ValueError(
                    "A custom nonlinearity must return "
                    "(G, G_prime) with the same shape as Y."
                )

            return G, G_prime

        if self.fun == "logcosh":
            G = np.tanh(self.alpha * Y)
            G_prime = self.alpha * (1.0 - G**2)
            return G, G_prime

        if self.fun == "cube":
            G = Y**3
            G_prime = 3.0 * Y**2
            return G, G_prime

        if self.fun == "exp":
            exp_term = np.exp(-(Y**2) / 2.0)
            G = Y * exp_term
            G_prime = (1.0 - Y**2) * exp_term
            return G, G_prime

        raise ValueError(
            "fun must be 'logcosh', 'cube', 'exp', "
            "or a callable returning (G, G_prime)."
        )

    # ================================================================
    # Symmetric decorrelation
    # ================================================================

    def _symmetric_decorrelation(self, W: Array) -> Array:
        # M = W W^T
        M = W @ W.T

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        eigenvalues = np.maximum(
            eigenvalues,
            self.decorrelation_eps,
        )

        # M^(-1/2)
        M_inv_sqrt = (
            eigenvectors
            @ np.diag(1.0 / np.sqrt(eigenvalues))
            @ eigenvectors.T
        )

        return M_inv_sqrt @ W

    # ================================================================
    # Fit
    # ================================================================

    def fit(self, X: Array) -> "FastICAFromScratch":
        """
        Fit FastICA.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        self
        """
        X = self._validate_X(X)

        self.n_channels_in_ = X.shape[0]
        self.n_samples_in_ = X.shape[1]

        # Same subset is used for centering, whitening, and ICA fitting.
        X_train, train_indices = self._select_training_samples(X)

        self.train_indices_ = train_indices
        self.n_train_samples_ = X_train.shape[1]

        Xc_train, mean = self._center_fit(X_train)
        self.mean_ = mean

        Z_train = self._fit_whitening(Xc_train)

        r = self.n_components_

        rng = np.random.default_rng(self.random_state)

        W = rng.normal(size=(r, r))
        W = self._symmetric_decorrelation(W)

        convergence_history = []
        converged = False

        if self.verbose:
            print("=" * 62)
            print("FastICAFromScratch v1.0")
            print("=" * 62)
            print(f"Input shape           : {X.shape}")
            print(f"Training shape        : {X_train.shape}")
            print(f"Estimated rank        : {self.estimated_rank_}")
            print(f"ICA components        : {self.n_components_}")
            print(f"Whitening max error   : {self.whitening_error_:.3e}")
            print("-" * 62)

        # Fixed-point iterations
        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Y = W Z
            Y = W @ Z_train

            G, G_prime = self._nonlinearity(Y)

            # E[g(Y) Z^T]
            term1 = (G @ Z_train.T) / Z_train.shape[1]

            # diag(E[g'(Y)]) W
            term2 = G_prime.mean(axis=1)[:, None] * W

            # FastICA update
            W_raw = term1 - term2

            # Keep rows orthonormal
            W_new = self._symmetric_decorrelation(W_raw)

            # Sign-invariant convergence measure
            alignment = np.diag(W_new @ W_old.T)
            lim = float(
                np.max(
                    np.abs(
                        np.abs(alignment) - 1.0
                    )
                )
            )

            convergence_history.append(lim)
            W = W_new

            if self.verbose and (
                iteration == 0
                or (iteration + 1) % self.print_every == 0
                or lim < self.tol
            ):
                print(
                    f"Iteration {iteration + 1:4d} "
                    f"| convergence = {lim:.8e}"
                )

            if lim < self.tol:
                converged = True
                break

        # Final transforms
        self.W_ica_ = W

        # Total unmixing:
        # S = W_ICA K (X - mean)
        self.unmixing_ = W @ self.whitening_
        self.components_ = self.unmixing_

        # Mixing:
        # X_centered ~= A S
        self.mixing_ = self.dewhitening_ @ W.T

        self.convergence_history_ = np.asarray(
            convergence_history,
            dtype=np.float64,
        )

        self.n_iter_ = iteration + 1
        self.converged_ = converged
        self.final_convergence_ = float(convergence_history[-1])

        self.orthogonality_error_ = float(
            np.max(
                np.abs(
                    W @ W.T - np.eye(r)
                )
            )
        )

        self._is_fitted = True

        if self.verbose:
            print("-" * 62)
            print(f"Converged              : {self.converged_}")
            print(f"Iterations             : {self.n_iter_}")
            print(
                f"Final convergence      : "
                f"{self.final_convergence_:.8e}"
            )
            print(
                f"Orthogonality error    : "
                f"{self.orthogonality_error_:.3e}"
            )
            print("=" * 62)

        return self

    # ================================================================
    # Transform
    # ================================================================

    def transform(self, X: Array) -> Array:
        """
        Transform sensor/channel data to ICA source space.

        Returns
        -------
        S : ndarray, shape (n_components, n_samples)
        """
        self._check_fitted()

        X = self._validate_X(X)

        if X.shape[0] != self.n_channels_in_:
            raise ValueError(
                f"Expected {self.n_channels_in_} channels/features, "
                f"received {X.shape[0]}."
            )

        Xc = X - self.mean_

        return self.unmixing_ @ Xc

    # ================================================================
    # Fit + transform
    # ================================================================

    def fit_transform(self, X: Array) -> Array:
        """
        Fit the model, then transform the complete input X.
        """
        self.fit(X)
        return self.transform(X)

    # ================================================================
    # Reconstruction
    # ================================================================

    def inverse_transform(self, S: Array) -> Array:
        """
        Reconstruct channel/sensor data from ICA source activations.

        Parameters
        ----------
        S : ndarray, shape (n_components, n_samples)

        Returns
        -------
        X_hat : ndarray, shape (n_channels, n_samples)
        """
        self._check_fitted()

        S = np.asarray(S, dtype=np.float64)

        if S.ndim != 2:
            raise ValueError(
                "S must be 2-D with shape "
                "(n_components, n_samples)."
            )

        if S.shape[0] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} components, "
                f"received {S.shape[0]}."
            )

        return self.mixing_ @ S + self.mean_

    # ================================================================
    # Diagnostics
    # ================================================================

    def reconstruction_error(self, X: Array) -> float:
        """
        Relative L2 reconstruction error using ALL retained ICs.

        If n_components < numerical rank, a nonzero reconstruction error
        is expected because dimensionality was intentionally reduced.
        """
        self._check_fitted()

        X = self._validate_X(X)

        S = self.transform(X)
        X_hat = self.inverse_transform(S)

        denominator = np.linalg.norm(X)

        if denominator == 0:
            return 0.0

        return float(
            np.linalg.norm(X - X_hat)
            / denominator
        )

    def get_summary(self) -> dict:
        """
        Return core fitted-model diagnostics.
        """
        self._check_fitted()

        return {
            "version": __version__,
            "n_channels_in": self.n_channels_in_,
            "n_samples_in": self.n_samples_in_,
            "n_train_samples": self.n_train_samples_,
            "estimated_rank": self.estimated_rank_,
            "n_components": self.n_components_,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "final_convergence": self.final_convergence_,
            "whitening_error": self.whitening_error_,
            "orthogonality_error": self.orthogonality_error_,
        }

    # ================================================================
    # Save / load fitted transform
    # ================================================================

    def save(self, path: str) -> None:
        """
        Save the fitted numerical ICA transform to a compressed .npz file.

        This saves what is needed for later transform(),
        inverse_transform(), and reconstruction_error() use.

        A custom Python callable used as `fun` is not serialized.
        """
        self._check_fitted()

        np.savez_compressed(
            path,
            mean=self.mean_,
            whitening=self.whitening_,
            dewhitening=self.dewhitening_,
            W_ica=self.W_ica_,
            unmixing=self.unmixing_,
            mixing=self.mixing_,
            eigenvalues=self.eigenvalues_,
            train_indices=self.train_indices_,
            convergence_history=self.convergence_history_,
            n_channels_in=np.array(self.n_channels_in_),
            n_samples_in=np.array(self.n_samples_in_),
            n_train_samples=np.array(self.n_train_samples_),
            estimated_rank=np.array(self.estimated_rank_),
            n_components=np.array(self.n_components_),
            n_iter=np.array(self.n_iter_),
            converged=np.array(self.converged_),
            final_convergence=np.array(self.final_convergence_),
            whitening_error=np.array(self.whitening_error_),
            orthogonality_error=np.array(self.orthogonality_error_),
            rank_tol=np.array(self.rank_tol_),
            version=np.array(__version__),
        )

    @classmethod
    def load(cls, path: str) -> "FastICAFromScratch":
        """
        Load a fitted transform saved by save().
        """
        data = np.load(path, allow_pickle=False)

        model = cls(verbose=False)

        model.mean_ = data["mean"]
        model.whitening_ = data["whitening"]
        model.dewhitening_ = data["dewhitening"]
        model.W_ica_ = data["W_ica"]
        model.unmixing_ = data["unmixing"]
        model.components_ = model.unmixing_
        model.mixing_ = data["mixing"]
        model.eigenvalues_ = data["eigenvalues"]
        model.train_indices_ = data["train_indices"]
        model.convergence_history_ = data["convergence_history"]

        model.n_channels_in_ = int(data["n_channels_in"].item())
        model.n_samples_in_ = int(data["n_samples_in"].item())
        model.n_train_samples_ = int(data["n_train_samples"].item())
        model.estimated_rank_ = int(data["estimated_rank"].item())
        model.n_components_ = int(data["n_components"].item())
        model.n_iter_ = int(data["n_iter"].item())
        model.converged_ = bool(data["converged"].item())
        model.final_convergence_ = float(
            data["final_convergence"].item()
        )
        model.whitening_error_ = float(
            data["whitening_error"].item()
        )
        model.orthogonality_error_ = float(
            data["orthogonality_error"].item()
        )
        model.rank_tol_ = float(data["rank_tol"].item())

        model._is_fitted = True

        return model
