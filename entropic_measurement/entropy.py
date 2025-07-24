import numpy as np
from typing import Sequence, Union

class EntropyEstimator:
    """
    Statistical entropy estimation and information theory calculations.

    .. versionadded:: 1.0.0
    """

    @staticmethod
    def shannon_entropy(
        distribution: Sequence[float],
        *,
        check_normalized: bool = True,
        tol: float = 1e-8
    ) -> float:
        """
        Calculate the Shannon entropy of a probability distribution.

        Parameters
        ----------
        distribution : Sequence[float]
            A 1D array or sequence representing a probability distribution. Values must be non-negative and sum to 1 (unless check_normalized=False).
        check_normalized : bool, optional
            If True, checks that the distribution sums to 1 (default: True).
        tol : float, optional
            Tolerance for normalization checking (default: 1e-8).

        Returns
        -------
        float
            The Shannon entropy in bits.

        Raises
        ------
        ValueError
            If the distribution is empty, contains negatives, is not normalized (when check_normalized=True), or is not 1D.
        TypeError
            If the input is not an array-like sequence of floats.

        Examples
        --------
        >>> EntropyEstimator.shannon_entropy([0.5, 0.5])
        1.0

        .. versionadded:: 1.0.0
        """
        arr = np.asarray(distribution, dtype=float)

        if arr.ndim != 1:
            raise ValueError("Input distribution must be a 1D array or sequence of floats.")

        if arr.size == 0:
            raise ValueError("Input distribution must not be empty.")

        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("Probabilities must be non-negative finite values.")

        if check_normalized and not np.isclose(np.sum(arr), 1.0, atol=tol):
            raise ValueError(
                f"Input probabilities must sum to 1 (got {np.sum(arr)}). "
                "Set check_normalized=False to disable this error."
            )

        p = arr[arr > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def kullback_leibler(
        P: Sequence[float],
        Q: Sequence[float],
        *,
        check_normalized: bool = True,
        tol: float = 1e-8
    ) -> float:
        """
        Calculate the Kullback-Leibler divergence between two distributions.

        Parameters
        ----------
        P : Sequence[float]
            Reference probability distribution. 1D, non-negative, sums to 1.
        Q : Sequence[float]
            Comparison probability distribution. Must match P in shape.
        check_normalized : bool, optional
            If True, checks normalization of inputs (default: True).
        tol : float, optional
            Tolerance for normalization checking (default: 1e-8).

        Returns
        -------
        float
            The Kullback-Leibler divergence (nats, natural log base).

        Raises
        ------
        ValueError
            If P and Q do not have the same shape, are empty, contain negatives, are not normalized, or Q contains zeros where P is non-zero.
        TypeError
            If inputs are not sequences or arrays of floats.

        Examples
        --------
        >>> EntropyEstimator.kullback_leibler([0.5, 0.5], [0.5, 0.5])
        0.0

        .. versionadded:: 1.0.0
        """
        p = np.asarray(P, dtype=float)
        q = np.asarray(Q, dtype=float)

        if p.shape != q.shape:
            raise ValueError("P and Q must be the same shape.")

        if p.ndim != 1 or p.size == 0:
            raise ValueError("Distributions must be 1D and non-empty.")

        if np.any(p < 0) or np.any(q < 0) or not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            raise ValueError("Distributions must contain only non-negative finite values.")

        if check_normalized:
            if not np.isclose(np.sum(p), 1.0, atol=tol):
                raise ValueError(f"P must sum to 1 (got {np.sum(p)}))")
            if not np.isclose(np.sum(q), 1.0, atol=tol):
                raise ValueError(f"Q must sum to 1 (got {np.sum(q)}))")

        mask = (p > 0) & (q > 0)
        if not np.any(mask):
            raise ValueError(
                "Distributions must have at least one shared, strictly positive entry."
            )
        if np.any((p > 0) & (q == 0)):
            raise ValueError("Q contains zero where P is nonzero: KL divergence is infinite.")

        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
