import numpy as np


class EntropyEstimator:
    """Statistical entropy estimation and information theory calculations.
    
    This class provides static methods for computing various entropy measures
    commonly used in information theory and statistical analysis.
    
    .. versionadded:: 1.0.0
    """
    
    @staticmethod
    def shannon_entropy(distribution: np.ndarray) -> float:
        """Calculate the Shannon entropy of a probability distribution.
        
        Shannon entropy measures the average information content in a message
        or the uncertainty in a probability distribution. It is defined as
        H(X) = -∑ P(x) * log₂(P(x)) for all x where P(x) > 0.
        
        Parameters
        ----------
        distribution : np.ndarray
            A 1D array representing a probability distribution. Values should
            be non-negative and typically sum to 1, though normalization is
            not required for entropy calculation.
            
        Returns
        -------
        float
            The Shannon entropy in bits. Returns 0.0 for deterministic
            distributions (single non-zero probability).
            
        Raises
        ------
        ValueError
            If the distribution array is empty or contains only zeros.
        TypeError
            If the input is not a numpy array or array-like object.
            
        Examples
        --------
        >>> import numpy as np
        >>> dist = np.array([0.5, 0.5])
        >>> EntropyEstimator.shannon_entropy(dist)
        1.0
        
        >>> dist = np.array([1.0, 0.0, 0.0])
        >>> EntropyEstimator.shannon_entropy(dist)
        0.0
        
        Notes
        -----
        Zero probabilities are automatically excluded from the calculation
        to avoid log(0) which is undefined.
        
        .. versionadded:: 1.0.0
        """
        p = distribution[distribution > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def kullback_leibler(P: np.ndarray, Q: np.ndarray) -> float:
        """Calculate the Kullback-Leibler divergence between two distributions.
        
        The Kullback-Leibler (KL) divergence measures the difference between
        two probability distributions P and Q. It is defined as
        KL(P||Q) = ∑ P(x) * log(P(x)/Q(x)) for all x where P(x) > 0 and Q(x) > 0.
        
        Parameters
        ----------
        P : np.ndarray
            The first probability distribution (reference distribution).
            Should be a 1D array with non-negative values.
        Q : np.ndarray
            The second probability distribution (comparison distribution).
            Should be a 1D array with the same shape as P and non-negative values.
            
        Returns
        -------
        float
            The Kullback-Leibler divergence in nats (natural logarithm base).
            Returns 0.0 when P and Q are identical. Always non-negative.
            
        Raises
        ------
        ValueError
            If P and Q have different shapes, are empty, or if Q contains
            zeros where P is non-zero (making KL divergence infinite).
        TypeError
            If inputs are not numpy arrays or array-like objects.
            
        Examples
        --------
        >>> import numpy as np
        >>> P = np.array([0.5, 0.5])
        >>> Q = np.array([0.5, 0.5])
        >>> EntropyEstimator.kullback_leibler(P, Q)
        0.0
        
        >>> P = np.array([1.0, 0.0])
        >>> Q = np.array([0.5, 0.5])
        >>> EntropyEstimator.kullback_leibler(P, Q)
        0.6931471805599453
        
        Notes
        -----
        The KL divergence is not symmetric: KL(P||Q) ≠ KL(Q||P) in general.
        Only elements where both P and Q are positive contribute to the sum.
        
        The natural logarithm is used in this implementation, so results
        are in nats. To convert to bits, multiply by log₂(e) ≈ 1.4427.
        
        .. versionadded:: 1.0.0
        """
        mask = (P > 0) & (Q > 0)
        return np.sum(P[mask] * np.log(P[mask] / Q[mask]))
