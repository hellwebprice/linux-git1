import numpy as np
from scipy.special import expit


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError("Grad oracle is not implemented.")


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def _func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w without l2 regularization; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        logits = X @ w[1:] + w[0]
        return np.logaddexp(0, -logits * y).mean()

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        return self._func(X, y, w) + self.l2_coef * w[1:] @ w[1:]

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        logits = X @ w[1:] + w[0]
        temp = -expit(-logits * y) * y
        return np.r_[
            np.mean(temp),
            temp @ X / X.shape[0] + 2 * self.l2_coef * w[1:],
        ]
