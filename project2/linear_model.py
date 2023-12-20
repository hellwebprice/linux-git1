import time

from collections.abc import Sequence

import numpy as np
from scipy.special import expit


def BatchGenerator(
    list_of_sequences: Sequence[np.ndarray],
    batch_size: int,
    shuffle: bool = False,
):
    """
    Parameters
    ----------
    :param list_of_sequences: Sequence[np.ndarray]
        Sequence of numpy arrays to sampling
    :param batch_size: int
    :param shuffle: bool

    Returns
    -------
    : generator
        Generator of list of batches
    """
    sequence_len = len(list_of_sequences[0])
    if sequence_len <= batch_size:
        yield list_of_sequences
        return

    indeces = np.arange(sequence_len)
    if shuffle:
        np.random.shuffle(indeces)

    for i in range(sequence_len // batch_size):
        yield [
            sequence[indeces[i * batch_size : (i + 1) * batch_size]]
            for sequence in list_of_sequences
        ]
    if sequence_len % batch_size:
        yield [
            sequence[indeces[sequence_len // batch_size * batch_size :]]
            for sequence in list_of_sequences
        ]


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs,
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def lr_schedule(self, epoch: int):
        """
        Parameters
        ----------
        epoch: int

        Returns
        -------
        : float
            Learning rate for this epoch
        """
        return self.step_alpha / epoch**self.step_beta

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        np.random.seed(self.random_seed)

        history = {"time": [], "func": [], "func_val": []}

        if w_0 is None:
            self.w = np.random.normal(scale=1 / (X.shape[1] * 6), size=X.shape[1] + 1)
        else:
            self.w = w_0.copy()

        if X_val is None:
            X_val = X

        if y_val is None:
            y_val = y

        previous_loss = np.inf
        for epoch in range(self.max_iter):
            lr = self.lr_schedule(epoch + 1)

            start = time.time()
            for X_batch, y_batch in BatchGenerator([X, y], self.batch_size, True):
                self.w -= lr * self.loss_function.grad(X_batch, y_batch, self.w)
            epoch_time = time.time() - start

            loss = self.loss_function.func(X, y, self.w)
            if trace:
                history["time"].append(epoch_time)
                history["func"].append(self.loss_function._func(X, y, self.w))
                history["func_val"].append(
                    self.loss_function._func(X_val, y_val, self.w)
                )

            if np.abs(previous_loss - loss) < self.tolerance:
                if trace:
                    print(f"Early stopping on {epoch + 1} epoch")
                break

            previous_loss = loss

        return history

    def predict(self, X, threshold=0):
        """
        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        probabilities = expit(X @ self.w[1:] + self.w[0])
        return np.where(probabilities > threshold, -1, 1)

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function._func(X, y, self.w)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]
