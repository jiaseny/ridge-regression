from __future__ import division
from utils import *


class RidgeRegression(object):
    """
    Iterative, sketching-based ridge regression.
    """
    def __init__(self, A, b, lmbd):
        """
        Args:
            A: array((n, d))
            b: array((n, ))
            lmbd: float
        """
        self.n, self.d = A.shape
        assert b.shape == (self.n, ), b

        self.A = deepcopy(A)  # Design matrix
        self.b = deepcopy(b)  # Response vector

        assert lmbd >= 0
        self.lmbd = lmbd

        self.U, self.S, Vh = svd(self.A, full_matrices=False)  # Thin SVD
        self.V = Vh.T

        return

    def direct_solver(self):
        """
        Computes the direct solution to the ridge regression problem.

        Returns:
            x_opt: array, optimal solution vector.
        """
        # Using thin SVD: x_opt = V (S^2 + lmbd I)^{-1} S U^T b
        Utb = self.U.T.dot(self.b)  # (r, )
        D = self.S / (self.S**2 + self.lmbd)  # (r, )
        D_Utb = D * Utb  # (r, )
        x_opt = self.V.dot(D_Utb)  # (d, )
        assert x_opt.shape == (self.d, )

        return x_opt

    def leverage_scores(self):
        """
        Computes the statistical leverage scores of the input matrix A.

        Returns:
            array, (normalized) leverage score sampling probabilities.
        """
        scores = norm(self.V, axis=1)**2
        probs = scores / np.sum(scores)
        assert probs.shape == (self.d, ), probs

        return probs

    def ridge_leverage_scores(self, lmbd=None):
        """
        Computes the ridge leverage scores of the input matrix A.

        Returns:
            array, (normalized) ridge leverage score sampling probabilities.
        """
        if lmbd is None:
            lmbd = self.lmbd

        Sreg = self.S / np.sqrt(self.S**2 + lmbd)  # \Sigma_reg
        scores = norm(self.V.dot(np.diag(Sreg)), axis=1)**2
        probs = scores / np.sum(scores)
        assert probs.shape == (self.d, ), probs

        return probs

    def sampling_matrix(self, probs, num_cols):
        """
        Construct sampling-and-rescaling matrix S.

        Args:
            num_cols: int, number of columns to sample.
            probs: array(d), column-sampling probabilities.

        Returns:
            S: sp.sparse matrix (d, s).
        """
        d = self.d
        check_prob_vector(probs)

        inds = rand.choice(a=d, p=probs, size=num_cols, replace=True)
        vals = 1. / np.sqrt(num_cols * probs[inds])
        S = csc_matrix((vals, (inds, range(num_cols))), shape=(d, num_cols))

        return S

    def iterative_solver(self, num_cols, num_iters, probs):
        """
        Iterative algorithm for ridge regression.

        Args:
            num_cols: int, number of sampled columns.
            num_iters: int, max. number of iterations.
            probs: array(d), column-sampling probabilities.

        Returns:
            x_opt: array, computed solution vector.
            x_hist: array, solution vector computed at each iteration.
        """
        n = self.n
        d = self.d
        A = self.A

        x = np.zeros((num_iters, d))
        y = np.zeros((num_iters, n))
        b = np.zeros((num_iters, n))

        b[0] = self.b
        for i in xrange(1, num_iters):
            if i % 50 == 0:
                print "Iterative solver: iteration %d ..." % i

            if i == 1:
                S = self.sampling_matrix(probs, num_cols)

            b[i] = b[i-1] - self.lmbd * y[i-1] - A.dot(x[i-1])

            SA = S.T.dot(A.T)
            H = SA.T.dot(SA) + self.lmbd * identity(n)
            H_inv = inv(H)  # Could also perform implicit inverse via SVD of SA

            y[i] = H_inv.dot(b[i])
            x[i] = A.T.dot(y[i])

        x_opt = np.sum(x, axis=0)
        x_hist = np.cumsum(x, axis=0)

        return x_opt, x_hist

    def obj_vals(self, x):
        """
        Computes the objective value of the L2-regularized optimization problem.

        Args:
            x: array (d,) or (num_iters, d), solution vector(s).

        Returns:
            float or array(num_iters), evaluated objective value.
        """
        x = np.atleast_2d(x).T   # (d, num_iters)
        resid = self.A.dot(x) - self.b.reshape((self.n, 1))  # (d, num_iters)
        mse = np.sum(resid**2, axis=0)  # (n, )
        reg = self.lmbd * np.sum(x**2, axis=0)

        return mse + reg

    def rel_err(self, x, x_opt):
        """
        Computes the relative error for the iterative solver.

        Args:
            x: array (d,) or (num_iters, d), solution vector(s).
            x_opt: array(d,), optimal solution vector.

        Returns:
            float or array(num_iters), evaluated relative error(s).
        """
        x = np.atleast_2d(x)
        assert x.shape[1] == self.d
        assert len(x_opt) == self.d

        return norm(x - x_opt, axis=1) / norm(x_opt)
