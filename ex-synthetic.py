from __future__ import division
from utils import *
from ridge_regression import RidgeRegression


def expr(A, b, lmbd, num_cols_list, num_iters=50):
    """
    Perform experiments using the iterative ridge regression algorithm
      comparing various sampling methods (uniform, leverage scores, and
      ridge leverage scores) by evaluating the relative errors and
      objective errors of the solutions obtained using each sampling method
      with varying sketch sizes and at each iteration.

    Args:
        A: array, design matrix.
        b: array, response vector.
        num_cols_list: array, list of sketch sizes used in the experiments.
        num_iters: int, maximum number of iterations to run the algorithm.

    Returns:
        rel_errs_unif, rel_errs_levr, rel_errs_rdge: each of type
            array(len(num_cols_list), num_iters)), relative errors.
        obj_errs_unif, obj_errs_levr, obj_errs_rdge: each of type
            array(len(num_cols_list), num_iters)), objective errors.
    """
    model = RidgeRegression(A, b, lmbd=lmbd)
    x_opt = model.direct_solver()
    obj_opt = model.obj_vals(x_opt)

    num_cols_list = map(int, num_cols_list)

    rel_errs_unif = np.zeros((len(num_cols_list), num_iters))
    rel_errs_levr = np.zeros((len(num_cols_list), num_iters))
    rel_errs_rdge = np.zeros((len(num_cols_list), num_iters))

    obj_errs_unif = np.zeros((len(num_cols_list), num_iters))
    obj_errs_levr = np.zeros((len(num_cols_list), num_iters))
    obj_errs_rdge = np.zeros((len(num_cols_list), num_iters))

    for k, num_cols in enumerate(num_cols_list):
        print "k = %d; number of sampled columns = %d\n" % (k, num_cols)

        probs_unif = np.ones(d) / d
        probs_levr = model.leverage_scores()
        probs_rdge = model.ridge_leverage_scores()

        _, x_unif = model.iterative_solver(num_cols, num_iters, probs=probs_unif)
        _, x_levr = model.iterative_solver(num_cols, num_iters, probs=probs_levr)
        _, x_rdge = model.iterative_solver(num_cols, num_iters, probs=probs_rdge)

        rel_errs_unif[k] = model.rel_err(x_unif, x_opt)
        rel_errs_levr[k] = model.rel_err(x_levr, x_opt)
        rel_errs_rdge[k] = model.rel_err(x_rdge, x_opt)

        obj_errs_unif[k] = model.obj_vals(x_unif) / obj_opt - 1.
        obj_errs_levr[k] = model.obj_vals(x_levr) / obj_opt - 1.
        obj_errs_rdge[k] = model.obj_vals(x_rdge) / obj_opt - 1.

    return rel_errs_unif, rel_errs_levr, rel_errs_rdge, \
        obj_errs_unif, obj_errs_levr, obj_errs_rdge


if __name__ == "__main__":
    # lmbd: float, regularization parameter
    # seed: int, random seed
    lmbd, seed = sys.argv[1:]

    lmbd = float(lmbd)
    seed = int(seed)

    rand.seed(seed)

    # Generate synthetic dataset
    n = 500  # Number of rows
    d = 50000  # Number of columns
    s = 50  # rank(A)

    alpha = .05
    gamma = 5

    M = rand.normal(size=(n, s))
    D = np.diag(1. - (np.arange(s) - 1.) / d)
    # Random column-orthonormal matrix
    Q, _ = qr(rand.normal(size=(d, s)), mode='economic')
    E = rand.normal(size=(n, d))
    A = M.dot(D).dot(Q.T) + alpha * E  # Design matrix (n, d)

    x = rand.normal(size=d)  # Target vector
    e = rand.normal(size=n)  # Noise
    b = A.dot(x) + gamma * e  # Response vector

    # Run experiments
    num_cols_list = np.linspace(2000, 20000, num=6)
    num_iters = 50
    res = expr(A, b, lmbd, num_cols_list, num_iters)

    # Save results as cPickle file
    pckl_write(res, "ridge-synthetic-lmbd%.1f.res" % lmbd)

    print 'Finished!\n'
