from __future__ import division
import cPickle
from collections import Counter
from copy import deepcopy
import itertools as it
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
import os
from operator import itemgetter
import scipy as sp
from scipy.linalg import det, eigvals, norm, expm, inv, pinv, qr, svd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, identity, \
    issparse
# import pandas as pd
# import seaborn as sns
import sys
import warnings


def check_prob_vector(p):
    """
    Check if a vector is a probability vector.

    Args:
        p, array/list.
    """
    assert np.all(p >= 0), p
    assert np.isclose(np.sum(p), 1), p

    return True


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def pckl_write(data, filename):
    with open(filename, 'w') as f:
        cPickle.dump(data, f)

    return


def pckl_read(filename):
    with open(filename, 'r') as f:
        data = cPickle.load(f)

    return data
