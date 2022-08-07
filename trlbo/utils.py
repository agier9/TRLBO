###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import numpy as np


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim,semi_opposite=True):
    """Basic Latin hypercube implementation with center perturbation."""
    pts = n_pts if not semi_opposite else n_pts//2
    X = np.zeros((pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, pts)) / float(2 * pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (pts, dim)) / float(2 * pts)
    X += pert
    if semi_opposite:
        x_opp = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                opp = 1.0-X[i][j]
                (left,right) = (opp,X[i][j]) if opp < 0.5 else (X[i][j],opp)
                x_opp[i][j] = np.random.random()*(right-left)+left
        X = np.vstack((X,x_opp))
    return X
