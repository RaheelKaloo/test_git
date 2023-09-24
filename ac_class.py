# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:24:06 2022

@author: umroot
"""
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np
import pandas as pd
import constant as c
import csv
from kmodes.kmodes import KModes
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import seaborn as sns1; sns1.set()
from sklearn.metrics import jaccard_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn import metrics
from random import sample
class _HungarianState(object):
    """State of one execution of the Hungarian algorithm.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    """

    def __init__(self, cost_matrix):
        cost_matrix = np.atleast_2d(cost_matrix)

        # If there are more rows (n) than columns (m), then the algorithm
        # will not be able to work correctly. Therefore, we
        # transpose the cost function when needed. Just have to
        # remember to swap the result columns back later.
        transposed = (cost_matrix.shape[1] < cost_matrix.shape[0])
        if transposed:
            self.C = (cost_matrix.T).copy()
        else:
            self.C = cost_matrix.copy()
        self.transposed = transposed

        # At this point, m >= n.
        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=np.bool)
        self.col_uncovered = np.ones(m, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


def _hungarian(cost_matrix):
    """The Hungarian algorithm.
    Calculate the Munkres solution to the classical assignment problem and
    return the indices for the lowest-cost pairings.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    Returns
    -------
    indices : 2D array of indices
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    """
    state = _HungarianState(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    # Look for the starred columns
    results = np.array(np.where(state.marked == 1)).T

    # We need to swap the columns because we originally
    # did a transpose on the input cost matrix.
    if state.transposed:
        results = results[:, ::-1]

    return results


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(np.int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= state.col_uncovered.astype(dtype=np.int, copy=False)
    n = state.C.shape[0]
    m = state.C.shape[1]
    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if not state.marked[row, star_col] == 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    state.row_uncovered.astype(dtype=np.int, copy=False))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if not state.marked[row, path[count, 1]] == 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[np.logical_not(state.row_uncovered)] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4



def linear_assignment(X):
    """Solve the linear assignment problem using the Hungarian algorithm.
    The problem is also known as maximum weight matching in bipartite graphs.
    The method is also known as the Munkres or Kuhn-Munkres algorithm.
    Parameters
    ----------
    X : array
        The cost matrix of the bipartite graph
    Returns
    -------
    indices : array
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    References
    ----------
    1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.
    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *Journal of the Society of Industrial and Applied Mathematics*,
       5(1):32-38, March, 1957.
    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    indices = _hungarian(X).tolist()
    indices.sort()
    # Re-force dtype to ints in case of empty list
    indices = np.array(indices, dtype=int)
    # Make sure the array is 2D with 2 columns.
    # This is needed when dealing with an empty list
    indices.shape = (-1, 2)
    return indices


def permutations_matrix(true, prediction ):
    import seaborn as sns; sns.set()
    cm = confusion_matrix(true, prediction)
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    sns.heatmap(cm2, annot=True, fmt=".1f")
    plt.show()
    accuracy_permution=np.trace(cm2) / np.sum(cm2)
    print("accuracy_permution",accuracy_permution)
    accuracy_nmi = normalized_mutual_info_score(true, prediction)
    print("NMI Final clustering  ",accuracy_nmi)
    accuracy_ari=adjusted_rand_score(true, prediction)
    print("ARI",accuracy_ari)
    return cm

# def permutations_matrix(true, prediction ):
#     import seaborn as sns; sns.set()
#     cm = confusion_matrix(true, prediction)
#     #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     indexes = linear_assignment(_make_cost_m(cm))
#     js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
#     cm2 = cm[:, js]
#     cm2_df=pd.DataFrame(data=cm2,columns=indexes[:,1])
#     cm2_df=cm2_df.loc[:,indexes[:,0]]
#     sns.heatmap(cm2_df, annot=True, fmt=".1f")
#     plt.show()
#     accuracy_permution=np.trace(cm2) / np.sum(cm2)
#     print("accuracy",accuracy_permution)
#     return cm
    

def permutations_matrix1(true, prediction ):
    import seaborn as sns; sns.set()
    cm = confusion_matrix(true, prediction)
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    sns.heatmap(cm2, annot=True, fmt=".1f")
    plt1.show()
    accuracy_permution=np.trace(cm2) / np.sum(cm2)
    print("accuracy_permution",accuracy_permution)
    return cm


def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def _make_cost_m1(cm):
    s = np.max(cm)
    return (- cm + s)
        

def clustering_accurcy(true, prediction):
    
    cm1 = confusion_matrix(true, prediction)
    indexes = linear_assignment(_make_cost_m1(cm1))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm3 = cm1[:, js]
    accuracy_permution=np.trace(cm3) / np.sum(cm3)
    print("accuracy_permution",accuracy_permution)
    return accuracy_permution

  
