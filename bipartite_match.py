import numpy as np
import scipy as sp
import pandas as pd


def bipartite_match(true_centers, pred_centers, max_match_dist, dim_resolution=1): 
    """
    Match true and predicted markers using max-weight bipartite matching.
    This function is a wrapper around scipy.spatial.distance.cdist and scipy.optimize.linear_sum_assignment.
    40 to 100 times faster than networkx.
    
    Parameters
    ----------
    true_centers : numpy.array or pandas.DataFrame
      array of dim(n_cells, n_dim)

    pred_centers : numpy.array or pandas.DataFrame
      array of dim(n_pred_cells, n_dim)

    max_distance : float
      distance below which two markers are never matched. If dim_resolution is given this distance must be in that scale.

    dim_resolution: list of float
      resolution of input data (e.g. micron per voxel axis)

    return:
      DataFrame of matched and not matched centers.
    """
    if np.isscalar(dim_resolution):
        dim_resolution = (dim_resolution, ) * true_centers.shape[1]
    
    dim_resolution = np.array(dim_resolution)

    scaled_true = true_centers * dim_resolution
    scaled_pred = pred_centers * dim_resolution

    dist = sp.spatial.distance.cdist(scaled_true, scaled_pred, metric='euclidean')
    dist[dist >= max_match_dist] = 1e6
    dist = np.where(dist > 0.001, 1/dist, 1/0.001)

    true_idxs, pred_idxs = sp.optimize.linear_sum_assignment(dist, maximize=True)

    #print(np.any([idx in set(pred_idxs) - set([idx]) for idx in pred_idxs]))

    pred_TP = [pred_idxs[i] for i in range(len(true_idxs)) if dist[true_idxs[i], pred_idxs[i]] > 1 / max_match_dist]
    true_TP = [true_idxs[i] for i in range(len(true_idxs)) if dist[true_idxs[i], pred_idxs[i]] > 1 / max_match_dist]
    FP = [idx for idx in range(pred_centers.shape[0]) if idx not in pred_TP]
    FN = [idx for idx in range(true_centers.shape[0]) if idx not in true_TP]
    
    # create data frame of labeled centers
    colnames = ["x", "y", "z", "radius", "shape", "name", "comment", "R", "G", "B"]
    node_eval = []

    # create data frame of labeled centers
    colnames = ["x", "y", "z", "radius", "shape", "name", "comment", "R", "G", "B"]
    node_eval = []
    for i in pred_TP:
        x, y, z = pred_centers[i, 0], pred_centers[i, 1], pred_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'TP', 'predicted', 0, 255, 0])

    for i in FP:
        x, y, z = pred_centers[i, 0], pred_centers[i, 1], pred_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'FP', 'predicted', 255, 0, 0])

    for i in FN:
        x, y, z = true_centers[i, 0], true_centers[i, 1], true_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'FN', 'true', 255, 128, 0])
    
    labeled_centers = pd.DataFrame(node_eval, columns=colnames)

    # creates data frame of matched true centers and predicted centers   
    matched = []
    for i, j in zip(pred_TP, true_TP):
      xp, yp, zp = pred_centers[i, 0], pred_centers[i, 1], pred_centers[i, 2]
      xt, yt, zt = true_centers[j, 0], true_centers[j, 1], true_centers[j, 2]
      matched.append([[xp, yp, zp], [xt, yt, zt]])
    
    colnames = ["predicted_centers", "true_centers"]
    matched = pd.DataFrame(matched, columns = colnames)

    return labeled_centers, matched