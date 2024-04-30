from random import sample
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import re 
import os 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

from ortools.graph import pywrapgraph
import time


def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(cnt_1, dtype=np.int).tolist() + adj_mat.flatten().astype(np.int).tolist() + np.ones(cnt_2, dtype=np.int).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int).tolist() + cost_mat.flatten().astype(np.int).tolist() + np.zeros(cnt_2, dtype=np.int).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]
    # supplies = [min(cnt_1, cnt_2)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_1, cnt_2)]
    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        # print('Total cost = ', min_cost_flow.OptimalCost())
        # print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    # print('set A item %d assigned to set B item %d.  Cost = %d' % (
                    #     min_cost_flow.Tail(arc)-1,
                    #     min_cost_flow.Head(arc)-cnt_1-1,
                    #     min_cost_flow.UnitCost(arc)))
                    match_results.append([min_cost_flow.Tail(arc)-1,
                                          min_cost_flow.Head(arc)-cnt_1-1,
                                          min_cost_flow.UnitCost(arc)])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results



def resample_laneline_in_y_lrj(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility
    return x_values, z_values

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values

def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T

def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im

def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d