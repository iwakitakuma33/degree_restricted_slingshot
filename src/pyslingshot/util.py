import numpy as np

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from scipy.sparse import csr_matrix
import numpy.typing as npt


def mahalanobis(X1, X2, S1, S2):
    S_inv = np.linalg.inv(S1 + S2)
    diff = (X1 - X2).reshape(-1, 1)
    return np.matmul(np.matmul(diff.T, S_inv), diff)


def isint(x):
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        return False
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def isstr(x):
    return isinstance(x, str)


def scale_to_range(x, a=0, b=1):
    return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a


def restricted_minimum_spanning_tree(csgraph):
    V = range(len(csgraph))  # 頂点集合
    E = [(i, j) for i in V for j in V if i < j and csgraph[i, j] > 0]  # 辺集合
    phi = {(i, j): csgraph[i, j] for (i, j) in E}  # 重み

    # モデル作成
    m = gp.Model()

    # 変数定義
    x = m.addVars(E, vtype=GRB.BINARY, name="x")
    y = m.addVars(
        [(i, j, k) for (i, j) in E for k in V] + [(j, i, k) for (i, j) in E for k in V],
        vtype=GRB.BINARY,
        name="y",
    )
    # 目的関数: 重みの総和を最小化
    m.setObjective(gp.quicksum(phi[i, j] * x[i, j] for (i, j) in E), GRB.MINIMIZE)

    # 制約1: MST の辺数は n - 1
    m.addConstr(gp.quicksum(x[i, j] for (i, j) in E) == len(V) - 1, "edges_count")

    # 変数の二値制約
    for e in E:
        m.addConstr(x[e] >= 0, f"c4_{e}")

    # 各頂点の次数制約
    for i in V:
        m.addConstr(gp.quicksum(x[e] for e in E if i in e) <= 3, f"c2_{i}")
        m.addConstr(gp.quicksum(x[e] for e in E if i in e) >= 1, f"deg_{i}_min")

    for i, j in E:
        for k in V:
            m.addConstr(y[i, j, k] + y[j, i, k] == x[i, j], f"y_flow_{i}_{j}_{k}")

    # 制約3: フロー保存制約 (flow conservation)
    for i, j in E:
        m.addConstr(
            gp.quicksum(y[i, k, j] for k in V if k != i and k != j) + x[i, j] == 1,
            f"flow_conservation_{i}_{j}",
        )
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise Exception("Optimal solution not found!")
    mst_edges = [(i, j) for (i, j) in E if x[i, j].X > 0.5]

    mst_matrix = np.zeros_like(csgraph)
    for i, j in mst_edges:
        mst_matrix[i, j] = csgraph[i, j]

    return csr_matrix(mst_matrix)
