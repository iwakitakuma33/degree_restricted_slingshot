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


def restricted_minimum_spanning_tree(csgraph: npt.NDArray[np.float64]) -> csr_matrix:
    # 頂点集合の定義
    V = range(len(csgraph))

    # 辺集合とその重みの定義
    E = []
    d = {}
    for i in V:
        for j in V:
            if i < j and csgraph[i][j] > 0:
                E.append((i, j))
                d[(i, j)] = csgraph[i][j]

    # モデルの作成
    m = gp.Model()
    m.Params.OutputFlag = 0
    # 変数の定義
    x = m.addVars(E, vtype=GRB.BINARY, name="x")

    # 目的関数の設定
    m.setObjective(gp.quicksum(d[e] * x[e] for e in E), GRB.MINIMIZE)

    # 変数の二値制約
    for e in E:
        m.addConstr(x[e] >= 0, f"c4_{e}")

    # 各頂点の次数制約
    for i in V:
        m.addConstr(gp.quicksum(x[e] for e in E if i in e) <= 3, f"c2_{i}")
        m.addConstr(gp.quicksum(x[e] for e in E if i in e) >= 1, f"deg_{i}_min")

    # # 制約の追加
    m.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, "c1")

    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._x)
            edges = [(i, j) for i, j in model._x.keys() if vals[i, j] > 0.5]

            G = nx.Graph()
            G.add_edges_from(edges)
            components = list(nx.connected_components(G))
            if len(components) > 1:
                for component in components:
                    if len(component) < len(V):
                        component_list = list(component)
                        model.cbLazy(
                            gp.quicksum(
                                x[i, j]
                                for i in component_list
                                for j in component_list
                                if i < j and (i, j) in E
                            )
                            <= len(component_list) - 1
                        )

    # コールバック関数を設定
    m._x = x
    m.Params.lazyConstraints = 1
    m.optimize(subtour_elimination)

    if m.Status != GRB.OPTIMAL:
        raise Exception("Optimal solution not found!")

    # 最小全域木の辺を取得
    mst_edges = [(i, j) for i, j in E if x[i, j].X > 0.5]

    # 最小全域木の隣接行列を構築
    mst_matrix = np.zeros_like(csgraph)
    for i, j in mst_edges:
        mst_matrix[i, j] = csgraph[i, j]

    # csr_matrix に変換して返す
    return csr_matrix(mst_matrix)
