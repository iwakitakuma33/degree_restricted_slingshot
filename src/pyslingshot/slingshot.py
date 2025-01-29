from typing import Union

import numpy as np
import numpy.typing as npt
from anndata import AnnData

from .pcurve_custom import PrincipalCurve
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from collections import deque
from tqdm.autonotebook import tqdm

from .util import (
    scale_to_range,
    mahalanobis,
    isint,
    isstr,
    restricted_minimum_spanning_tree,
)
from .lineage import Lineage
from .plotter import SlingshotPlotter


class Slingshot:
    def __init__(
        self,
        data: Union[AnnData, np.ndarray],
        means: Union[npt.ArrayLike, None] = None,
        cluster_labels_onehot=None,
        celltype_key=None,
        obsm_key="X_umap",
        start_node=0,
        end_nodes=None,
        debug_level=None,
        is_restricted=False,
    ):
        """
        Constructs a new `Slingshot` object.
        Args:
            data: either an AnnData object or a numpy array containing the dimensionality-reduced data of shape (num_cells, 2)
            cluster_labels: cluster assignments of shape (num_cells). Only required if `data` is not an AnnData object.
            celltype_key: key into AnnData.obs indicating cell type. Only required if `data` is an AnnData object.
            obsm_key: key into AnnData.obsm indicating the dimensionality-reduced data. Only required if `data` is an AnnData object.
            start_node: the starting node of the minimum spanning tree
            end_nodes: any terminal nodes
            debug_level:
        """
        self.is_restricted = is_restricted
        if isinstance(data, AnnData):
            assert celltype_key is not None, "Must provide celltype key if data is an AnnData object"
            cluster_labels = data.obs[celltype_key]
            if isint(cluster_labels.iloc[0]):
                cluster_max = cluster_labels.max()
                self.cluster_label_indices = cluster_labels
            elif isstr(cluster_labels[0]):
                cluster_max = len(np.unique(cluster_labels))

                self.cluster_label_indices = np.array(
                    [np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels]
                )
            else:
                raise ValueError("Unexpected cluster label dtype.")
            cluster_labels_onehot = np.zeros((cluster_labels.shape[0], cluster_max + 1))
            cluster_labels_onehot[np.arange(cluster_labels.shape[0]), self.cluster_label_indices] = 1

            data = data.obsm[obsm_key]
        else:
            assert cluster_labels_onehot is not None, "Must provide cluster labels if data is not an AnnData object"
            cluster_labels = self.cluster_labels_onehot.argmax(axis=1)
        self.data = data
        self.cluster_labels_onehot = cluster_labels_onehot
        self.cluster_labels = cluster_labels
        self.num_clusters = self.cluster_label_indices.max() + 1
        self.start_node = start_node
        self.end_nodes = [] if end_nodes is None else end_nodes
        cluster_centres = [data[self.cluster_label_indices == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.lineages: Union[list[Lineage], None] = None
        self.cluster_lineages = None
        self.curves = None
        self.cell_weights: Union[npt.NDArray, None] = None
        self.distances = None
        self.branch_clusters = None
        self._tree = None
        self.mst = None
        debug_level = 0 if debug_level is None else dict(verbose=1)[debug_level]
        self.debug_level = debug_level
        self._set_debug_axes(None)
        self.plotter = SlingshotPlotter(self)

        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    @property
    def tree(self):
        if self._tree is None:
            self.construct_mst(self.start_node)
        return self._tree

    def load_params(self, filepath):
        if self.curves is None:
            self.get_lineages()
        params = np.load(filepath, allow_pickle=True).item()
        self.curves = params["curves"]
        self.cell_weights = params["cell_weights"]
        self.distances = params["distances"]

    def save_params(self, filepath):
        params = dict(curves=self.curves, cell_weights=self.cell_weights, distances=self.distances)
        np.save(filepath, params)

    def _set_debug_axes(self, axes):
        self.debug_axes = axes
        self.debug_plot_mst = axes is not None
        self.debug_plot_lineages = axes is not None
        self.debug_plot_avg = axes is not None

    def construct_mst(self, start_node):
        emp_covs = np.stack([np.cov(self.data[self.cluster_label_indices == i].T) for i in range(self.num_clusters)])
        dists = np.zeros((self.num_clusters, self.num_clusters))
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                dist = mahalanobis(self.cluster_centres[i], self.cluster_centres[j], emp_covs[i], emp_covs[j])
                dists[i, j] = dist
                dists[j, i] = dist

        mst_dists = np.delete(np.delete(dists, self.end_nodes, axis=0), self.end_nodes, axis=1)
        if self.is_restricted:
            self.mst = restricted_minimum_spanning_tree(mst_dists)
        else:
            self.mst = minimum_spanning_tree(mst_dists)
        tree = self.mst
        index_mapping = np.array([c for c in range(self.num_clusters - len(self.end_nodes))])
        for i, end_node in enumerate(self.end_nodes):
            index_mapping[end_node - i :] += 1

        connections = {k: list() for k in range(self.num_clusters)}
        cx = tree.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            i = index_mapping[i]
            j = index_mapping[j]
            connections[i].append(j)
            connections[j].append(i)

        for end in self.end_nodes:
            i = np.argmin(np.delete(dists[end], self.end_nodes))
            connections[i].append(end)
            connections[end].append(i)

        visited = [False for _ in range(self.num_clusters)]
        queue = list()
        queue.append(start_node)
        children = {k: list() for k in range(self.num_clusters)}
        while len(queue) > 0:
            current_node = queue.pop()
            visited[current_node] = True
            for child in connections[current_node]:
                if not visited[child]:
                    children[current_node].append(child)
                    queue.append(child)

        if self.debug_plot_mst:
            self.plotter.clusters(self.debug_axes[0, 0], alpha=0.5)
            for root, kids in children.items():
                for child in kids:
                    start = [self.cluster_centres[root][0], self.cluster_centres[child][0]]
                    end = [self.cluster_centres[root][1], self.cluster_centres[child][1]]
                    self.debug_axes[0, 0].plot(start, end, c="black")
            self.debug_plot_mst = False

        self._tree = children
        return children

    def fit(self, num_epochs=10, debug_axes=None):
        self._set_debug_axes(debug_axes)
        if self.curves is None:
            self.get_lineages()
            self.construct_initial_curves()
            self.cell_weights = [
                self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1) for l in range(len(self.lineages))
            ]
            self.cell_weights = np.stack(self.cell_weights, axis=1)

        for epoch in tqdm(range(num_epochs)):
            self.calculate_cell_weights()

            self.fit_lineage_curves()

            for l_idx, lineage in enumerate(self.lineages):
                curve = self.curves[l_idx]
                min_time = np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
                curve.pseudotimes_interp -= min_time

            shrinkage_percentages, cluster_children, cluster_avg_curves = self.avg_curves()

            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)

            self.debug_plot_lineages = False
            self.debug_plot_avg = False

            if self.debug_axes is not None and epoch == num_epochs - 1:
                self.plotter.clusters(self.debug_axes[1, 1], s=2, alpha=0.5)
                self.plotter.curves(self.debug_axes[1, 1], self.curves)

    def construct_initial_curves(self):
        """Constructs lineage principal curves using piecewise linear initialisation"""
        piecewise_linear = list()
        distances = list()

        for l_idx, lineage in enumerate(self.lineages):
            p = np.stack(self.cluster_centres[lineage.clusters])

            cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
            cells_involved = self.data[cell_mask]

            curve = PrincipalCurve(k=3)
            curve.project_to_curve(cells_involved, points=p)
            d_sq, dist = curve.project_to_curve(self.data, points=curve.points_interp[curve.order])
            distances.append(d_sq)

            piecewise_linear.append(curve)

        self.curves = piecewise_linear
        self.distances = distances

    def get_lineages(self):
        tree = self.construct_mst(self.start_node)

        branch_clusters = deque()

        def recurse_branches(path, v):
            num_children = len(tree[v])
            if num_children == 0:
                return path + [v, None]
            elif num_children == 1:
                return recurse_branches(path + [v], tree[v][0])
            else:
                branch_clusters.append(v)
                return [recurse_branches(path + [v], tree[v][i]) for i in range(num_children)]

        def flatten(li):
            if li[-1] is None:
                yield Lineage(li[:-1])
            else:
                for l in li:
                    yield from flatten(l)

        lineages = recurse_branches([], self.start_node)
        lineages = list(flatten(lineages))
        self.lineages = lineages
        self.branch_clusters = branch_clusters

        self.cluster_lineages = {k: list() for k in range(self.num_clusters)}
        for l_idx, lineage in enumerate(self.lineages):
            for k in lineage:
                self.cluster_lineages[k].append(l_idx)

        if self.debug_level > 0:
            print("Lineages:", lineages)

    def fit_lineage_curves(self):
        """Updates curve using a cubic spline and projection of data"""
        assert self.lineages is not None
        assert self.curves is not None
        distances = list()

        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]

            curve.fit(self.data, max_iter=1, w=self.cell_weights[:, l_idx])
            # curve.fit(
            #     self.data,
            #     max_iter=1,
            #     w=self.cell_weights[:, l_idx],
            #     knot_size=len(lineage) * 2
            # )
            if self.debug_plot_lineages:
                cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
                cells_involved = self.data[cell_mask]
                self.debug_axes[0, 1].scatter(cells_involved[:, 0], cells_involved[:, 1], s=2, alpha=0.5)
                alphas = curve.pseudotimes_interp
                alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                for i in np.random.permutation(self.data.shape[0])[:50]:
                    path_from = (self.data[i][0], curve.points_interp[i][0])
                    path_to = (self.data[i][1], curve.points_interp[i][1])
                    self.debug_axes[0, 1].plot(path_from, path_to, c="black", alpha=alphas[i])
                self.debug_axes[0, 1].plot(
                    curve.points_interp[curve.order, 0], curve.points_interp[curve.order, 1], label=str(lineage)
                )

            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[curve.order])
            distances.append(d_sq)
        self.distances = distances
        if self.debug_plot_lineages:
            self.debug_axes[0, 1].legend()

    def calculate_cell_weights(self):
        """TODO: annotate, this is a translation from R"""
        cell_weights = [
            self.cluster_labels_onehot[:, self.lineages[l].clusters].sum(axis=1) for l in range(len(self.lineages))
        ]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(self.distances, axis=1)
        d_ord = np.argsort(d_sq, axis=None)
        w_prob = cell_weights / cell_weights.sum(axis=1, keepdims=True)
        w_rnk_d = np.cumsum(w_prob.reshape(-1)[d_ord]) / w_prob.sum()

        z = d_sq
        z_shape = z.shape
        z = z.reshape(-1)
        z[d_ord] = w_rnk_d
        z = z.reshape(z_shape)
        z_prime = 1 - z**2
        z_prime[cell_weights == 0] = np.nan
        w0 = cell_weights.copy()
        cell_weights = z_prime / np.nanmax(z_prime, axis=1, keepdims=True)
        np.nan_to_num(cell_weights, nan=1, copy=False)

        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0
        cell_weights[w0 == 0] = 0

        reassign = True
        if reassign:
            cell_weights[z < 0.5] = 1

            ridx = (z.max(axis=1) > 0.9) & (cell_weights.min(axis=1) < 0.1)
            w0 = cell_weights[ridx]
            z0 = z[ridx]
            w0[(z0 > 0.9) & (w0 < 0.1)] = 0
            cell_weights[ridx] = w0

        self.cell_weights = cell_weights

    def avg_curves(self):
        """
        Starting at leaves, calculate average curves for each branch

        :return: shrinkage_percentages, cluster_children, cluster_avg_curves
        """
        cell_weights = self.cell_weights
        shrinkage_percentages = list()
        cluster_children = dict()
        lineage_avg_curves = dict()
        cluster_avg_curves = dict()
        branch_clusters = self.branch_clusters.copy()
        if self.debug_level > 0:
            print("Reversing from leaf to root")
        if self.debug_plot_avg:
            self.plotter.clusters(self.debug_axes[1, 0], s=4, alpha=0.4)

        while len(branch_clusters) > 0:
            k = branch_clusters.pop()
            branch_lineages = self.cluster_lineages[k]
            cluster_children[k] = set()
            for l_idx in branch_lineages:
                if l_idx in lineage_avg_curves:
                    curve = lineage_avg_curves[l_idx]
                else:
                    curve = self.curves[l_idx]
                cluster_children[k].add(curve)

            branch_curves = list(cluster_children[k])
            avg_curve = self.avg_branch_curves(branch_curves)
            cluster_avg_curves[k] = avg_curve

            common = cell_weights[:, branch_lineages] > 0
            common_mask = common.mean(axis=1) == 1.0
            shrinkage_percent = dict()
            for curve in branch_curves:
                shrinkage_percent[curve] = self.shrinkage_percent(curve, common_mask)
            shrinkage_percentages.append(shrinkage_percent)

            for l in branch_lineages:
                lineage_avg_curves[l] = avg_curve

        if self.debug_plot_avg:
            self.debug_axes[1, 0].legend()
        return shrinkage_percentages, cluster_children, cluster_avg_curves

    def shrink_curves(self, cluster_children, shrinkage_percentages, cluster_avg_curves):
        """
        Starting at root, shrink curves for each branch

        Parameters:
            cluster_children:
            shrinkage_percentages:
            cluster_avg_curves:
        :return:
        """
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            k = branch_clusters.popleft()
            shrinkage_percent = shrinkage_percentages.pop()
            branch_curves = list(cluster_children[k])
            cluster_avg_curve = cluster_avg_curves[k]
            if self.debug_level > 0:
                print(f"Shrinking branch @{k} with curves:", branch_curves)

            self.shrink_branch_curves(branch_curves, cluster_avg_curve, shrinkage_percent)

    def shrink_branch_curves(self, branch_curves, avg_curve, shrinkage_percent):
        """
        Shrinks curves through a branch to the average curve.

        :param branch_curves: list of `PrincipalCurve`s associated with the branch.
        :param avg_curve: `PrincipalCurve` for average curve.
        :param shrinkage_percent: percentage shrinkage, in same order as curve.pseudotimes
        """
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        for curve in branch_curves:
            pct = shrinkage_percent[curve]

            s_interp, p_interp, order = curve.unpack_params()
            avg_s_interp, avg_p_interp, avg_order = avg_curve.unpack_params()
            shrunk_curve = np.zeros_like(p_interp)
            for j in range(num_dims_reduced):
                orig = p_interp[order, j]
                avg = np.interp(
                    s_interp[order],
                    avg_s_interp[avg_order],
                    avg_p_interp[avg_order, j],
                )
                shrunk_curve[:, j] = avg * pct + orig * (1 - pct)
            curve.project_to_curve(self.data, points=shrunk_curve)

    def shrinkage_percent(self, curve, common_ind):
        """Determines how much to shrink a curve"""

        s_interp, order = curve.pseudotimes_interp, curve.order

        x = self.kernel_x
        y = self.kernel_y
        y = (y.sum() - np.cumsum(y)) / sum(y)
        q1 = np.percentile(s_interp[common_ind], 25)
        q3 = np.percentile(s_interp[common_ind], 75)
        a = q1 - 1.5 * (q3 - q1)
        b = q3 + 1.5 * (q3 - q1)
        x = scale_to_range(x, a=a, b=b)
        if q1 == q3:
            pct_l = np.zeros(s_interp.shape[0])
        else:
            pct_l = np.interp(s_interp[order], x, y)

        return pct_l

    def avg_branch_curves(self, branch_curves):
        """branch_lineages is a list of lineages passing through branch"""

        num_cells = branch_curves[0].points_interp.shape[0]
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        branch_s_interps = np.stack([c.pseudotimes_interp for c in branch_curves], axis=1)
        max_shared_pseudotime = branch_s_interps.max(axis=0).min()
        combined_pseudotime = np.linspace(0, max_shared_pseudotime, num_cells)
        curves_dense = list()
        for curve in branch_curves:
            lineage_curve = np.zeros((combined_pseudotime.shape[0], num_dims_reduced))
            order = curve.order

            for j in range(num_dims_reduced):
                lin_interpolator = interp1d(
                    curve.pseudotimes_interp[order],
                    curve.points_interp[order, j],
                    assume_sorted=True,
                )
                lineage_curve[:, j] = lin_interpolator(combined_pseudotime)
            curves_dense.append(lineage_curve)

        curves_dense = np.stack(curves_dense, axis=1)

        avg = curves_dense.mean(axis=1)
        avg_curve = PrincipalCurve()
        avg_curve.project_to_curve(self.data, points=avg)
        if self.debug_plot_avg:
            self.debug_axes[1, 0].plot(avg[:, 0], avg[:, 1], c="blue", linestyle="--", label="average", alpha=0.7)
            _, p_interp, order = avg_curve.unpack_params()
            self.debug_axes[1, 0].plot(
                p_interp[order, 0], p_interp[order, 1], c="red", label="data projected", alpha=0.7
            )

        return avg_curve

    @property
    def unified_pseudotime(self):
        pseudotime = np.zeros_like(self.curves[0].pseudotimes_interp)
        for l_idx, lineage in enumerate(self.lineages):
            curve = self.curves[l_idx]
            cell_mask = np.logical_or.reduce(np.array([self.cluster_label_indices == k for k in lineage]))
            pseudotime[cell_mask] = curve.pseudotimes_interp[cell_mask]
        return pseudotime

    def list_lineages(self, cluster_to_label):
        for lineage in self.lineages:
            print(", ".join([cluster_to_label[l] for l in lineage]))
