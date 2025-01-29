from scipy.ndimage import binary_fill_holes, convolve, gaussian_filter
from skimage import filters, measure, morphology
from skimage.measure import find_contours
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label as measure_label
from shapely.geometry import Polygon, MultiPolygon
import alphashape
import math

ALPHA = 0.01
SIGMA = 8
SIGMA_RATIO = 0.8
DISTANCE_THRESHOLD_LEAF_BRANCH = 2.8
DISTANCE_THRESHOLD_BRANCH = 2


def auto_round_down(value):
    if value == 0:
        return 0
    digits = int(math.log10(abs(value)))
    factor = 10**digits
    return factor


def pow(value):
    v = auto_round_down(value)
    cube_root = math.floor(value / v) * v ** (1 / 3)
    return auto_round_down(max(int(cube_root), 10))


def calc_area_ratio(data, alpha=ALPHA):
    alpha_shape = alphashape.alphashape(data, alpha)

    area_ratio = None
    if isinstance(alpha_shape, Polygon):
        total_area = alpha_shape.area
        min_x, min_y, max_x, max_y = alpha_shape.bounds
        rectangle_area = (max_x - min_x) * (max_y - min_y)
        area_ratio = total_area / rectangle_area
    elif isinstance(alpha_shape, MultiPolygon):
        total_area = sum(polygon.area for polygon in alpha_shape.geoms)
        min_x = min(polygon.bounds[0] for polygon in alpha_shape.geoms)
        min_y = min(polygon.bounds[1] for polygon in alpha_shape.geoms)
        max_x = max(polygon.bounds[2] for polygon in alpha_shape.geoms)
        max_y = max(polygon.bounds[3] for polygon in alpha_shape.geoms)
        rectangle_area = (max_x - min_x) * (max_y - min_y)
        area_ratio = total_area / rectangle_area
    return area_ratio


def calc_bin_img_info(data, bins):
    x, y = data[:, 0], data[:, 1]
    var_x, var_y = np.var(x), np.var(y)
    pow_vars = pow(var_x + var_y)
    SMALL_OBJ = max(int(pow_vars * 0.02), 2)
    SMALL_HOLL = max(int(pow_vars * 0.02), 2)
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)  # 解像度に応じてbins数を調整
    hist_blurred = gaussian_filter(hist, sigma=SIGMA) * SIGMA_RATIO
    hist_scaled = (255 * (hist - np.min(hist_blurred)) / (np.max(hist_blurred) - np.min(hist_blurred))).astype(np.uint8)
    threshold = filters.threshold_otsu(hist_scaled)
    binary_image = hist_scaled > threshold  # 濃い部分のみを残す
    binary_image = morphology.remove_small_holes(binary_image, area_threshold=SMALL_HOLL)
    binary_image = morphology.remove_small_objects(binary_image, min_size=SMALL_OBJ)
    binary_image = binary_fill_holes(binary_image)
    labeled_image = measure_label(binary_image)
    num_objects = labeled_image.max()
    pixel_ratio = np.sum(binary_image) / binary_image.size  # 割合
    img = {"binary_image": binary_image, "xedges": xedges, "yedges": yedges}
    info = {"bins": bins, "num_objects": num_objects, "pixel_ratio": pixel_ratio}
    return {"img": img, "info": info}


def skeletonize_tree(binary_image, xedges, yedges, ax=None):
    skeleton = morphology.skeletonize(binary_image.T)
    skeleton = morphology.remove_small_objects(skeleton, min_size=3, connectivity=2)
    labeled_skeleton, num_labels = measure.label(skeleton, connectivity=2, return_num=True)

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(skeleton.astype(int), neighbor_kernel, mode="constant", cval=0)

    branch_points_x, branch_points_y = np.array([]), np.array([])
    end_points_x, end_points_y = np.array([]), np.array([])

    for label in range(1, num_labels + 1):
        component = labeled_skeleton == label
        branch_points = (neighbor_count >= 3) & component  # 分岐点
        end_points = (neighbor_count == 1) & component  # 端点（葉）

        _end_points_y, _end_points_x = np.where(end_points)
        _branch_points_y, _branch_points_x = np.where(branch_points)

        branch_points_x = np.concatenate([branch_points_x, _branch_points_x])
        branch_points_y = np.concatenate([branch_points_y, _branch_points_y])
        end_points_x = np.concatenate([end_points_x, _end_points_x])
        end_points_y = np.concatenate([end_points_y, _end_points_y])

    branch_points = np.array(list(zip(branch_points_x, branch_points_y)))
    end_points = np.array(list(zip(end_points_x, end_points_y)))
    end_points_to_remove = set()
    if branch_points.shape != (0,) and end_points.shape != (0,):
        dist_matrix_leaf_branch = cdist(branch_points, end_points)
        for i, _ in enumerate(branch_points):
            for j, _ in enumerate(end_points):
                if dist_matrix_leaf_branch[i, j] < DISTANCE_THRESHOLD_LEAF_BRANCH:
                    end_points_to_remove.add(i)  # 削除する葉のインデックスを追加

    # 葉の配列から削除対象のインデックスを除外
    end_points = np.array([point for idx, point in enumerate(end_points) if idx not in end_points_to_remove])

    # 分岐点同士の距離を計算
    branch_points_to_remove = set()
    if branch_points.shape != (0,):
        dist_matrix_branch_branch = cdist(branch_points, branch_points)

        # 分岐点同士が近すぎる場合のランダム削除
        for i in range(len(branch_points)):
            for j in range(i + 1, len(branch_points)):
                if dist_matrix_branch_branch[i, j] < DISTANCE_THRESHOLD_BRANCH:
                    branch_points_to_remove.add(i)

    # 分岐点の配列から削除対象のインデックスを除外
    branch_points = np.array([point for idx, point in enumerate(branch_points) if idx not in branch_points_to_remove])
    if branch_points.shape[0] > 0 and end_points.shape[0] > 0:
        p_x = np.concatenate([branch_points[:, 0], end_points[:, 0]])
        p_y = np.concatenate([branch_points[:, 1], end_points[:, 1]])
    else:
        p_x, p_y = np.array([]), np.array([])  # 空の配列を代入

    binary_image = np.logical_not(binary_image)
    contours = find_contours(binary_image.T, level=0.8)
    contour = max(contours, key=len)
    centroid_x = np.mean(contour[:, 1])
    centroid_y = np.mean(contour[:, 0])

    x_bin_size = xedges[1] - xedges[0]
    y_bin_size = yedges[1] - yedges[0]
    mapped_centroid_x = centroid_x * x_bin_size + xedges[0]
    mapped_centroid_y = centroid_y * y_bin_size + yedges[0]
    mapped_p_x = p_x * x_bin_size + xedges[0]
    mapped_p_y = p_y * y_bin_size + yedges[0]
    initial_centers = np.column_stack((mapped_p_x, mapped_p_y))
    n_clusters = len(mapped_p_x)
    return {
        "n_clusters": n_clusters,
        "initial_centers": initial_centers,
        "mapped_centroid_x": mapped_centroid_x,
        "mapped_centroid_y": mapped_centroid_y,
        "p_x": p_x,
        "p_y": p_y,
        "mapped_p_x": mapped_p_x,
        "mapped_p_y": mapped_p_y,
        "skeleton": skeleton,
    }
