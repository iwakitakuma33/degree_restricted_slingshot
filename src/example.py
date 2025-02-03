# mstいい感じに出せる
import cospar as cs
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from pyslingshot import Slingshot
import util
from sklearn.cluster import KMeans
import pyslingshot.img_util as img_util


data = cs.hf.read("your_path/degree_restricted_slingshot_data/data/cospar/data.npy")

X, Y = data[:, 0], data[:, 1]
VAR_X = np.var(X)
VAR_Y = np.var(Y)
RANDOM_STATE = 150
INIT_BINS = img_util.pow(VAR_X + VAR_Y)


area_ratio = img_util.calc_area_ratio(data)
results = {}
bins = None
for _bins in range(INIT_BINS, INIT_BINS * 10, INIT_BINS):
    img_info = img_util.calc_bin_img_info(data, _bins)
    if img_info["info"]["pixel_ratio"] < area_ratio:
        results[_bins] = img_info

print("====== choice bins")
sorted_results = dict(
    sorted(
        results.items(), key=lambda item: item[1]["info"]["pixel_ratio"], reverse=True
    )
)
for _bins, img_info in sorted_results.items():
    img, info = img_info["img"], img_info["info"]
    num_objects, pixel_ratio = info["num_objects"], info["pixel_ratio"]
    if not bins and num_objects > 9 and num_objects < 20:
        bins = _bins
    binary_image, xedges, yedges = img["binary_image"], img["xedges"], img["yedges"]
    skelton_tree = img_util.skeletonize_tree(binary_image, xedges, yedges)
    sorted_results[_bins]["skelton_tree"] = skelton_tree
    print(
        f"bins: {_bins}, num_objects: {num_objects}, n_clusters: {skelton_tree['n_clusters']}, pixel_ratio: {round(pixel_ratio,3)}"
    )

if not bins:
    raise Exception("bins is None")

img_info = sorted_results[bins]
skelton_tree = img_info["skelton_tree"]
n_clusters = skelton_tree["n_clusters"]
initial_centers = skelton_tree["initial_centers"]
mapped_centroid_x = skelton_tree["mapped_centroid_x"]
mapped_centroid_y = skelton_tree["mapped_centroid_y"]

print("====== main process")
print(f"bins: {bins}, n_clusters: {n_clusters}")

kmeans = KMeans(
    n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=RANDOM_STATE
)
kmeans.fit(data)
ad = util.get_ann_data(data, kmeans.labels_)

target_point = np.array([mapped_centroid_x, mapped_centroid_y])
distances = np.linalg.norm(kmeans.cluster_centers_ - target_point, axis=1)
nearest_center_index = np.argmin(distances)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
tzinfo = datetime.timezone(datetime.timedelta(hours=9))
print(datetime.datetime.now(tzinfo))
start = time.time()
slingshot = Slingshot(
    ad,
    means=None,
    celltype_key="celltype",
    obsm_key="X_umap",
    start_node=int(nearest_center_index),
    debug_level="verbose",
    is_restricted=True,
)
slingshot.fit(num_epochs=1)

end = time.time()
elapsed_time = end - start
elapsed_minutes = elapsed_time / 60  # 分単位に変換
print(f"処理時間: {elapsed_minutes:.2f}分")
