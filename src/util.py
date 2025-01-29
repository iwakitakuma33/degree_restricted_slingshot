from env import DATA_PATH, NPY_BIN_DIR, OUTPUT_PATH
import csv
import numpy as np
import os
import numpy.typing as npt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from anndata import AnnData
from matplotlib import pyplot as plt
from PIL import Image
from typing import Optional
from matplotlib import colors
import random


def get_gse_file_pathes(gse_id: str, file_name: str) -> dict:
    _file_name = file_name.rstrip(".csv")
    _csv_file_name = f"{_file_name}.csv"
    _npy_file_name = f"{_file_name}.npy"

    return {
        "path": f"{DATA_PATH}/{gse_id}/{_csv_file_name}",
        "npy_path": f"{DATA_PATH}/{gse_id}/{NPY_BIN_DIR}/{_npy_file_name}",
    }


def _csv_to_nparray_data(file_path: str) -> npt.NDArray:
    result = {}
    with open(file_path, "rt") as f:
        full_header = np.array(next(csv.reader(f)))

        meta_data_header_start_index = np.where(full_header == "sample_sequencing_id")[0]  # fmt:skip
        header, meta_data_header = np.split(full_header, meta_data_header_start_index)
        reader = csv.reader(f)

        data = []
        meta_data = []
        for line in reader:
            line_data = []
            line_meta_data = []
            for i, row in enumerate(line):
                if i < meta_data_header_start_index:
                    line_data.append(float(row))
                else:
                    line_meta_data.append(row)
            data.append(line_data)
            meta_data.append(line_meta_data)

        result["header"] = header
        result["data"] = np.array(data)
        result["meta_data_header"] = meta_data_header
        result["meta_data"] = np.array(meta_data)

    return np.array(result)


def csv_to_nparray_data(gse_id: str, file_name: str) -> npt.NDArray:
    """
    "ZFP85","SCAP"
    0.323174,8.71516

    to

    {
        "header": ["ZFP85","SCAP"], "data": [[0.3,8.7],[0.3,8.7]...],
        "meta_data_header": ["ABC","DEF"], "meta_data": [["abc"],["def"]...],

    }
    """
    file_pathes = get_gse_file_pathes(gse_id, file_name)
    npy_path = file_pathes["npy_path"]
    csv_path = file_pathes["path"]
    if os.path.isfile(npy_path):
        print(f"csv_to_nparray_data: load npy data {npy_path}")
        result = np.load(npy_path, allow_pickle=True)
        return result.item()
    elif os.path.isfile(csv_path):
        result = _csv_to_nparray_data(csv_path)
        np.save(npy_path, result)
        print(f"csv_to_nparray_data: save npy data {npy_path}")
        return result.item()
    else:
        print(f"csv_to_nparray_data: not found file {gse_id} {file_name}")
        raise


def gmm_clustering(data: npt.NDArray, n_components: int):
    # KMeans で初期のクラスタの重心を取得 (k-means++)
    random_state = 81
    kmeans = KMeans(
        n_clusters=n_components, init="k-means++", random_state=random_state
    )
    kmeans.fit(data)
    initial_means = kmeans.cluster_centers_
    gmm = GaussianMixture(
        n_components=n_components,
        means_init=initial_means,
        random_state=random_state,
        covariance_type="full",
        # covariance_type="spherical",
        # covariance_type="spherical",
        # covariance_type="spherical",
    )
    gmm.fit(data)
    return kmeans.predict(data)


def get_ann_data(data: npt.NDArray, labels: npt.NDArray) -> AnnData:
    np.save("data.npy", data)
    ad = AnnData(np.zeros(data.shape))
    ad.obsm["X_umap"] = data
    ad.obs["celltype"] = labels
    return ad


def save_fig(name: str, data: npt.NDArray, labels: Optional[npt.NDArray] = None):
    file_path = f"{OUTPUT_PATH}/{name}.png"
    plt.clf()
    plt.close()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.savefig(file_path)
    im = Image.open(file_path)
    im.show()


def save_fig2(
    name: str,
    data: npt.NDArray,
    labels: Optional[npt.NDArray] = None,
    n_components: Optional[int] = None,
):
    file_path = f"{OUTPUT_PATH}/{name}.png"
    plt.clf()
    plt.close()
    colors = choose_colors(n_components)
    cluster_labels = [f"Cluster {i}" for i in range(n_components)]

    if not n_components:
        plt.scatter(data[:, 0], data[:, 1], c=labels)
    else:
        for i in range(n_components):
            plt.scatter(
                data[labels == i, 0],
                data[labels == i, 1],
                c=colors[i],
                label=cluster_labels[i],
            )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={"size": 8})
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    im = Image.open(file_path)
    im.show()


def int_input(prompt: str) -> int:
    while True:
        i = input(prompt)
        if i.isdecimal():
            i = int(i)
            return i
        print(f"{i}はint型である必要があります")


def choose_colors(num_colors):
    tmp = list(colors.CSS4_COLORS.values())
    random.shuffle(tmp)
    label2color = tmp[:num_colors]
    return label2color
