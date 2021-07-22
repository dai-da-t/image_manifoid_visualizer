import argparse
import glob
import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from tqdm import tqdm


def visualize_image_distribution(
    images: np.ndarray,
    coords: np.ndarray,
    zoom: float = 0.22,
    figsize: Tuple[float, float] = (25, 15),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    coords[i]にimages[i]をscatter
    """
    fig, ax = plt.subplots(figsize=figsize)

    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    ax.set_xlim(xmin * 0.9, xmax * 1.1)
    ax.set_ylim(ymin * 0.9, ymax * 1.1)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    artists = []
    for image, coord in zip(images, coords):
        # BGR->RGB cvtColorはint8を想定されているため正規化したデータでは使えない
        image = image[:, :, [2, 1, 0]]

        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (coord[0], coord[1]), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))

    return fig, ax


def load_images(image_dir: str, image_size: Tuple[int, int] = (290, 350)) -> np.ndarray:
    image_list = []

    image_path = os.path.join(image_dir, "*.jpg")

    for img_fname in glob.glob(image_path):
        image = cv2.imread(img_fname)
        image = cv2.resize(image, dsize=image_size)
        image_list.append(image / 255)

    return np.array(image_list)


def save_reduction_result(
    model,
    data: np.ndarray,
    fig_name: str,
    zoom: float = 0.25,
    figsize: Tuple[float, float] = (40, 25),
):
    """
    次元削減を行って結果を画像に保存
    """
    data_flatten = data.reshape(data.shape[0], -1)
    coords = model.fit_transform(data_flatten)

    fig, ax = visualize_image_distribution(data, coords, zoom=zoom, figsize=figsize)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight")
    plt.clf()
    plt.close()


def main(args: argparse.Namespace):
    images = load_images(args.input_dir)
    print("Num images: {}".format(images.shape[0]))

    models = {
        "isomap": Isomap(n_neighbors=args.n_neighbors, n_components=2),
        "lle": LocallyLinearEmbedding(n_neighbors=args.n_neighbors, n_components=2),
        "laplacian": SpectralEmbedding(n_neighbors=args.n_neighbors, n_components=2),
        "tsne": TSNE(n_components=2),
        "pca": PCA(n_components=2),
        "mds": MDS(n_components=2),
    }

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for model_name, model in tqdm(models.items()):
        tqdm.write("Processing {}".format(model_name))
        save_reduction_result(model, images, os.path.join(args.output_dir, model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", "-i", help="path to image dir", required=True, type=str
    )
    parser.add_argument(
        "--output_dir", "-o", help="path to output dir", default="outputs", type=str
    )
    parser.add_argument(
        "--n_neighbors", "-n", help="num neighbors", default=8, type=int
    )

    args = parser.parse_args()

    main(args)
