import argparse
import glob
import os
from typing import Optional, Tuple

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


def initialize_canvas(figsize, coords: Optional[np.ndarray] = None, font_size: int = 14):
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()

    if coords is not None:
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        ax.set_xlim(xmin * 0.9, xmax * 1.1)
        ax.set_ylim(ymin * 0.9, ymax * 1.1)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    return fig, ax


def visualize_image_distribution(
    images: np.ndarray,
    coords: np.ndarray,
    figsize: Tuple[float, float],
    zoom: float,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    coords[i]にimages[i]をscatter
    """
    fig, ax = initialize_canvas(figsize, coords, font_size=20)

    for image, coord in zip(images, coords):
        # BGR->RGB cvtColorはint8を想定されているため正規化したデータでは使えない
        image = image[:, :, [2, 1, 0]]

        image_offset = OffsetImage(image, zoom=zoom)
        annotationbbox = AnnotationBbox(image_offset, (coord[0], coord[1]), xycoords="data", frameon=False)
        ax.add_artist(annotationbbox)

    return fig, ax


def visualize_reduction(
    labels: np.ndarray, coords: np.ndarray, figsize: Tuple[float, float]
):
    fig, ax = initialize_canvas(figsize, coords)

    cmap = plt.get_cmap("jet")
    for i, name in enumerate(np.unique(labels)):
        indices = np.where(labels == name)[0]
        ax.scatter(coords[indices, 0], coords[indices, 1], s=50, color=cmap(i/np.unique(labels).shape[0]), label=name)

    ax.legend()

    return fig, ax


def load_images_with_label(
    image_dir: str, image_size: Tuple[int, int] = (290, 350)
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = []
    labels = []
    image_path = os.path.join(image_dir, "*.jpg")

    for img_fname in glob.glob(image_path):
        image = cv2.imread(img_fname)
        image = cv2.resize(image, dsize=image_size)
        image_list.append(image / 255)

        label = img_fname.split('/')[-2]
        labels.append(label)

    return np.array(image_list), np.array(labels)


def save_figure(fig: plt.Figure, ax: plt.Axes, save_path: str) -> None:
    fig.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close()


def main(args: argparse.Namespace):
    images, labels = load_images_with_label(args.input_dir)
    images_flatten = images.reshape(images.shape[0], -1)
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

        coords = model.fit_transform(images_flatten)

        fig, ax = visualize_reduction(labels, coords, figsize=(9, 6))
        save_figure(fig, ax, os.path.join(args.output_dir, model_name))

        fig, ax = visualize_image_distribution(images, coords, zoom=0.25, figsize=(40, 25))
        save_figure(fig, ax, os.path.join(args.output_dir, '{}_image'.format(model_name)))


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

    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.handlelength"] = 1

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    main(args)
