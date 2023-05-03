"""Analysis using CPU only."""
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Limit how many product items should be processed
USER_RECORDS_LIMIT = 100000  # TODO include in filenames to reduce necessary number of preprocessing
CLUSTER_SIZE = 100
PREPROCESSED_MATRICES = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "data", "preproccessed_" + str(USER_RECORDS_LIMIT) + ".npy")
)
PREPROCESSED_TFIDF_MATRICES = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "data", "preproccessed_tfidf_" + str(USER_RECORDS_LIMIT) + ".npy"
    )
)
PREPROCESSED_PCA_MATRICES = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "data", "preproccessed_pca_" + str(USER_RECORDS_LIMIT) + ".npy"
    )
)
ASSIGNED_LABELS = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "data", "assigned_labels_" + str(USER_RECORDS_LIMIT) + ".npy")
)
PRODUCTS_PCA_IMG = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "images", "products_pca_" + str(USER_RECORDS_LIMIT) + ".png")
)
CATEGORIES_PCA_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "categories_pca_" + str(USER_RECORDS_LIMIT) + ".png"
    )
)
PRODUCTS_HIST_IMG = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "images", "products_hist_" + str(USER_RECORDS_LIMIT) + ".svg")
)
CATEGORIES_HIST_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "categories_hist_" + str(USER_RECORDS_LIMIT) + ".svg"
    )
)
PRODUCTS_ELBOW_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "products_elbow_" + str(USER_RECORDS_LIMIT) + ".svg"
    )
)
CATEGORIES_ELBOW_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "categories_elbow_" + str(USER_RECORDS_LIMIT) + ".svg"
    )
)
PRODUCTS_DBSCAN_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "products_dbscan_" + str(USER_RECORDS_LIMIT) + ".png"
    )
)
CATEGORIES_DBSCAN_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "categories_dbscan_" + str(USER_RECORDS_LIMIT) + ".png"
    )
)
PRODUCTS_CLUSTER_HIST_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "products_cluster_hist_" + str(USER_RECORDS_LIMIT) + ".svg"
    )
)
CATEGORIES_CLUSTER_HIST_IMG = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "images", "categories_cluster_hist_" + str(USER_RECORDS_LIMIT) + ".svg"
    )
)
PCA_N_COMPONENTS = 3


if not os.path.exists(PREPROCESSED_MATRICES) or not os.path.exists(PREPROCESSED_TFIDF_MATRICES):
    print("Preprocessed files unavailable. Do preprocessing first.")

print("Preprocessed files found, loading...")
with open(PREPROCESSED_MATRICES, "rb") as f:
    print("Loading products users...")
    products_users = np.load(f)
    print("Loading products...")
    products = np.load(f)
    print("Loading products BoW...")
    products_bow = np.load(f)
    print("Loading categories users...")
    categories_users = np.load(f)
    print("Loading categories...")
    categories = np.load(f)
    print("Loading categories BoW..")
    categories_bow = np.load(f)

with open(PREPROCESSED_TFIDF_MATRICES, "rb") as f:
    print("Loading products TF-IDF...")
    products_tfidf = np.load(f)
    print("Loading categories TF-IDF...")
    categories_tfidf = np.load(f)

print("Loading DONE.")

print("Computing basic characteristics.")
# Calculate the figure size to match half an A4 page linewidth (approximately 3.35 inches)
FIGURE_WIDTH = 5  # inches
ASPECT_RATIO = 1 / 1
FIGURE_HEIGHT = FIGURE_WIDTH * ASPECT_RATIO  # Keep the aspect ratio
FIGURE_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)
for data in zip(
    [products_tfidf, categories_tfidf],
    ["Products TF-IDF distance", "Categories TF-IDF distance"],
    [PRODUCTS_HIST_IMG, CATEGORIES_HIST_IMG],
    [PRODUCTS_ELBOW_IMG, CATEGORIES_ELBOW_IMG],
):
    distances = 1 - (
        np.matmul(data[0], data[0].transpose())
        / (
            np.sqrt(np.sum(np.square(data[0]), 1)).reshape([1, data[0].shape[0]])
            * np.sqrt(np.sum(np.square(data[0]), 1)).reshape([data[0].shape[0], 1])
        )
    )

    print("Drawing an elbow rule.")
    # Create the scatter plot
    sorted_distances = np.sort(distances, axis=1)
    y = np.sort(sorted_distances[:, CLUSTER_SIZE])
    x = np.arange(y.shape[0])
    fig = plt.figure(figsize=FIGURE_SIZE)
    # plt.xscale("log")
    plt.plot(x, y)
    plt.grid(visible=True, which="both")
    # Set the title and axis labels
    plt.title(data[1] + " K=" + str(CLUSTER_SIZE), fontsize=12, pad=10)
    plt.xlabel("Points", fontsize=12, labelpad=5)
    plt.ylabel("Distance", fontsize=12, labelpad=5)
    # Adjust plot margins to ensure axis labels are within the image
    plt.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.1)
    # Save the figure as an SVG file
    # plt.show()
    plt.savefig(data[3], format="svg")
    # plt.close(fig)

    for i in tqdm(range(distances.shape[0])):
        distances[i, : i + 1] = np.nan
    counts, bins = np.histogram(distances[~np.isnan(distances)], bins=100)
    print("Drawing a histogram.")
    # Create the scatter plot
    fig = plt.figure(figsize=FIGURE_SIZE)
    plt.hist(bins[:-1], bins, weights=counts, log=True)
    plt.xlim([0, 1])
    plt.grid(visible=True, which="both")
    # Set the title and axis labels
    plt.title(data[1], fontsize=12, pad=10)
    plt.xlabel("Distance", fontsize=12, labelpad=5)
    plt.ylabel("Count", fontsize=12, labelpad=5)
    # Adjust plot margins to ensure axis labels are within the image
    plt.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.1)
    # Save the figure as an SVG file
    # plt.show()
    plt.savefig(data[2], format="svg")
    # plt.close(fig)

print("Computing PCA")
pca = PCA(n_components=PCA_N_COMPONENTS)
pca.fit(products_tfidf)
products_pca = pca.transform(products_tfidf)
products_tfidf_est = pca.inverse_transform(products_pca)
pca.fit(categories_tfidf)
categories_pca = pca.fit_transform(categories_tfidf)
categories_tfidf_est = pca.inverse_transform(categories_pca)
print("Saving PCA matrices")
with open(PREPROCESSED_PCA_MATRICES, "wb") as f:
    np.save(f, products_pca)
    np.save(f, products_tfidf_est)
    np.save(f, categories_pca)
    np.save(f, categories_tfidf_est)


print("Generating figures")
for data2 in zip(
    [products_pca, categories_pca],
    ["Products TF-IDF transformed to 3D using PCA", "Categories TF-IDF transformed to 3D using PCA"],
    [PRODUCTS_PCA_IMG, CATEGORIES_PCA_IMG],
    [(70, 20), (20, 60)],
):
    # Create the scatter plot
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(projection="3d")
    plt.plot(
        data2[0][:, 0],
        data2[0][:, 1],
        data2[0][:, 2],
        "o",
        markerfacecolor="b",
        markeredgecolor="k",
        markersize=5,
    )
    ax.set_xlabel("PCA dimension 1", fontsize=12, labelpad=5)
    ax.set_ylabel("PCA dimension 2", fontsize=12, labelpad=5)
    ax.set_zlabel("PCA dimension 3", fontsize=12, labelpad=5)
    ax.azim = data2[3][0]
    ax.elev = data2[3][1]
    ax.redraw_in_frame()
    plt.subplots_adjust(left=0.05, right=1, top=0.93, bottom=0.05)
    ax.redraw_in_frame()
    # Set the title and axis labels
    plt.title(data2[1], fontsize=12, pad=10)
    # Save the figure as an SVG file
    # fig.tight_layout()
    # plt.show()
    plt.savefig(data2[2], format="png")
    # plt.close(fig)

print("Starting with clustering")
label_list = []
for data3 in zip(
    [products_tfidf, categories_tfidf],
    [products_pca, categories_pca],
    ["Products TF-IDF DBSCAN clusters", "Categories TF-IDF DBSCAN clusters"],
    [PRODUCTS_DBSCAN_IMG, CATEGORIES_DBSCAN_IMG],
    [(70, 20), (20, 60)],
    [(0.3, CLUSTER_SIZE), (0.25, CLUSTER_SIZE)],
    [
        ("Products cluster histogram", PRODUCTS_CLUSTER_HIST_IMG),
        ("Category cluster histogram", CATEGORIES_CLUSTER_HIST_IMG),
    ],
):
    db = DBSCAN(eps=data3[5][0], min_samples=data3[5][1], metric="cosine", n_jobs=-1)
    db.fit(data3[0])
    labels = db.labels_
    label_list.append(labels)
    print("Cluster sizes:")
    for i in np.unique(labels)[1:]:
        print(str(i) + ". cluster - " + str(labels[labels == i].shape[0]))

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(projection="3d")
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # Draw noise
    class_member_mask = labels == -1
    xyz = data3[1][~core_samples_mask, :]
    plt.plot(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], "o", markerfacecolor="k", markeredgecolor="None", markersize=1, alpha=0.5
    )
    unique_labels.remove(-1)
    for k, col in zip(unique_labels, colors):
        class_member_mask = labels == k
        xyz = data3[1][class_member_mask & core_samples_mask, :]
        plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=5)

    # Set the title and axis labels
    plt.title(data3[2], fontsize=12, pad=10)
    ax.set_xlabel("PCA dimension 1", fontsize=12, labelpad=5)
    ax.set_ylabel("PCA dimension 2", fontsize=12, labelpad=5)
    ax.set_zlabel("PCA dimension 3", fontsize=12, labelpad=5)
    ax.azim = data3[4][0]
    ax.elev = data3[4][1]
    ax.redraw_in_frame()
    # plt.show()
    plt.savefig(data3[3], format="png")
    # plt.close(fig)

    # Visualization of cluster distribution
    counts, bins = np.histogram(labels[labels != -1], bins=(max(labels)))
    print("Drawing a histogram.")
    # Create the scatter plot
    fig = plt.figure(figsize=FIGURE_SIZE)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlim([0, max(labels)])
    plt.grid(visible=True, which="both")
    # Set the title and axis labels
    plt.title(data3[6][0], fontsize=12, pad=10)
    plt.xlabel("Cluster", fontsize=12, labelpad=5)
    plt.ylabel("Size", fontsize=12, labelpad=5)
    # Adjust plot margins to ensure axis labels are within the image
    plt.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.1)
    # Save the figure as an SVG file
    # plt.show()
    plt.savefig(data3[6][1], format="svg")
    # plt.close(fig)

# Saving labels
print("Saving Labels")
products_labels = label_list[0]
categories_labels = label_list[1]
with open(ASSIGNED_LABELS, "wb") as f:
    np.save(f, products_labels)
    np.save(f, categories_labels)

print("Finished.")
