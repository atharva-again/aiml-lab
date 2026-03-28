import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.style.use("seaborn-v0_8-whitegrid")


def kmeans(X, k, max_iter=100, random_state=None):
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sum((X - centroids[i]) ** 2, axis=1)

        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = X[np.random.choice(n_samples)]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    inertia = 0
    for i in range(k):
        inertia += np.sum((X[labels == i] - centroids[i]) ** 2)

    return labels, centroids, inertia


X, _ = make_blobs(n_samples=300, centers=4, cluster_std=2.0, random_state=42)

k_range = range(1, 11)
inertias = []

for k in k_range:
    _, _, inertia = kmeans(X, k, random_state=42)
    inertias.append(inertia)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.patch.set_facecolor("#FAFAFA")

ax.plot(k_range, inertias, "o-", linewidth=2.5, markersize=10, color="#2563EB")
ax.set_xlabel("Number of Clusters (K)", fontsize=13, fontweight="bold")
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=13, fontweight="bold")
ax.set_title(
    "K-Means: Elbow Method to Find Optimal K", fontsize=16, fontweight="bold", pad=15
)
ax.set_xticks(k_range)
ax.grid(True, alpha=0.3)

elbow_k = 4
ax.axvline(x=elbow_k, color="#DC2626", linestyle="--", linewidth=2, alpha=0.7)
ax.annotate(
    f"Elbow at K={elbow_k}",
    xy=(elbow_k, inertias[elbow_k - 1]),
    xytext=(elbow_k + 1.5, inertias[elbow_k - 1] + 3000),
    fontsize=12,
    fontweight="bold",
    color="#DC2626",
    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2),
)

ax.text(
    0.5,
    0.02,
    "Sharp decrease stops after K=4 → Optimal clusters = 4",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
    style="italic",
    color="#4A5568",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("kmeans_elbow.png", dpi=150, facecolor="#FAFAFA")
plt.show()

k = 4

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.patch.set_facecolor("#FAFAFA")

colors_palette = ["#2563EB", "#DC2626", "#059669", "#D97706"]

for run in range(5):
    labels, centroids, inertia = kmeans(X, k, random_state=run * 10)

    for i in range(k):
        mask = labels == i
        axes[run].scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors_palette[i],
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    axes[run].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="white",
        marker="X",
        s=250,
        edgecolors="black",
        linewidths=2,
    )

    axes[run].set_title(
        f"Run {run + 1}\nInertia: {inertia:.0f}", fontsize=12, fontweight="bold"
    )
    axes[run].set_xlabel("Feature 1", fontsize=10)
    axes[run].set_ylabel("Feature 2", fontsize=10)
    axes[run].set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    axes[run].set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    axes[run].grid(True, alpha=0.3)

fig.suptitle(
    "K-Means: Different Initializations (K=4)", fontsize=16, fontweight="bold", y=1.05
)

fig.text(
    0.5,
    -0.02,
    "Each run uses different random seeds → Different initial centroids → Different final clusters",
    ha="center",
    fontsize=11,
    style="italic",
    color="#4A5568",
)

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig("kmeans_multiple_runs.png", dpi=150, facecolor="#FAFAFA")
plt.show()

print("Done! Generated: kmeans_elbow.png, kmeans_multiple_runs.png")
