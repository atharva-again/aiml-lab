import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.style.use("seaborn-v0_8-whitegrid")

custom_cmap = ListedColormap(["#E8F4FD", "#BDE0FE"])


def plot_decision_boundary(ax, clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.6, cmap=custom_cmap, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors="#2D3748", linewidths=1.5, levels=[0.5])

    colors = ["#2563EB", "#DC2626"]
    for i, c in enumerate(colors):
        mask = y == i
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=c,
            edgecolors="white",
            s=50,
            alpha=0.9,
            linewidths=1,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.tick_params(labelsize=9)


X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_svm = SVC(kernel="linear", C=1.0)
linear_svm.fit(X_train, y_train)
acc_linear = accuracy_score(y_test, linear_svm.predict(X_test))

rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_svm.fit(X_train, y_train)
acc_rbf = accuracy_score(y_test, rbf_svm.predict(X_test))

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.patch.set_facecolor("#FAFAFA")
plot_decision_boundary(
    axes1[0], linear_svm, X, y, f"Linear Kernel (Acc: {acc_linear:.1%})"
)
plot_decision_boundary(axes1[1], rbf_svm, X, y, f"RBF Kernel (Acc: {acc_rbf:.1%})")
fig1.suptitle(
    "SVM: Linear vs RBF Kernel on Moons Dataset", fontsize=16, fontweight="bold"
)
fig1.text(
    0.5,
    0.02,
    "Linear fails on non-linear data | RBF handles curves via Gaussian kernel",
    ha="center",
    fontsize=10,
    style="italic",
    color="#4A5568",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("svm_kernels.png", dpi=150, facecolor="#FAFAFA")
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor("#FAFAFA")
C_values = [0.01, 1.0, 100.0]
C_labels = ["Underfit (C=0.01)", "Optimal (C=1.0)", "Strong (C=100)"]
for idx, (C, label) in enumerate(zip(C_values, C_labels)):
    clf = SVC(kernel="rbf", C=C, gamma="scale")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    plot_decision_boundary(axes2[idx], clf, X, y, f"{label}\nAccuracy: {acc:.1%}")
fig2.suptitle(
    "SVM: Effect of C Parameter (Penalty/Regularization)",
    fontsize=16,
    fontweight="bold",
)
fig2.text(
    0.5,
    0.02,
    "Low C = smooth boundary (underfit) | High C = strict margin (may overfit)",
    ha="center",
    fontsize=10,
    style="italic",
    color="#4A5568",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("svm_c_variation.png", dpi=150, facecolor="#FAFAFA")
plt.show()

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.patch.set_facecolor("#FAFAFA")
gamma_values = [0.1, 1.0, 10.0]
gamma_labels = ["Underfit (γ=0.1)", "Optimal (γ=1.0)", "Overfit (γ=10)"]
for idx, (gamma, label) in enumerate(zip(gamma_values, gamma_labels)):
    clf = SVC(kernel="rbf", C=1.0, gamma=gamma)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    plot_decision_boundary(axes3[idx], clf, X, y, f"{label}\nAccuracy: {acc:.1%}")
fig3.suptitle(
    "SVM: Effect of Gamma (Kernel Coefficient)", fontsize=16, fontweight="bold"
)
fig3.text(
    0.5,
    0.02,
    "Low γ = wide influence (smooth) | High γ = narrow influence (wiggly/overfit)",
    ha="center",
    fontsize=10,
    style="italic",
    color="#4A5568",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("svm_gamma_variation.png", dpi=150, facecolor="#FAFAFA")
plt.show()

print("Done! Generated: svm_kernels.png, svm_c_variation.png, svm_gamma_variation.png")
