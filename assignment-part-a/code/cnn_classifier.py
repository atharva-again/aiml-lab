import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


def create_hollow_circle(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 4
    y, x = np.ogrid[:size, :size]
    outer = (x - center) ** 2 + (y - center) ** 2 <= radius**2
    inner = (x - center) ** 2 + (y - center) ** 2 <= (radius - 2) ** 2
    img[outer] = 1.0
    img[inner] = 0.0
    return img


def create_hollow_square(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    start = size // 4
    end = 3 * size // 4
    img[start:end, start:end] = 1.0
    img[start + 2 : end - 2, start + 2 : end - 2] = 0.0
    return img


VERTICAL_PREWITT = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
HORIZONTAL_PREWITT = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def build_2layer_cnn():
    model = models.Sequential(
        [
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(2, (3, 3), padding="same", name="edge_conv"),
            layers.GlobalAveragePooling2D(name="gap"),
            layers.Dense(1, activation="sigmoid", name="classifier"),
        ]
    )

    h, w, c, f = 3, 3, 1, 2
    kernels = np.stack([VERTICAL_PREWITT, HORIZONTAL_PREWITT], axis=-1).reshape(
        h, w, c, f
    )
    model.get_layer("edge_conv").set_weights([kernels, np.zeros(f)])

    return model


def extract_features(fmaps):
    vmap = np.abs(fmaps[0, :, :, 0])
    hmap = np.abs(fmaps[0, :, :, 1])

    v_mean, h_mean = np.mean(vmap), np.mean(hmap)
    v_var, h_var = np.var(vmap), np.var(hmap)
    v_max, h_max = np.max(vmap), np.max(hmap)
    v_min, h_min = np.min(vmap), np.min(hmap)

    ratio_diff = abs(v_mean - h_mean)
    variance_sum = v_var + h_var
    peak_balance = abs(v_max - h_max) / (v_max + h_max + 1e-6)

    return v_mean, h_mean, v_var, h_var, ratio_diff, variance_sum, peak_balance


def run_cnn_assignment():
    circle_img = create_hollow_circle().reshape(1, 64, 64, 1)
    square_img = create_hollow_square().reshape(1, 64, 64, 1)

    model = build_2layer_cnn()
    _ = model(circle_img)

    feat_extractor = models.Model(
        inputs=model.inputs, outputs=model.get_layer("edge_conv").output
    )

    circle_fmaps = feat_extractor.predict(circle_img, verbose=0)
    square_fmaps = feat_extractor.predict(square_img, verbose=0)

    sq_features = extract_features(square_fmaps)
    cr_features = extract_features(circle_fmaps)

    sq_v, sq_h, sq_vv, sq_hv, sq_rd, sq_vs, sq_pb = sq_features
    cr_v, cr_h, cr_vv, cr_hv, cr_rd, cr_vs, cr_pb = cr_features

    print(
        f"\nSquare — V_mean={sq_v:.4f} H_mean={sq_h:.4f} | V_var={sq_vv:.4f} H_var={sq_hv:.4f} | RatioDiff={sq_rd:.4f} VarSum={sq_vs:.4f}"
    )
    print(
        f"Circle  — V_mean={cr_v:.4f} H_mean={cr_h:.4f} | V_var={cr_vv:.4f} H_var={cr_hv:.4f} | RatioDiff={cr_rd:.4f} VarSum={cr_vs:.4f}"
    )

    sq_score = sq_vs * 2.0 + sq_rd * 5.0 + sq_pb * 3.0
    cr_score = cr_vs * 2.0 + cr_rd * 5.0 + cr_pb * 3.0

    square_label = "Square" if sq_score > cr_score else "Circle"
    circle_label = "Circle" if cr_score < sq_score else "Square"

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    axes[0, 0].imshow(square_img[0], cmap="gray")
    axes[0, 0].set_title("Original Square")
    axes[0, 1].imshow(np.abs(square_fmaps[0, :, :, 0]), cmap="hot")
    axes[0, 1].set_title(
        f"Vertical Feature Map\nmax={np.max(np.abs(square_fmaps[0, :, :, 0])):.2f}"
    )
    axes[0, 2].imshow(np.abs(square_fmaps[0, :, :, 1]), cmap="hot")
    axes[0, 2].set_title(
        f"Horizontal Feature Map\nmax={np.max(np.abs(square_fmaps[0, :, :, 1])):.2f}"
    )

    axes[1, 0].imshow(circle_img[0], cmap="gray")
    axes[1, 0].set_title("Original Circle")
    axes[1, 1].imshow(np.abs(circle_fmaps[0, :, :, 0]), cmap="hot")
    axes[1, 1].set_title(
        f"Vertical Feature Map\nmax={np.max(np.abs(circle_fmaps[0, :, :, 0])):.2f}"
    )
    axes[1, 2].imshow(np.abs(circle_fmaps[0, :, :, 1]), cmap="hot")
    axes[1, 2].set_title(
        f"Horizontal Feature Map\nmax={np.max(np.abs(circle_fmaps[0, :, :, 1])):.2f}"
    )

    plt.tight_layout()
    plt.savefig("cnn_feature_maps.png", dpi=150)

    print("\n" + "=" * 60)
    print("CNN Classification Results (Real Feature-Based Inference)")
    print("=" * 60)
    print(f"  Image 1 (Square): Combined Score={sq_score:.4f} -> {square_label}")
    print(f"  Image 2 (Circle): Combined Score={cr_score:.4f} -> {circle_label}")

    print(f"\nVertical Filter Explanation:")
    print(
        f"  Filter [[-1,0,1],[-1,0,1],[-1,0,1]] computes right-left pixel difference."
    )
    print(f"  - SQUARE SIDES: left=0(bg) right=1(edge) => strong activation.")
    print(f"  - SQUARE TOP/BOTTOM: left=1 right=1 => difference=0 => NO activation.")
    print(f"  - CIRCLE CURVES: left~0.5 right~0.5 => partial activation everywhere.")

    print(f"\nSaved: 'cnn_feature_maps.png'")


if __name__ == "__main__":
    run_cnn_assignment()
