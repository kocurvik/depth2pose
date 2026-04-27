import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "./")
from datasets.colmap_utils import qvec2rotmat, read_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare COLMAP and RealityScan reconstructions for one or more scenes."
    )
    parser.add_argument(
        "scene_names",
        nargs="+",
        help="One or more scene folder names to process.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory that contains the scene folders.",
    )
    return parser.parse_args()


def load_images(model_path):
    _, images, _ = read_model(model_path)
    poses = {}
    for image in images.values():
        rotation = qvec2rotmat(image.qvec)
        center = -rotation.T @ image.tvec
        poses[Path(image.name).name] = {
            "R": rotation,
            "C": center,
        }
    return poses


def rotation_angle_deg(rotation):
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def vector_angle_deg(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    cosine = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def compute_auc(errors, thresholds_deg):
    errors = np.sort(np.asarray(errors, dtype=float))
    if errors.size == 0:
        return [0.0 for _ in thresholds_deg]

    recall = (np.arange(errors.size) + 1) / errors.size
    recall = np.r_[0.0, recall]
    errors = np.r_[0.0, errors]

    aucs = []
    for threshold in thresholds_deg:
        last_index = np.searchsorted(errors, threshold, side="right")
        recall_segment = np.r_[recall[:last_index], recall[last_index - 1]]
        error_segment = np.r_[errors[:last_index], threshold]
        aucs.append(float(np.trapz(recall_segment, x=error_segment) / threshold))
    return aucs


def get_common_poses(base_folder):
    colmap_path = base_folder / "colmap_gt" / "sparse" 
    #colmap_path = base_folder / "colmap" / "sparse" / "0"
    #colmap_path = base_folder / "colmap" / "sparse" / "1"
    realityscan_path = base_folder / "realityscan_gt" / "undistorted" / "sparse"
    
    colmap_poses = load_images(colmap_path)
    realityscan_poses = load_images(realityscan_path)

    common_names = sorted(set(colmap_poses) & set(realityscan_poses))
    common_colmap = [colmap_poses[name] for name in common_names]
    common_realityscan = [realityscan_poses[name] for name in common_names]

    return common_names, common_colmap, common_realityscan


def compute_relative_pose_errors(poses1, poses2):
    rotation_errors = []
    translation_errors = []

    for i in range(len(poses1)):
        for j in range(i + 1, len(poses1)):
            rel_rot1 = poses1[j]["R"] @ poses1[i]["R"].T
            rel_rot2 = poses2[j]["R"] @ poses2[i]["R"].T
            rotation_errors.append(rotation_angle_deg(rel_rot2 @ rel_rot1.T))

            # Compare the relative translation direction in each camera's
            # local frame so the metric stays invariant to the arbitrary
            # global similarity transform of each reconstruction.
            rel_t1 = poses1[i]["R"] @ (poses1[j]["C"] - poses1[i]["C"])
            rel_t2 = poses2[i]["R"] @ (poses2[j]["C"] - poses2[i]["C"])
            translation_errors.append(vector_angle_deg(rel_t1, rel_t2))

    return {
        "mean_relative_rotation_deg": float(np.mean(rotation_errors)),
        "median_relative_rotation_deg": float(np.median(rotation_errors)),
        "mean_relative_translation_deg": float(np.mean(translation_errors)),
        "median_relative_translation_deg": float(np.median(translation_errors)),
        "num_pairs": len(rotation_errors),
    }


def similarity_align_points(points_src, points_dst):
    if len(points_src) != len(points_dst):
        raise ValueError("Point sets must contain the same number of points.")
    if len(points_src) < 3:
        raise ValueError("At least 3 points are required for similarity alignment.")

    src_center = points_src.mean(axis=0)
    dst_center = points_dst.mean(axis=0)

    src_centered = points_src - src_center
    dst_centered = points_dst - dst_center

    covariance = (dst_centered.T @ src_centered) / len(points_src)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.sum(singular_values * np.diag(correction)) / src_var
    translation = dst_center - scale * (rotation @ src_center)
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    return scale * (points @ rotation.T) + translation


def align_poses(poses_src, poses_dst):
    centers_src = np.stack([pose["C"] for pose in poses_src], axis=0)
    centers_dst = np.stack([pose["C"] for pose in poses_dst], axis=0)
    align_scale, align_rotation, align_translation = similarity_align_points(centers_src, centers_dst)

    aligned_poses = []
    for pose in poses_src:
        aligned_center = apply_similarity(
            pose["C"][None, :], align_scale, align_rotation, align_translation
        )[0]
        aligned_rotation = pose["R"] @ align_rotation.T
        aligned_poses.append({
            "R": aligned_rotation,
            "C": aligned_center,
        })

    return aligned_poses, align_scale, align_rotation, align_translation


def set_equal_axes_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius < 1e-12:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_plot_paths(base_folder, plot_dir):
    prefix = f"plot_{base_folder.name}_recon"
    return (
        plot_dir / f"{prefix}_center.png",
        plot_dir / f"{prefix}_cams.png",
    )


def get_output_dir(base_folder):
    output_dir = base_folder / "comparing_colmap_realityscan"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_metrics(metrics, metrics_path):
    lines = []
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            value = f"{value:.3f}"
        lines.append(f"{key}: {value}")
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_aligned_centers(base_folder, names, centers1, aligned_centers2, align_scale, plot_path):
    all_points = np.vstack([centers1, aligned_centers2])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(221, projection="3d")
    ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2], s=18, c="tab:blue", label="COLMAP")
    ax.scatter(
        aligned_centers2[:, 0],
        aligned_centers2[:, 1],
        aligned_centers2[:, 2],
        s=18,
        c="tab:red",
        label="RealityScan (aligned)",
    )
    ax.set_title("Aligned Camera Centers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes_3d(ax, all_points)
    ax.legend(loc="best")

    projection_specs = [
        (222, 0, 1, "XY"),
        (223, 0, 2, "XZ"),
        (224, 1, 2, "YZ"),
    ]
    for subplot_idx, axis_a, axis_b, title in projection_specs:
        ax_proj = fig.add_subplot(subplot_idx)
        ax_proj.scatter(centers1[:, axis_a], centers1[:, axis_b], s=18, c="tab:blue")
        ax_proj.scatter(aligned_centers2[:, axis_a], aligned_centers2[:, axis_b], s=18, c="tab:red")
        ax_proj.set_title(f"{title} Projection")
        ax_proj.set_xlabel(["X", "Y", "Z"][axis_a])
        ax_proj.set_ylabel(["X", "Y", "Z"][axis_b])
        ax_proj.set_aspect("equal", adjustable="box")
        ax_proj.grid(True, alpha=0.3)

    mean_center_error = np.mean(np.linalg.norm(centers1 - aligned_centers2, axis=1))
    fig.suptitle(
        f"{base_folder.name}: {len(names)} common cameras | "
        f"scale={align_scale:.4f} | mean center err={mean_center_error:.4f}"
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_camera_frustums(ax, poses, color, label, frustum_scale):
    image_plane = np.array([
        [-0.6, -0.4, 1.0],
        [0.6, -0.4, 1.0],
        [0.6, 0.4, 1.0],
        [-0.6, 0.4, 1.0],
    ])

    for idx, pose in enumerate(poses):
        center = pose["C"]
        cam_to_world = pose["R"].T
        world_corners = center + (image_plane * frustum_scale) @ cam_to_world.T
        loop = np.vstack([world_corners, world_corners[0]])

        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, alpha=0.55, linewidth=0.8)
        for corner in world_corners:
            segment = np.vstack([center, corner])
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, alpha=0.4, linewidth=0.8)

        forward = center + cam_to_world[:, 2] * frustum_scale * 1.4
        axis = np.vstack([center, forward])
        ax.plot(
            axis[:, 0],
            axis[:, 1],
            axis[:, 2],
            color=color,
            alpha=0.9,
            linewidth=1.4,
            label=label if idx == 0 else None,
        )


def plot_aligned_cameras(base_folder, names, poses1, aligned_poses2, align_scale, plot_path):
    centers1 = np.stack([pose["C"] for pose in poses1], axis=0)
    aligned_centers2 = np.stack([pose["C"] for pose in aligned_poses2], axis=0)
    all_points = np.vstack([centers1, aligned_centers2])
    extent = np.max(all_points.max(axis=0) - all_points.min(axis=0))
    frustum_scale = max(extent * 0.03, 1e-3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    draw_camera_frustums(ax, poses1, "tab:blue", "COLMAP", frustum_scale)
    draw_camera_frustums(ax, aligned_poses2, "tab:red", "RealityScan (aligned)", frustum_scale)

    set_equal_axes_3d(ax, all_points)
    ax.set_title(
        f"{base_folder.name}: aligned camera frustums | "
        f"{len(names)} common cameras | scale={align_scale:.4f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_reconstructions(base_folder, names, poses1, poses2, plot_dir):
    centers1 = np.stack([pose["C"] for pose in poses1], axis=0)
    aligned_poses2, align_scale, _, _ = align_poses(poses2, poses1)
    aligned_centers2 = np.stack([pose["C"] for pose in aligned_poses2], axis=0)
    center_plot_path, camera_plot_path = get_plot_paths(base_folder, plot_dir)
    plot_aligned_centers(base_folder, names, centers1, aligned_centers2, align_scale, center_plot_path)
    plot_aligned_cameras(base_folder, names, poses1, aligned_poses2, align_scale, camera_plot_path)


def compute_absolute_pose_errors(poses1, poses2):
    centers1 = np.stack([pose["C"] for pose in poses1], axis=0)
    aligned_poses2, align_scale, align_rotation, align_translation = align_poses(poses2, poses1)
    aligned_centers2 = np.stack([pose["C"] for pose in aligned_poses2], axis=0)

    center_errors = np.linalg.norm(centers1 - aligned_centers2, axis=1)

    rotation_errors = []
    for pose1, aligned_pose2 in zip(poses1, aligned_poses2):
        rotation_errors.append(rotation_angle_deg(aligned_pose2["R"] @ pose1["R"].T))

    return {
        "alignment_scale": float(align_scale),
        "mean_absolute_center_error": float(np.mean(center_errors)),
        "mean_absolute_rotation_deg": float(np.mean(rotation_errors)),
        "num_cameras": len(poses1),
    }


def compare_reconstructions(base_folder, output_dir=None):
    common_names, poses1, poses2 = get_common_poses(base_folder)
    if output_dir is not None:
        plot_aligned_reconstructions(base_folder, common_names, poses1, poses2, output_dir)

    relative_metrics = compute_relative_pose_errors(poses1, poses2)
    absolute_metrics = compute_absolute_pose_errors(poses1, poses2)

    metrics = {
        "base_folder": str(base_folder),
        "num_common_cameras": len(common_names),
    }
    metrics.update(relative_metrics)
    metrics.update(absolute_metrics)
    return metrics


def main():
    args = parse_args()

    for scene_name in args.scene_names:
        base_folder = args.root / scene_name
        output_dir = get_output_dir(base_folder)
        metrics = compare_reconstructions(base_folder, output_dir=output_dir)
        write_metrics(metrics, output_dir / "metrics.txt")
        print(metrics)


if __name__ == "__main__":
    main()
