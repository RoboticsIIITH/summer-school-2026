"""
Coordinate Transforms with nuScenes
====================================
Visualize 6 cameras + LIDAR_TOP in a fixed global frame using Rerun.

The global frame is fixed. At each timestamp, every sensor has a pose
(rotation + translation) relative to this global frame. We transform all
sensor data into the global frame before logging it.

Transform chain:  sensor -> ego_vehicle -> global
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm

DATA_ROOT = Path("./data")

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


# ═══════════════════════════════════════════════════════════════════
#  TASK 1 — Build a 4×4 homogeneous transform matrix
# ═══════════════════════════════════════════════════════════════════


def build_transform(R, t):
    """
    Parameters
    ----------
    R : ndarray
        Shape (3, 3). Rotation matrix.
    t : ndarray
        Shape (3,). Translation vector.

    Returns
    -------
    T : ndarray
        Shape (4, 4). Homogeneous transform matrix with R in the top-left
        3×3 block and t in the top-right 3×1 column. The bottom row is
        [0, 0, 0, 1].
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
#  TASK 2 — Apply a 4×4 transform to 3D points
# ═══════════════════════════════════════════════════════════════════


def transform_points(T, points):
    """
    Parameters
    ----------
    T : ndarray
        Shape (4, 4). Homogeneous transform matrix.
    points : ndarray
        Shape (3, N). Point cloud in the source frame.

    Returns
    -------
    transformed : ndarray
        Shape (3, N). Point cloud in the target frame.
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
#  TASK 3 — Decompose a 4×4 into rotation + translation
# ═══════════════════════════════════════════════════════════════════


def transform_to_Rt(T):
    """
    Parameters
    ----------
    T : ndarray
        Shape (4, 4). Homogeneous transform matrix.

    Returns
    -------
    R : ndarray
        Shape (3, 3). Rotation matrix.
    t : ndarray
        Shape (3,). Translation vector.
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
#  TASK 4 — Build the transform from sensor frame to global frame
#
#  At each timestamp every sensor has two records that define its pose
#  relative to the fixed global frame:
#
#    calibrated_sensor:  rotation + translation  (sensor → ego)
#    ego_pose:           rotation + translation  (ego → global)
#
#  Compose them to get sensor → global.
# ═══════════════════════════════════════════════════════════════════


def sensor_to_global(nusc, sample_data_token):
    """
    Parameters
    ----------
    nusc : NuScenes
        The loaded nuScenes dataset object.
    sample_data_token : str
        Token identifying one sensor reading at one timestamp.

    Returns
    -------
    T_sensor_global : ndarray
        Shape (4, 4). Homogeneous transform that maps a point in the
        sensor's local coordinate frame to the global coordinate frame.
    """
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    raise NotImplementedError


def quaternion_to_rotation_matrix(q):
    """Unit quaternion [w, x, y, z] → 3×3 rotation matrix (closed-form)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


_NUSCENES_COLORMAP = get_colormap()


def make_annotation_context(nusc):
    """Build Rerun AnnotationContext from nuScenes categories."""
    annotations = []
    for i, cat in enumerate(nusc.category):
        rgb = _NUSCENES_COLORMAP.get(cat["name"], (128, 128, 128))
        annotations.append(rr.AnnotationInfo(id=i, label=cat["name"], color=rgb))
    return rr.AnnotationContext(annotations)


def load_labels(nusc, lidar_sample_data_token):
    """Return (N,) uint8 class IDs for each LiDAR point, or None."""
    for ls in nusc.lidarseg:
        if ls["sample_data_token"] == lidar_sample_data_token:
            return np.fromfile(str(DATA_ROOT / ls["filename"]), dtype=np.uint8)
    return None


def collect_trajectory(nusc, scene):
    """Collect ego vehicle positions across all samples."""
    positions = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        T = build_transform(
            quaternion_to_rotation_matrix(ego["rotation"]),
            np.array(ego["translation"]),
        )
        _, t = transform_to_Rt(T)
        positions.append(t)
        token = sample["next"]
    return np.array(positions)


def load_sweep(nusc, sample_token):
    """Load one LiDAR sweep and transform it to the global frame."""
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(str(DATA_ROOT / lidar_sd["filename"]))
    T = sensor_to_global(nusc, sample["data"]["LIDAR_TOP"])
    pts = transform_points(T, pc.points[:3, :]).T
    labels = load_labels(nusc, sample["data"]["LIDAR_TOP"])
    return pts, labels


def log_sample(nusc, sample_token):
    """Log LiDAR + 6 cameras into Rerun for a single timestamp."""
    sample = nusc.get("sample", sample_token)

    # --- LiDAR point cloud ---
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(str(DATA_ROOT / lidar_sd["filename"]))
    T = sensor_to_global(nusc, sample["data"]["LIDAR_TOP"])
    pts = transform_points(T, pc.points[:3, :]).T

    reflectance = np.sqrt(np.maximum(pc.points[3, :], 0))
    r_min, r_max = reflectance.min(), reflectance.max()
    colors = np.zeros((len(pts), 3), dtype=np.uint8)
    if r_max > r_min:
        idx = ((reflectance - r_min) / (r_max - r_min) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(idx, cv2.COLORMAP_TURBO)[:, 0, ::-1]

    rr.log("world/lidar", rr.Points3D(pts, colors=colors))

    # --- 6 cameras ---
    for ch in CAMERAS:
        cam_sd = nusc.get("sample_data", sample["data"][ch])
        cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"], dtype=np.float32)

        T_cam = sensor_to_global(nusc, sample["data"][ch])
        R_world_from_cam, t_world_from_cam = transform_to_Rt(T_cam)

        rr.log(
            f"world/{ch}",
            rr.Transform3D(
                translation=t_world_from_cam,
                mat3x3=R_world_from_cam,
            ),
        )
        rr.log(
            f"world/{ch}",
            rr.Pinhole(
                image_from_camera=K,
                camera_xyz=rr.ViewCoordinates.RDF,
                resolution=[cam_sd["width"], cam_sd["height"]],
                image_plane_distance=1.0,
            ),
        )

        img = cv2.cvtColor(
            cv2.imread(str(DATA_ROOT / cam_sd["filename"])),
            cv2.COLOR_BGR2RGB,
        )
        rr.log(f"world/{ch}/image", rr.Image(img))


def make_blueprint():
    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
        rrb.TimePanel(
            state=rrb.PanelState.Collapsed,
            playback_speed=2.0,
        ),
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world",
                    contents=[
                        "world/lidar_map/**",
                        "world/CAM_FRONT/**",
                        "world/CAM_FRONT_LEFT/**",
                        "world/CAM_FRONT_RIGHT/**",
                        "world/CAM_BACK/**",
                        "world/CAM_BACK_LEFT/**",
                        "world/CAM_BACK_RIGHT/**",
                        "world/trajectory/**",
                    ],
                    name="LiDAR Map (all timestamps)",
                ),
                rrb.Spatial3DView(
                    origin="world",
                    contents=[
                        "world/lidar/**",
                        "world/CAM_FRONT/**",
                        "world/CAM_FRONT_LEFT/**",
                        "world/CAM_FRONT_RIGHT/**",
                        "world/CAM_BACK/**",
                        "world/CAM_BACK_LEFT/**",
                        "world/CAM_BACK_RIGHT/**",
                        "world/trajectory/**",
                    ],
                    name="Current Frame",
                ),
                column_shares=[1, 1],
            ),
            rrb.Horizontal(
                *(
                    rrb.Spatial2DView(origin=f"world/{ch}/image", name=ch)
                    for ch in CAMERAS
                ),
            ),
            row_shares=[3, 2],
        ),
        auto_views=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Open Rerun viewer")
    parser.add_argument("--scene", type=int, default=0, help="Scene index to visualize")
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    rr.init("nuscenes_coordinate_transforms")

    if args.viewer:
        rr.spawn()
    else:
        rr.save("output.rrd")

    rr.send_blueprint(make_blueprint(), make_active=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", make_annotation_context(nusc), static=True)

    scene = nusc.scene[args.scene]
    print(f"Scene {args.scene}: {scene['name']}")

    # Collect all sample tokens
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]

    # Per-timestamp: log current + accumulate world map + trajectory
    map_pts = []
    map_ids = []
    traj_pts = []
    for idx, token in enumerate(tqdm(tokens, desc="Samples")):
        rr.set_time("sample_index", sequence=idx)

        # Current sweep (intensity)
        log_sample(nusc, token)

        # Accumulate into world map (semantic)
        pts, labels = load_sweep(nusc, token)
        map_pts.append(pts)
        if labels is not None:
            map_ids.append(labels)
        rr.log(
            "world/lidar_map",
            rr.Points3D(
                np.vstack(map_pts),
                class_ids=np.concatenate(map_ids) if map_ids else None,
            ),
        )

        # Accumulate trajectory
        sample = nusc.get("sample", token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        T = build_transform(
            quaternion_to_rotation_matrix(ego["rotation"]),
            np.array(ego["translation"]),
        )
        _, t = transform_to_Rt(T)
        traj_pts.append(t.copy())
        rr.log("world/trajectory", rr.LineStrips3D([np.array(traj_pts)]))

    save_path = Path("output.rrd").resolve()
    print(f"\nSaved to {save_path}")
    print(f"Open with: rerun {save_path}")


if __name__ == "__main__":
    main()
