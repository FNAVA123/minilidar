#!/usr/bin/env python3
"""Experimental monocular mapping: metric Depth Anything V2 + ORB/PnP + Open3D.

World axes are those of the first camera: X right, Y down, Z forward.
No ground truth, loop closure, global optimization, or measured depth is used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def backproject(pixels, depth, K):
    """Pixels and their optical-axis Z depths -> camera XYZ."""
    return np.column_stack(((pixels[:, 0] - K[0, 2]) * depth / K[0, 0],
                            (pixels[:, 1] - K[1, 2]) * depth / K[1, 1], depth))


def world_from_current(world_from_previous, current_from_previous):
    # solvePnP maps reference-camera 3D into current-camera coordinates.
    return world_from_previous @ np.linalg.inv(current_from_previous)


def estimate_motion(points, pixels, K, min_inliers=20, max_step=5.0, max_rotation=30.0):
    if len(points) < min_inliers:
        return None, {"reason": "few_matches", "matches": len(points)}
    points = np.ascontiguousarray(points, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, pixels, K, None, iterationsCount=200, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    n = 0 if inliers is None else len(inliers)
    stats = {"matches": len(points), "inliers": n}
    if not ok or n < min_inliers or n / len(points) < 0.3:
        return None, {**stats, "reason": "few_inliers"}
    ids = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(points[ids], pixels[ids], K, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(points[ids], rvec, tvec, K, None)[0].reshape(-1, 2)
    error = float(np.median(np.linalg.norm(projected - pixels[ids], axis=1)))
    step = float(np.linalg.norm(tvec))
    angle = float(np.degrees(np.linalg.norm(rvec)))
    front = float(np.mean((points[ids] @ rotation.T + tvec.ravel())[:, 2] > 0))
    stats.update(median_reprojection_px=error, step_estimated_m=step, rotation_deg=angle)
    if not np.isfinite([error, step, angle]).all() or error > 3 or front < 0.95:
        return None, {**stats, "reason": "invalid_geometry"}
    if step > max_step or angle > max_rotation:
        return None, {**stats, "reason": "motion_limit"}
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, tvec.ravel()
    return transform, {**stats, "reason": "accepted"}


def camera_parameters(calibration, original_size, resized_size, fov=None):
    import yaml
    ow, oh = original_size
    w, h = resized_size
    if calibration:
        text = Path(calibration).read_text()
        if text.startswith("%YAML:1.0"):
            text = text.split("\n", 1)[1]
        settings = yaml.safe_load(text)
        if (int(settings["Camera.width"]), int(settings["Camera.height"])) != (ow, oh):
            raise ValueError("La resolución de calibración no coincide con el video original")
        K = np.array([[settings["Camera.fx"], 0, settings["Camera.cx"]],
                      [0, settings["Camera.fy"], settings["Camera.cy"]], [0, 0, 1.]], dtype=float)
        dist = np.array([settings.get(f"Camera.{key}", 0) for key in ("k1", "k2", "p1", "p2", "k3")], dtype=float)
    elif fov is not None:
        focal = ow / (2 * np.tan(np.deg2rad(fov) / 2))
        K = np.array([[focal, 0, (ow - 1) / 2], [0, focal, (oh - 1) / 2], [0, 0, 1.]])
        dist = np.zeros(5)
    else:
        raise ValueError("Usa --calibration o proporciona explícitamente --fov-deg para una aproximación")
    K[0] *= w / ow
    K[1] *= h / oh
    return K, dist


class MetricDepth:
    def __init__(self, scene, repo, input_size):
        import torch
        from huggingface_hub import hf_hub_download
        package = Path(repo).resolve() / "metric_depth"
        if not (package / "depth_anything_v2/dpt.py").is_file():
            raise FileNotFoundError(f"Falta el código oficial de profundidad métrica en {package}")
        sys.path.insert(0, str(package))
        from depth_anything_v2.dpt import DepthAnythingV2
        dataset, title, max_depth = ("hypersim", "Hypersim", 20) if scene == "indoor" else ("vkitti", "VKITTI", 80)
        self.model_id = f"depth-anything/Depth-Anything-V2-Metric-{title}-Small"
        filename = f"depth_anything_v2_metric_{dataset}_vits.pth"
        local_checkpoint = ROOT / "modelos" / filename
        checkpoint = str(local_checkpoint) if local_checkpoint.is_file() else hf_hub_download(self.model_id, filename)
        self.revision = "bundled-local-weights" if local_checkpoint.is_file() else Path(checkpoint).parent.name
        self.sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=max_depth)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.input_size = input_size
        print(f"Modelo: {self.model_id}; dispositivo: {self.device}", flush=True)

    def predict(self, image):
        import torch
        with torch.inference_mode():
            result = self.model.infer_image(image, self.input_size)
        if not np.isfinite(result).all():
            raise RuntimeError("El modelo devolvió profundidades no finitas")
        return result


def make_cloud(image, depth, K, pose, args):
    import open3d as o3d
    v, u = np.mgrid[2:depth.shape[0]-2:args.pixel_step, 2:depth.shape[1]-2:args.pixel_step]
    z = depth[v, u]
    # Avoid joining surfaces across strong predicted depth discontinuities.
    dx = np.abs(depth[v, u + 1] - depth[v, u - 1])
    dy = np.abs(depth[v + 1, u] - depth[v - 1, u])
    valid = (z > args.min_depth) & (z < args.max_depth) & (dx + dy < z * 0.15)
    xyz = backproject(np.column_stack((u[valid], v[valid])), z[valid], K)
    xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(image[v[valid], u[valid], ::-1].astype(float) / 255)
    return cloud.voxel_down_sample(args.voxel)


class FeatureExtractor:
    def __init__(self, backend, repo):
        self.backend = backend
        self.device = "cpu"
        self.weights_sha256 = None
        if backend == "orb":
            self.model = cv2.ORB_create(nfeatures=2000, fastThreshold=12, edgeThreshold=15)
        else:
            import torch
            repo = Path(repo).resolve()
            sys.path.insert(0, str(repo))
            from modules.xfeat import XFeat
            checkpoint = repo / "weights/xfeat.pt"
            self.weights_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = XFeat(weights=weights, top_k=2000)
            self.device = str(self.model.dev)
        print(f"Características: {backend}; dispositivo: {self.device}", flush=True)

    def extract(self, image):
        if self.backend == "orb":
            return self.model.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        result = self.model.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0]
        pixels = result["keypoints"].cpu().numpy()
        descriptors = result["descriptors"].cpu().numpy().astype(np.float32)
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pixels]
        return keypoints, descriptors if len(descriptors) else None


def descriptor_matches(old, current, backend):
    if backend == "xfeat":
        # XFeat descriptors have unit L2 norm. Mutual NN plus cosine >= 0.8.
        matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(old, current)
        return [match for match in matches if match.distance <= np.sqrt(2 * (1 - .8))]
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old, current, k=2)
    return [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .8 * pair[1].distance]


def match_reference(reference, keypoints, descriptors, K, args):
    old_keypoints, old_descriptors, depth, _ = reference
    if old_descriptors is None or descriptors is None:
        return None, {"reason": "no_descriptors"}
    matches = descriptor_matches(old_descriptors, descriptors, args.features)
    # A current feature may only receive one 3D correspondence.
    unique = {}
    for match in sorted(matches, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    if not matches:
        return None, {"reason": "no_matches"}
    old_pixels = np.array([old_keypoints[m.queryIdx].pt for m in matches])
    new_pixels = np.array([keypoints[m.trainIdx].pt for m in matches])
    uv = np.rint(old_pixels).astype(int)
    z = depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(z) & (z > args.min_depth) & (z < args.max_depth)
    return estimate_motion(backproject(old_pixels[valid], z[valid], K), new_pixels[valid], K,
                           args.min_inliers, args.max_step, args.max_rotation)


class ReferenceTracker:
    """Keep original last-frame tracking; consult strong backups only on failure."""
    def __init__(self, K, size, fps, args, matcher=match_reference):
        self.K, self.size, self.fps, self.args = K, size, fps, args
        self.matcher = matcher
        self.primary = None
        self.bank = []
        self.last_pose, self.last_frame = None, None

    def strong_features(self, keypoints, descriptors):
        if descriptors is None or len(keypoints) < self.args.min_reference_features:
            return False
        pixels = np.array([keypoint.pt for keypoint in keypoints])
        coverage = np.prod(np.ptp(pixels, axis=0) / np.array(self.size))
        return coverage >= .1

    def locate(self, keypoints, descriptors, frame, recovering=False):
        if self.primary is None:
            return np.eye(4), {"reason": "initial"}
        # Preserve the original estimator, thresholds and frame-to-frame chain.
        # A previous failure must not tighten an otherwise valid normal match.
        reference_frame, reference = self.primary
        transform, stats = self.matcher(reference, keypoints, descriptors, self.K, self.args)
        if transform is not None:
            return world_from_current(reference[3], transform), {
                **stats, "reference_frame": reference_frame, "reference_attempts": 1,
                "recovered": recovering, "backup_used": False}
        candidates = [r for r in reversed(self.bank) if r[0] != reference_frame]
        attempts = 1
        last_stats = {**stats, "reference_frame": reference_frame}
        primary_failure = stats["reason"]
        for reference_frame, reference in candidates:
            gap = frame - reference_frame
            if gap / self.fps > self.args.reference_max_age:
                continue
            attempts += 1
            options = argparse.Namespace(**vars(self.args))
            # The reference can be older than one sampling interval.
            interval = max(1., gap / self.args.stride)
            options.max_step *= interval
            options.max_rotation = min(180., options.max_rotation * interval)
            options.min_inliers = max(options.min_inliers, self.args.recovery_min_inliers)
            transform, stats = self.matcher(reference, keypoints, descriptors, self.K, options)
            last_stats = {**stats, "reference_frame": reference_frame}
            if transform is None:
                continue
            if stats.get("inliers", 0) / max(1, stats.get("matches", 0)) < .4:
                last_stats["reason"] = "weak_recovery_geometry"
                continue
            pose = world_from_current(reference[3], transform)
            since_last = max(1., (frame - self.last_frame) / self.args.stride)
            step = float(np.linalg.norm(pose[:3, 3] - self.last_pose[:3, 3]))
            rotation = pose[:3, :3] @ self.last_pose[:3, :3].T
            angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))
            if step > self.args.max_step * since_last or angle > self.args.max_rotation * since_last:
                last_stats["reason"] = "recovery_pose_jump"
                continue
            return pose, {**stats, "reference_frame": reference_frame, "reference_attempts": attempts,
                          "recovered": True, "backup_used": True, "primary_failure": primary_failure}
        return None, {**last_stats, "reference_attempts": attempts, "primary_failure": primary_failure}

    def accept(self, frame, keypoints, descriptors, depth, pose, stats):
        # Every accepted frame updates the original tracking chain. Quality
        # filtering applies only to admission into the recovery bank.
        self.last_pose, self.last_frame = pose.copy(), frame
        reference = (frame, (keypoints, descriptors, depth, pose.copy()))
        self.primary = reference
        strong = self.strong_features(keypoints, descriptors)
        strong &= stats["reason"] == "initial" or stats.get("inliers", 0) >= self.args.min_reference_inliers
        if not strong:
            return False
        if not self.bank or (frame - self.bank[-1][0]) / self.fps >= self.args.reference_interval:
            self.bank.append(reference)
            self.bank = self.bank[-self.args.reference_count:]
        return True


def save_outputs(output, cloud, rows, summary):
    import open3d as o3d
    if len(cloud.points) > 30:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if not o3d.io.write_point_cloud(str(output / "nube_3d.ply"), cloud):
        raise RuntimeError("No se pudo guardar la nube")
    fields = ["frame", "timestamp_s", "x", "y", "z"] + [f"T{i}{j}" for i in range(4) for j in range(4)]
    with (output / "trayectoria.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)
    path = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    summary["points_saved"] = len(cloud.points)
    summary["accepted_frames"] = len(rows)
    summary["estimated_path_length_m"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    (output / "resumen.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_previews(output, cloud, rows)


def render_previews(output, cloud, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    ids = np.random.default_rng(0).choice(len(xyz), min(len(xyz), 60000), replace=False)
    points, color = xyz[ids], colors[ids]
    path = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(points[:, 2], points[:, 0], c=color, s=0.5, alpha=0.5, rasterized=True)
    ax.plot(path[:, 2], path[:, 0], color="crimson", linewidth=2, label="Cámara estimada")
    ax.scatter(path[0, 2], path[0, 0], color="limegreen", s=45, label="Inicio", zorder=5)
    ax.set(xlabel="Z (m estimados)", ylabel="X (m estimados)", aspect="equal",
           title="Mini-LiDAR · Proyección X–Z del sistema de la primera cámara")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.text(.5, .02, "Profundidad aprendida · Sin alineación al suelo ni cierre de bucles · No es un mapa de ocupación", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "vista_superior.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=color, s=0.4, depthshade=False, rasterized=True)
    ax.plot(path[:, 0], path[:, 2], -path[:, 1], color="crimson", linewidth=2)
    ax.set(xlabel="X", ylabel="Z", zlabel="−Y", title="Nube fusionada · Escala métrica estimada")
    spans = np.ptp(np.column_stack((points[:, 0], points[:, 2], -points[:, 1])), axis=0)
    ax.set_box_aspect(np.maximum(spans, .1))
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=28, azim=-65)
    fig.tight_layout()
    fig.savefig(output / "nube_3d_preview.png", dpi=140)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path, help="Camera.fx/fy/cx/cy, width/height y distorsión en YAML")
    calibration.add_argument("--fov-deg", type=float, help="FOV horizontal aproximado; reduce la fiabilidad métrica")
    parser.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--features", choices=["orb", "xfeat"], default="orb")
    parser.add_argument("--xfeat-repo", type=Path, default=ROOT / "vendor/xfeat")
    parser.add_argument("--depth-repo", type=Path, default=ROOT / "vendor/depth-anything-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mini_lidar")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=392, help="Lado corto del tensor del modelo; conserva aspecto")
    parser.add_argument("--voxel", type=float, default=.08)
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=.25)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-step", type=float, default=5.)
    parser.add_argument("--max-rotation", type=float, default=30.)
    parser.add_argument("--max-failures", type=int, default=None, help="Límite opcional de fallos; por defecto se usa --recovery-seconds")
    parser.add_argument("--recovery-seconds", type=float, default=3.)
    parser.add_argument("--reference-count", type=int, default=12)
    parser.add_argument("--reference-interval", type=float, default=.5)
    parser.add_argument("--reference-max-age", type=float, default=8.)
    parser.add_argument("--min-reference-features", type=int, default=200)
    parser.add_argument("--min-reference-inliers", type=int, default=30)
    parser.add_argument("--recovery-min-inliers", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=1000000)
    parser.add_argument("--max-frames", type=int, default=0, help="Fotogramas muestreados, 0 para todo el video")
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args(argv)
    args.max_depth = args.max_depth if args.max_depth is not None else (15. if args.scene == "indoor" else 40.)
    if min(args.width, args.stride, args.input_size, args.pixel_step, args.max_points, args.reference_count,
           args.min_reference_features, args.min_reference_inliers, args.recovery_min_inliers) <= 0:
        parser.error("Tamaños, pasos y límites deben ser positivos")
    if args.max_failures is not None and args.max_failures <= 0:
        parser.error("--max-failures debe ser positivo")
    if min(args.recovery_seconds, args.reference_interval, args.reference_max_age) <= 0:
        parser.error("Los intervalos de recuperación y referencias deben ser positivos")
    if not (0 < args.min_depth < args.max_depth) or not (0 < args.voxel) or args.min_inliers < 6:
        parser.error("Rango de profundidad, vóxel o número de inliers inválido")
    if args.max_frames < 0 or args.start_frame < 0 or args.max_step <= 0 or args.max_rotation <= 0:
        parser.error("Límites inválidos")
    if args.fov_deg is not None and not 1 < args.fov_deg < 179:
        parser.error("El FOV debe estar entre 1 y 179 grados")
    return args


def main(argv=None):
    args = parse_args(argv)
    import open3d as o3d
    cv2.setRNGSeed(0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("resumen.json", "nube_3d.ply", "tracking.jsonl")):
        raise FileExistsError("El directorio ya contiene resultados; usa otro --output")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, first = cap.read()
    if not ok:
        raise ValueError("El video no tiene un fotograma inicial legible")
    oh, ow = first.shape[:2]
    size = (min(args.width, ow), max(8, round(oh * min(args.width, ow) / ow)))
    K, dist = camera_parameters(args.calibration, (ow, oh), size, args.fov_deg)
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("El video no proporciona FPS válidos")
    depth_model = MetricDepth(args.scene, args.depth_repo, args.input_size)
    extractor = FeatureExtractor(args.features, args.xfeat_repo)
    tracker = ReferenceTracker(K, size, fps, args)
    failure_limit = args.max_failures or max(1, int(np.ceil(args.recovery_seconds * fps / args.stride)))
    cloud = o3d.geometry.PointCloud()
    pose = np.eye(4)
    rows, timings, feature_timings = [], [], []
    rejected, consecutive, sampled = 0, 0, 0
    recoveries, weak_references_skipped, backup_uses = 0, 0, 0
    stop_reason = "end_of_video"
    started = time.perf_counter()
    with (output / "tracking.jsonl").open("w") as log:
        index = args.start_frame
        frame = first
        while True:
            if (index - args.start_frame) % args.stride == 0:
                sampled += 1
                image = cv2.remap(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), *maps, cv2.INTER_LINEAR)
                feature_started = time.perf_counter()
                keypoints, descriptors = extractor.extract(image)
                feature_timings.append(time.perf_counter() - feature_started)
                candidate_pose, stats = tracker.locate(keypoints, descriptors, index, recovering=consecutive > 0)
                stats["features"] = len(keypoints)
                if candidate_pose is None:
                    rejected += 1
                    consecutive += 1
                    print(f"frame={index}: rechazado {stats}", flush=True)
                else:
                    stats["recovered_after_failures"] = consecutive
                    consecutive = 0
                    pose = candidate_pose
                    recoveries += int(stats.get("recovered", False))
                    backup_uses += int(stats.get("backup_used", False))
                    before = time.perf_counter()
                    depth = depth_model.predict(image)
                    timings.append(time.perf_counter() - before)
                    depth = cv2.medianBlur(depth.astype(np.float32), 5)
                    cloud += make_cloud(image, depth, K, pose, args)
                    if len(rows) % 10 == 0 or len(cloud.points) > args.max_points:
                        cloud = cloud.voxel_down_sample(args.voxel)
                    stats["backup_reference_eligible"] = tracker.accept(index, keypoints, descriptors, depth, pose, stats)
                    stats["reference_updated"] = True
                    weak_references_skipped += int(not stats["backup_reference_eligible"])
                    record = dict(frame=index, timestamp_s=index / fps, x=float(pose[0, 3]), y=float(pose[1, 3]), z=float(pose[2, 3]))
                    record.update({f"T{i}{j}": float(pose[i, j]) for i in range(4) for j in range(4)})
                    rows.append(record)
                    print(f"frame={index}: aceptado, inliers={stats.get('inliers', 0)}, profundidad={timings[-1]:.2f}s, puntos={len(cloud.points)}", flush=True)
                log.write(json.dumps({"frame": index, **stats}) + "\n")
                log.flush()
                if consecutive >= failure_limit:
                    stop_reason = "tracking_lost"
                    break
                if len(cloud.points) > args.max_points:
                    stop_reason = "point_budget"
                    break
                if args.max_frames and sampled >= args.max_frames:
                    stop_reason = "frame_limit"
                    break
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
    cap.release()
    cloud = cloud.voxel_down_sample(args.voxel)
    if not len(cloud.points):
        raise RuntimeError("No quedaron puntos válidos: revisa la profundidad y sus límites")
    summary = {"video": str(args.video.resolve()), "model": depth_model.model_id,
               "model_revision": depth_model.revision, "checkpoint_sha256": depth_model.sha256,
               "device": depth_model.device, "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "intrinsics_resized": K.tolist(), "resized_size": size,
               "sampled_frames": sampled, "rejected_frames": rejected, "stop_reason": stop_reason,
               "recoveries": recoveries, "backup_uses": backup_uses,
               "weak_backup_references_skipped": weak_references_skipped,
               "tracking_policy": "original last accepted frame; strong backup references only after primary failure",
               "failure_limit": failure_limit, "reference_bank_size": len(tracker.bank),
               "processing_seconds_excluding_load_and_export": time.perf_counter() - started,
               "depth_seconds_per_accepted_frame": timings,
               "feature_backend": args.features, "feature_device": extractor.device,
               "feature_weights_sha256": extractor.weights_sha256,
               "feature_seconds_per_sampled_frame": feature_timings,
               "coordinate_system": "first camera: X right, Y down, Z forward; poses map camera to world",
               "limitations": ["Learned approximate metric scale", "No loop closure or global optimization", "XZ projection is not floor aligned or an occupancy grid", "No ground truth used"]}
    save_outputs(output, cloud, rows, summary)
    print(json.dumps({key: summary[key] for key in ("accepted_frames", "rejected_frames", "points_saved", "stop_reason", "estimated_path_length_m")}), flush=True)
    print(f"Resultados: {output}", flush=True)
    return 2 if stop_reason == "tracking_lost" else 0


if __name__ == "__main__":
    raise SystemExit(main())
