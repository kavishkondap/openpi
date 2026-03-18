"""
Optimized script for converting a dataset to LeRobot format.

Key optimizations:
  1. Source MP4s are re-encoded to MJPEG with black padding to TARGET_HxTARGET_W.
     MJPEG stores each frame as an independent JPEG, making random-access seeks
     during training fast without keyframe searches. Padding and encoding are
     both handled by a single ffmpeg call per video -- no Python frame loop.
  2. Parquet files are written directly via pyarrow, bypassing the O(n^2)
     concatenate_datasets() growth in _save_episode_table().
  3. Loading/encoding episode N+1 is overlapped with metadata writes for
     episode N via a background loader thread.

Ordering constraint:
  install_videos(N) must complete before meta.save_episode(N) because
  update_video_info() opens the mp4 to read codec info.

Pipeline timeline:
  loader: |---load+encode(0)---|---load+encode(1)---|---load+encode(2)---|
  main  :                      |meta(0)|             |meta(1)|            |meta(2)|

Usage:
  uv run convert_libero_data_to_lerobot.py \
      --data_dirs /path/to/data1 /path/to/data2 \
      --output_dir my_org/my_dataset
"""

# import json  # unused after annotation removal
import shutil
import subprocess
# import time  # unused after timing removal
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro

# Target frame dimensions
TARGET_H = 480
TARGET_W = 848

# MJPEG quality: 1 (best) to 31 (worst). 2-5 is visually lossless.
MJPEG_QUALITY = 3


# ---------------------------------------------------------------------------
# Video install: pad + reencode to MJPEG
# ---------------------------------------------------------------------------

def install_video(src: Path, dst: Path, fps: int) -> None:
    """
    Re-encode src into dst as MJPEG, padding to TARGET_HxTARGET_W with black.

    MJPEG stores each frame as an independent JPEG, so training dataloaders can
    seek to any frame in O(1) without scanning for the nearest keyframe.
    The pad filter handles both same-size and smaller-than-target inputs.

    -vsync cfr + -video_track_timescale forces container timestamps to be
    exact multiples of 1/fps, matching the parquet timestamps and keeping
    LeRobot's decode_video_frames tolerance check happy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        # 1. Force FFmpeg to ignore source timestamps and treat it as a constant stream
        "-r", str(fps), 
        "-i", str(src),
        # 2. Re-generate Presentation Timestamps (PTS) starting from 0 based on frame index
        "-vf", f"pad={TARGET_W}:{TARGET_H}:0:0:black,setpts=N/(FRAME_RATE*TB)",
        # 3. 'vfr' or '0' can sometimes preserve jitter; 'cfr' forces Constant Frame Rate 
        # but combined with the input -r, it ensures 1:1 frame mapping.
        "-vsync", "cfr",
        "-vcodec", "mjpeg",
        "-q:v", str(MJPEG_QUALITY),
        "-pix_fmt", "yuvj420p",
        "-video_track_timescale", str(fps),
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg mjpeg encode failed for {}:\n{}".format(src, result.stderr.decode())
        )


def get_frame_count(path: Path) -> int:
    """Return exact frame count of a video via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for {}:\n{}".format(path, result.stderr))
    return int(result.stdout.strip())


# ---------------------------------------------------------------------------
# Episode loading + video install  (safe to run in a background thread)
# ---------------------------------------------------------------------------

def load_and_install_episode(episode_path: Path, episode_index: int, dataset: LeRobotDataset) -> dict:
    """
    1. Load all numpy arrays.
    2. Re-encode the three source MP4s as MJPEG into the LeRobot video tree.
    3. Read annotations.

    Returns a plain dict of scalar data -- no dataset state is touched.
    """
    # numpy arrays
    action_left = np.load(episode_path / "action-left-pos.npy")
    action_right = np.load(episode_path / "action-right-pos.npy")

    left_gripper_pos = np.load(episode_path / "left-gripper_pos.npy")
    left_joint_pos = np.load(episode_path / "left-joint_pos.npy")
    right_gripper_pos = np.load(episode_path / "right-gripper_pos.npy")
    right_joint_pos = np.load(episode_path / "right-joint_pos.npy")

    left_eef_pose = np.load(episode_path / "left-joint_pose.npy").reshape((-1, 16))
    right_eef_pose = np.load(episode_path / "right-joint_pose.npy").reshape((-1, 16))

    actions = np.concatenate([action_left, action_right], axis=1).astype(np.float32)
    states = np.concatenate(
        [left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos], axis=1
    ).astype(np.float32)
    eef_poses = np.concatenate([left_eef_pose, right_eef_pose], axis=1).astype(np.float32)
    n_steps = actions.shape[0]
    # timestamps = np.load(episode_path / "timestamp.npy").astype(np.float32)
    # Hardware timestamps have duplicate values so we use synthetic frame_index/fps instead.

    # re-encode videos as MJPEG into the LeRobot directory tree
    chunk_str = "chunk-{:03d}".format(episode_index // dataset.meta.chunks_size)
    ep_str = "episode_{:06d}".format(episode_index)
    video_root = dataset.root / "videos" / chunk_str

    fps = dataset.fps
    install_video(
        episode_path / "top_camera-images-rgb.mp4",
        video_root / "exo_image" / "{}.mp4".format(ep_str),
        fps,
    )
    install_video(
        episode_path / "left_camera-images-rgb.mp4",
        video_root / "wrist_left_image" / "{}.mp4".format(ep_str),
        fps,
    )
    install_video(
        episode_path / "right_camera-images-rgb.mp4",
        video_root / "wrist_right_image" / "{}.mp4".format(ep_str),
        fps,
    )

    # The encoded video may have slightly fewer frames than the numpy arrays
    # (e.g. due to the source video having a different frame count). Trim all
    # arrays to the actual video frame count so parquet and video stay in sync.
    video_frame_count = get_frame_count(video_root / "exo_image" / "{}.mp4".format(ep_str))
    if video_frame_count != n_steps:
        # Trim to the shorter of the two -- the fps filter may add frames (VFR->CFR
        # upsampling) or drop frames, so we always take the min to keep arrays in sync.
        n_steps = min(video_frame_count, n_steps)
        print("WARNING: {}: video has {} frames, numpy has {} steps, using {}.".format(
            episode_path.name, video_frame_count, actions.shape[0], n_steps
        ))
        actions = actions[:n_steps]
        states = states[:n_steps]
        eef_poses = eef_poses[:n_steps]

    # annotations
    # with open(episode_path / "top_camera-images-rgb_annotation.json") as f:
    #     annotations = json.load(f)["annotations"]

    return {
        "actions": actions,
        "state": states,
        "eef_poses": eef_poses,
        "n_steps": n_steps,
        # "timestamps": timestamps,  # unused: synthetic timestamps used instead
        # "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Per-episode metadata + parquet save  (main thread only)
# ---------------------------------------------------------------------------

def save_episode_metadata(dataset: LeRobotDataset, ep_data: dict, episode_index: int) -> None:
    """
    Write the parquet file and update dataset metadata.
    Video files must already exist before calling this (update_video_info reads them).
    Bypasses LeRobot's concatenate_datasets to avoid O(n^2) growth.
    """
    n_steps = ep_data["n_steps"]
    actions = ep_data["actions"]
    states = ep_data["state"]
    eef_poses = ep_data["eef_poses"]
    # annotations = ep_data["annotations"]
    fps = dataset.fps

    frame_indices = np.arange(n_steps, dtype=np.int64)
    episode_indices = np.full(n_steps, episode_index, dtype=np.int64)
    # Use synthetic frame_index/fps timestamps. Hardware timestamps from
    # timestamp.npy can have duplicate values (clock didn't tick between frames)
    # which causes check_timestamps_sync to fail on load.
    timestamps = (frame_indices / fps).astype(np.float32)
    global_index = np.arange(
        dataset.meta.total_frames, dataset.meta.total_frames + n_steps, dtype=np.int64
    )

    # Resolve annotation label per frame
    # curr_ann = 0
    # subtasks = []
    # for step in range(n_steps):
    #     if (curr_ann < len(annotations) - 1
    #             and not (annotations[curr_ann]["from_frame"] <= step < annotations[curr_ann]["to_frame"])):
    #         curr_ann += 1
    #     subtasks.append(annotations[curr_ann]["label"])

    # Register task if new
    task = "t-shirt folding"
    task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        dataset.meta.add_task(task)
        task_index = dataset.meta.get_task_index(task)
    task_indices = np.full(n_steps, task_index, dtype=np.int64)

    # Write parquet directly (avoids O(n^2) concatenate_datasets)
    parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = "Fold the laundry"
    table = pa.table({
        "index":         pa.array(global_index,                    type=pa.int64()),
        "episode_index": pa.array(episode_indices,                 type=pa.int64()),
        "frame_index":   pa.array(frame_indices,                   type=pa.int64()),
        "timestamp":     pa.array(timestamps,                      type=pa.float32()),
        "task_index":    pa.array(task_indices,                    type=pa.int64()),
        "state":        pa.array(states.tolist(),                 type=pa.list_(pa.float32(), 14)),
        "actions":       pa.array(actions.tolist(),                type=pa.list_(pa.float32(), 14)),
        "eef_poses":     pa.array(eef_poses.tolist(),              type=pa.list_(pa.float32(), 32)),
        "prompt":        pa.array([prompt] * n_steps,              type=pa.string()),
    })
    pq.write_table(table, parquet_path)

    # Compute stats and update metadata
    ep_buffer = {
        "index":         global_index,
        "episode_index": episode_indices,
        "frame_index":   frame_indices,
        "timestamp":     timestamps,
        "task_index":    task_indices,
        "state":        states,
        "actions":       actions,
        "eef_poses":     eef_poses,
    }
    ep_stats = compute_episode_stats(ep_buffer, dataset.features)
    dataset.meta.save_episode(
        episode_index=episode_index,
        episode_length=n_steps,
        episode_tasks=[task],
        episode_stats=ep_stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dirs: list[str], output_dir: str):
    output_path = HF_LEROBOT_HOME / output_dir
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="yam",
        fps=30,
        features={
            "exo_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_right_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_left_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "state":    {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "actions":   {"dtype": "float32", "shape": (14,), "names": ["actions"]},
            "eef_poses": {"dtype": "float32", "shape": (32,), "names": ["eef_poses"]},
            "prompt":   {"dtype": "string",  "shape": (1,),  "names": ["prompt"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )

    all_episode_paths: list[Path] = []
    for raw_dir in data_dirs:
        eps = sorted(Path(raw_dir).glob("episode_*/"))
        all_episode_paths.extend(eps)
    
    total = len(all_episode_paths)

    # Pipeline:
    #
    #   loader thread : reencode 3 videos as MJPEG + load numpy for episode N+1
    #   main thread   : write parquet + update metadata for episode N
    #
    # The loader must finish before save_episode_metadata because
    # meta.save_episode() -> update_video_info() opens the mp4 to read codec info.
    #
    # Timeline:
    #   loader: |---load+encode(0)---|---load+encode(1)---|---load+encode(2)---|
    #   main  :                      |meta(0)|             |meta(1)|

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="loader") as loader:
        load_future: Future = loader.submit(
            load_and_install_episode, all_episode_paths[0], 0, dataset
        )

        for ep_idx, episode_path in tqdm.tqdm(enumerate(all_episode_paths), total=total, desc="Converting"):
            # Wait for this episode's data + videos to be ready
            # t0 = time.time()
            ep_data = load_future.result()
            # print("  load+encode: {:.2f}s  ({} steps)".format(time.time() - t0, ep_data["n_steps"]))

            # Kick off loading + encoding the next episode immediately
            if ep_idx + 1 < total:
                load_future = loader.submit(
                    load_and_install_episode, all_episode_paths[ep_idx + 1], ep_idx + 1, dataset
                )

            # Write parquet + update metadata (videos already in place)
            # t0 = time.time()
            save_episode_metadata(dataset, ep_data, ep_idx)
            # print("  meta+pq:     {:.2f}s".format(time.time() - t0))


if __name__ == "__main__":
    tyro.cli(main)