"""
Convert TSH (bimanual Franka) dataset to LeRobot format.

Data layout per trajectory:
  trajectory_N/
    joints.h5          -- gello_position (N, 16) and yam_position (N, 16)
    exo.mp4            -- top/exo camera (512x512)
    wrist_left.mp4     -- left wrist camera (512x512)
    wrist_right.mp4    -- right wrist camera (512x512)

State (16D):  yam_position  -- actual robot joint positions
              [left_joint_0..6, left_gripper, right_joint_0..6, right_gripper]
Actions (16D): gello_position -- teleoperator commanded positions (same layout as state)

Key optimizations (same as xdof_to_lerobot.py):
  1. Source MP4s are re-encoded to MJPEG with black padding to TARGET_HxTARGET_W.
  2. Parquet files are written directly via pyarrow, bypassing the O(n^2)
     concatenate_datasets() growth in _save_episode_table().
  3. Loading/encoding episode N+1 is overlapped with metadata writes for
     episode N via a background loader thread.

Usage:
  uv run tsh_to_lerobot.py \
      --data_dir /path/to/reformatted_remove_homing_fix_freq \
      --output_dir my_org/my_dataset
"""

import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Target frame dimensions -- source cameras are already 512x512, keep them square.
TARGET_H = 512
TARGET_W = 512

# MJPEG quality: 1 (best) to 31 (worst). 2-5 is visually lossless.
MJPEG_QUALITY = 3

# Language prompt stored in the dataset.
TASK_PROMPT = "Pick up the yellow tape, hand it over, and place it on the gray tape"


# ---------------------------------------------------------------------------
# Video install: pad + reencode to MJPEG
# ---------------------------------------------------------------------------

def install_video(src: Path, dst: Path, fps: int) -> None:
    """
    Re-encode src into dst as MJPEG, padding to TARGET_HxTARGET_W with black.

    MJPEG stores each frame as an independent JPEG, so training dataloaders can
    seek to any frame in O(1) without scanning for the nearest keyframe.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(src),
        "-vf", f"pad={TARGET_W}:{TARGET_H}:0:0:black,setpts=N/(FRAME_RATE*TB)",
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

def load_and_install_episode(
    episode_path: Path, episode_index: int, dataset: LeRobotDataset
) -> dict:
    """
    1. Load state and action arrays from joints.h5.
    2. Re-encode the three source MP4s as MJPEG into the LeRobot video tree.

    Returns a plain dict of scalar data -- no dataset state is touched.
    """
    # Load joint data from HDF5
    with h5py.File(episode_path / "joints.h5", "r") as f:
        # yam_position: actual robot joint positions -> state (no shift)
        # Shape: (N, 16) = [left_joint_0..6, left_gripper, right_joint_0..6, right_gripper]
        yam_pos = f["yam_position"][:].astype(np.float32)

        # gello_position: teleoperator commanded positions -> actions (shifted +1)
        # action[t] = gello_position[t+1], so we drop the last state frame.
        gello_pos = f["gello_position"][:].astype(np.float32)

    # Align: state[t] paired with gello[t+1], giving N-1 timesteps.
    states = yam_pos[:-1]
    actions = gello_pos[1:]

    n_steps = actions.shape[0]

    # Re-encode videos as MJPEG into the LeRobot directory tree
    chunk_str = "chunk-{:03d}".format(episode_index // dataset.meta.chunks_size)
    ep_str = "episode_{:06d}".format(episode_index)
    video_root = dataset.root / "videos" / chunk_str

    fps = dataset.fps
    install_video(
        episode_path / "exo.mp4",
        video_root / "exo_image" / "{}.mp4".format(ep_str),
        fps,
    )
    install_video(
        episode_path / "wrist_left.mp4",
        video_root / "wrist_left_image" / "{}.mp4".format(ep_str),
        fps,
    )
    install_video(
        episode_path / "wrist_right.mp4",
        video_root / "wrist_right_image" / "{}.mp4".format(ep_str),
        fps,
    )

    # Trim arrays to the actual encoded video frame count to keep parquet and
    # video in sync (VFR->CFR conversion may add or drop frames).
    video_frame_count = get_frame_count(video_root / "exo_image" / "{}.mp4".format(ep_str))
    if video_frame_count != n_steps:
        n_steps = min(video_frame_count, n_steps)
        print("WARNING: {}: video has {} frames, h5 has {} steps, using {}.".format(
            episode_path.name, video_frame_count, actions.shape[0], n_steps
        ))
        actions = actions[:n_steps]
        states = states[:n_steps]

    return {
        "actions": actions,
        "state": states,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Per-episode metadata + parquet save  (main thread only)
# ---------------------------------------------------------------------------

def save_episode_metadata(
    dataset: LeRobotDataset, ep_data: dict, episode_index: int
) -> None:
    """
    Write the parquet file and update dataset metadata.
    Video files must already exist before calling this (update_video_info reads them).
    Bypasses LeRobot's concatenate_datasets to avoid O(n^2) growth.
    """
    n_steps = ep_data["n_steps"]
    actions = ep_data["actions"]
    states = ep_data["state"]
    fps = dataset.fps

    frame_indices = np.arange(n_steps, dtype=np.int64)
    episode_indices = np.full(n_steps, episode_index, dtype=np.int64)
    # Synthetic timestamps -- hardware timestamps from the h5 may have duplicate
    # values (clock didn't tick between frames) which causes check_timestamps_sync
    # to fail on load, so we derive timestamps from frame index.
    timestamps = (frame_indices / fps).astype(np.float32)
    global_index = np.arange(
        dataset.meta.total_frames, dataset.meta.total_frames + n_steps, dtype=np.int64
    )

    # Register task if new
    task = TASK_PROMPT
    task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        dataset.meta.add_task(task)
        task_index = dataset.meta.get_task_index(task)
    task_indices = np.full(n_steps, task_index, dtype=np.int64)

    # Write parquet directly (avoids O(n^2) concatenate_datasets)
    parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "index":          pa.array(global_index,       type=pa.int64()),
        "episode_index":  pa.array(episode_indices,    type=pa.int64()),
        "frame_index":    pa.array(frame_indices,      type=pa.int64()),
        "timestamp":      pa.array(timestamps,         type=pa.float32()),
        "task_index":     pa.array(task_indices,       type=pa.int64()),
        "state":          pa.array(states.tolist(),    type=pa.list_(pa.float32(), 16)),
        "actions":        pa.array(actions.tolist(),   type=pa.list_(pa.float32(), 16)),
        "prompt":         pa.array([task] * n_steps,   type=pa.string()),
    })
    pq.write_table(table, parquet_path)

    # Compute stats and update metadata
    ep_buffer = {
        "index":          global_index,
        "episode_index":  episode_indices,
        "frame_index":    frame_indices,
        "timestamp":      timestamps,
        "task_index":     task_indices,
        "state":          states,
        "actions":        actions,
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

def main(data_dir: str, output_dir: str, fps: int = 30):
    """
    Args:
        data_dir:   Path to the root of the TSH dataset (contains trajectory_N subdirs).
        output_dir: LeRobot repo_id, e.g. "my_org/tsh_dataset".
        fps:        Frame rate of the source videos. Default 30.
    """
    output_path = HF_LEROBOT_HOME / output_dir
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="franka",
        fps=fps,
        features={
            "exo_image": {
                "dtype": "video",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_left_image": {
                "dtype": "video",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_right_image": {
                "dtype": "video",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            # 16D: [left_joint_0..6, left_gripper, right_joint_0..6, right_gripper]
            "state":   {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (16,), "names": ["actions"]},
            "prompt":  {"dtype": "string",  "shape": (1,),  "names": ["prompt"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )

    # Collect and sort all trajectory directories
    all_episode_paths: list[Path] = sorted(
        Path(data_dir).glob("trajectory_*/"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    total = len(all_episode_paths)
    if total == 0:
        raise ValueError("No trajectory_* directories found in {}".format(data_dir))

    # Pipeline:
    #   loader thread : reencode 3 videos as MJPEG + load h5 arrays for episode N+1
    #   main thread   : write parquet + update metadata for episode N
    #
    # Timeline:
    #   loader: |---load+encode(0)---|---load+encode(1)---|---load+encode(2)---|
    #   main  :                      |meta(0)|             |meta(1)|

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="loader") as loader:
        load_future: Future = loader.submit(
            load_and_install_episode, all_episode_paths[0], 0, dataset
        )

        for ep_idx, episode_path in tqdm.tqdm(
            enumerate(all_episode_paths), total=total, desc="Converting"
        ):
            ep_data = load_future.result()

            if ep_idx + 1 < total:
                load_future = loader.submit(
                    load_and_install_episode,
                    all_episode_paths[ep_idx + 1],
                    ep_idx + 1,
                    dataset,
                )

            save_episode_metadata(dataset, ep_data, ep_idx)


if __name__ == "__main__":
    tyro.cli(main)
