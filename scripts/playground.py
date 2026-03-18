# # """
# # Check that video frame counts match numpy array lengths for all episodes,
# # and optionally cross-check against the re-encoded videos in a LeRobot output dir.

# # Usage:
# #   # Check source data only
# #   python check_frame_counts.py --data_dirs /path/to/data1 /path/to/data2

# #   # Also check re-encoded output videos
# #   python check_frame_counts.py --data_dirs /path/to/data1 --output_dir /path/to/lerobot/dataset
# # """

# # import subprocess
# # from pathlib import Path

# # import numpy as np
# # import tqdm
# # import tyro


# # CAMERA_MAP = [
# #     # (source filename,              lerobot video key,   display label)
# #     ("top_camera-images-rgb.mp4",   "exo_image",          "exo"),
# #     ("left_camera-images-rgb.mp4",  "wrist_left_image",   "wrist_left"),
# #     ("right_camera-images-rgb.mp4", "wrist_right_image",  "wrist_right"),
# # ]

# # CHUNKS_SIZE = 1000  # must match dataset creation (LeRobot default)


# # def get_frame_count(path: Path) -> int:
# #     result = subprocess.run(
# #         [
# #             "ffprobe", "-v", "error",
# #             "-select_streams", "v:0",
# #             "-count_packets",
# #             "-show_entries", "stream=nb_read_packets",
# #             "-of", "csv=p=0",
# #             str(path),
# #         ],
# #         capture_output=True, text=True,
# #     )
# #     if result.returncode != 0:
# #         raise RuntimeError("ffprobe failed for {}:\n{}".format(path, result.stderr))
# #     return int(result.stdout.strip())


# # def output_video_path(output_dir: Path, episode_index: int, video_key: str) -> Path:
# #     chunk = episode_index // CHUNKS_SIZE
# #     return (
# #         output_dir
# #         / "videos"
# #         / "chunk-{:03d}".format(chunk)
# #         / video_key
# #         / "episode_{:06d}.mp4".format(episode_index)
# #     )


# # def main(data_dirs: list[str], output_dir: str | None = None):
# #     # Collect all episode paths across all data dirs in sorted order,
# #     # matching the order the conversion script processes them (and thus
# #     # the episode_index assigned to each).
# #     all_episode_paths = []
# #     for raw_dir in data_dirs:
# #         eps = sorted(Path(raw_dir).glob("episode_*/"))
# #         print("Found {} episodes in {}".format(len(eps), raw_dir))
# #         all_episode_paths.extend(eps)

# #     out_path = Path(output_dir) if output_dir else None
# #     if out_path:
# #         print("Also checking output videos in {}".format(out_path))

# #     source_mismatches = []
# #     output_mismatches = []

# #     for ep_idx, episode_path in enumerate(tqdm.tqdm(all_episode_paths)):
# #         n_steps = np.load(episode_path / "action-left-pos.npy").shape[0]

# #         for src_name, video_key, label in CAMERA_MAP:
# #             # -- source video vs numpy --
# #             src_video = episode_path / src_name
# #             if not src_video.exists():
# #                 source_mismatches.append({
# #                     "episode": episode_path.name,
# #                     "camera": label,
# #                     "issue": "missing source file",
# #                 })
# #                 continue

# #             src_frames = get_frame_count(src_video)
# #             if src_frames != n_steps:
# #                 source_mismatches.append({
# #                     "episode": episode_path.name,
# #                     "camera": label,
# #                     "numpy_steps": n_steps,
# #                     "video_frames": src_frames,
# #                     "diff": src_frames - n_steps,
# #                 })

# #             # -- output video vs numpy --
# #             if out_path is not None:
# #                 out_video = output_video_path(out_path, ep_idx, video_key)
# #                 if not out_video.exists():
# #                     output_mismatches.append({
# #                         "episode_index": ep_idx,
# #                         "episode": episode_path.name,
# #                         "camera": label,
# #                         "issue": "missing output file",
# #                     })
# #                     continue

# #                 out_frames = get_frame_count(out_video)
# #                 if out_frames != n_steps:
# #                     output_mismatches.append({
# #                         "episode_index": ep_idx,
# #                         "episode": episode_path.name,
# #                         "camera": label,
# #                         "numpy_steps": n_steps,
# #                         "video_frames": out_frames,
# #                         "diff": out_frames - n_steps,
# #                     })

# #     print("\n--- Source data ---")
# #     print("Episodes checked: {}".format(len(all_episode_paths)))
# #     if not source_mismatches:
# #         print("All source frame counts match!")
# #     else:
# #         print("{} mismatches:".format(len(source_mismatches)))
# #         for m in source_mismatches:
# #             print("  ", m)

# #     if out_path is not None:
# #         print("\n--- Output videos ---")
# #         if not output_mismatches:
# #             print("All output frame counts match!")
# #         else:
# #             print("{} mismatches:".format(len(output_mismatches)))
# #             for m in output_mismatches:
# #                 print("  ", m)


# # if __name__ == "__main__":
# #     tyro.cli(main)
# import subprocess
# result = subprocess.run(
#     [
#         "ffprobe", "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "packet=pts_time",
#         "-of", "csv=p=0",
#         "/home/kavishk/.cache/huggingface/lerobot/tshirt_combined3/videos/chunk-000/exo_image/episode_000000.mp4",
#     ],
#     capture_output=True, text=True,
# )
# pts = [float(x) for x in result.stdout.strip().split("\n")]

# # Check which ones don't land on exact multiples of 1/30
# fps = 30
# for i, t in enumerate(pts):
#     expected = i / fps
#     diff = abs(t - expected)
#     if diff > 0.0001:
#         print(f"frame {i}: pts={t:.6f}, expected={expected:.6f}, diff={diff:.6f}")
"""
Check the fps of all source videos across all episodes.

Usage:
  python check_fps.py --data_dirs /path/to/data1 /path/to/data2
"""

import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import tqdm
import tyro

CAMERAS = [
    ("top_camera-images-rgb.mp4",   "exo"),
    ("left_camera-images-rgb.mp4",  "wrist_left"),
    ("right_camera-images-rgb.mp4", "wrist_right"),
]


def get_fps(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for {}:\n{}".format(path, result.stderr))
    # r_frame_rate is returned as a fraction e.g. "9023/300"
    frac = Fraction(result.stdout.strip())
    return float(frac), str(frac)


def main(data_dirs: list[str]):
    # fps_counts[camera_label][fraction_string] = count
    fps_counts: dict[str, Counter] = {label: Counter() for _, label in CAMERAS}

    all_episode_paths = []
    for raw_dir in data_dirs:
        eps = sorted(Path(raw_dir).glob("episode_*/"))
        print("Found {} episodes in {}".format(len(eps), raw_dir))
        all_episode_paths.extend(eps)

    for episode_path in tqdm.tqdm(all_episode_paths):
        for video_name, label in CAMERAS:
            video_path = episode_path / video_name
            if not video_path.exists():
                fps_counts[label]["MISSING"] += 1
                continue
            _, frac_str = get_fps(video_path)
            fps_counts[label][frac_str] += 1

    print("\n--- FPS summary ---")
    for _, label in CAMERAS:
        print("\n{}:".format(label))
        for frac_str, count in fps_counts[label].most_common():
            if frac_str == "MISSING":
                print("  MISSING: {} episodes".format(count))
            else:
                fps_val = float(Fraction(frac_str))
                print("  {} = {:.6f} fps  ({} episodes)".format(frac_str, fps_val, count))


if __name__ == "__main__":
    tyro.cli(main)