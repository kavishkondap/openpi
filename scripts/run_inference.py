import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_ur5")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
checkpoint_dir = "/home/kavishk/openpi/checkpoints/pi05_ur5/run4/29999"

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
        "base_rgb": np.random.randint(256, size=(1080, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "joints": np.random.rand(7),
        "prompt": "do something",
    }

action_chunk = policy.infer(example)
