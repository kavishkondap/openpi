import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Creates a random input example for the Bimanual YAM policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class BimanualYAMInputs(transforms.DataTransformFn):

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI05

        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        state = np.asarray(data["state"])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        exo_image = _parse_image(data["exo_image"])
        wrist_right_image = _parse_image(data["wrist_right_image"])
        wrist_left_image = _parse_image(data["wrist_left_image"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": exo_image,
                "left_wrist_0_rgb": wrist_left_image,
                "right_wrist_0_rgb": wrist_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension.
        # The robot produces 7D actions (6 DoF + 1 gripper), and we pad these.
        inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        # inputs["actions"] = actions
        # inputs["actions"] = np.asarray(data["action"])

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # else:
        #     print(f"DEBUG: COULD NOT FIND TASK PROMPT")

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualYAMOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
