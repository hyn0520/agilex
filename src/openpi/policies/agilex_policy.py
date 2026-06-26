import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    arm: str = "both"

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/images/front"])
        left_image = _parse_image(data["observation/images/left"])
        right_image = _parse_image(data["observation/images/right"])
        state = np.asarray(data["observation/state"])
        if self.arm == "left":
            state = state[..., :7]
        elif self.arm == "right":
            state = state[..., 7:14]
        elif self.arm != "both":
            raise ValueError(f"Unsupported arm: {self.arm}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": left_image if self.arm == "both" else right_image,
                "right_wrist_0_rgb": right_image if self.arm == "both" else np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.arm == "both" else np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.arm == "left":
                actions = actions[..., :7]
            elif self.arm == "right":
                actions = actions[..., 7:14]
            inputs["actions"] = actions
            if "action_is_pad" in data:
                inputs["action_is_pad"] = np.asarray(data["action_is_pad"], dtype=bool)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
