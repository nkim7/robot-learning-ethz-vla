import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any

from ..opencv.camera_opencv import OpenCVCamera
from ..opencv.configuration_opencv import OpenCVCameraConfig, ColorMode, Cv2Backends, Cv2Rotation
from .configuration_opencv_crop import OpenCVCropCameraConfig


class OpenCVCropCamera(OpenCVCamera):
    """OpenCV camera that captures at a source resolution, crops a square ROI, then resizes."""

    def __init__(self, config: OpenCVCropCameraConfig):
        # Build an OpenCVCameraConfig using source resolution for capture
        base_cfg = OpenCVCameraConfig(
            index_or_path=config.index_or_path,
            color_mode=config.color_mode,
            rotation=config.rotation,
            warmup_s=config.warmup_s,
            fourcc=config.fourcc,
            backend=config.backend,
        )
        # Inject the source resolution and fps from the crop config
        base_cfg.fps = config.fps
        base_cfg.width = config.source_width
        base_cfg.height = config.source_height

        super().__init__(base_cfg)

        self._crop_config = config
        # Override the output width/height to the final resized dimensions
        self.width = config.width
        self.height = config.height

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Apply parent postprocessing (color + rotation), then crop and resize."""
        # Parent handles color conversion and rotation using capture dimensions
        processed = super()._postprocess_image(image)

        cfg = self._crop_config
        # Crop square ROI
        cropped = processed[cfg.roi_y: cfg.roi_y + cfg.roi_size,
                             cfg.roi_x: cfg.roi_x + cfg.roi_size]

        # Resize to target output resolution
        resized = cv2.resize(cropped, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)

        return resized
