from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation


@CameraConfig.register_subclass("opencv_crop")
@dataclass
class OpenCVCropCameraConfig(CameraConfig):
    """OpenCV camera with square ROI crop + resize.

    Captures at source_width x source_height, crops roi_size x roi_size
    starting at (roi_x, roi_y), then resizes to width x height.

    Example::
        {"type": "opencv_crop", "index_or_path": 0,
         "source_width": 640, "source_height": 480,
         "roi_x": 215, "roi_y": 0, "roi_size": 425,
         "width": 256, "height": 256, "fps": 30}
    """

    index_or_path: int | Path = 0
    source_width: int = 640
    source_height: int = 480
    roi_x: int = 0
    roi_y: int = 0
    roi_size: int = 480
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.backend = Cv2Backends(self.backend)
