import os
from glob import glob
from typing import Optional, Tuple, Any
import numpy as np
import cv2
import streamer.base as base


class KittiStreamer(base.DatasetStreamer):
    def __init__(
        self, data_root: str, dataset_streamer_adapter: base.DatasetStreamerAdapter
    ) -> None:
        """
        data_root: top folder of 2011_09_26 raw KITTI data
        """
        self.left_folder: str = os.path.join(data_root, "image_00/data")
        self.right_folder: str = os.path.join(data_root, "image_01/data")

        print("cwd: ", os.getcwd())
        print("left folder : ", self.left_folder)
        print("right folder : ", self.right_folder)

        self.left_images: list[str] = sorted(
            glob(os.path.join(self.left_folder, "*.png"))
        )
        self.right_images: list[str] = sorted(
            glob(os.path.join(self.right_folder, "*.png"))
        )

        print("Found ", len(self.left_images), " images")

        if len(self.left_images) != len(self.right_images):
            raise ValueError("Left and right image counts do not match!")

        self.index: int = 0
        self.total: int = len(self.left_images)
        print("Found ", self.total, " images")

        self.dataset_streamer_adapter: base.DatasetStreamerAdapter = (
            dataset_streamer_adapter
        )

    def reset(self) -> None:
        self.index = 0

    def has_next(self) -> bool:
        return self.index < self.total

    def next(
        self,
    ) -> Optional[Tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]]:
        if not self.has_next():
            return None

        img_left: np.ndarray | None = cv2.imread(
            self.left_images[self.index], cv2.IMREAD_GRAYSCALE
        )
        img_right: np.ndarray | None = cv2.imread(
            self.right_images[self.index], cv2.IMREAD_GRAYSCALE
        )

        self.index += 1

        return img_left, img_right

    def run(self) -> None:
        """Runs stream in given frequency in Hz."""

        while self.has_next():

            result = self.next()
            if result is None:
                print("Images are None!")
                continue

            left_img, right_img = result

            if left_img is None:
                print("Left image is None!")
                continue

            if right_img is None:
                print("Right image is None!")
                continue

            self.dataset_streamer_adapter.process((left_img, right_img))
