from streamer.kitti.streamer import KittiStreamer
from streamer.base import DatasetStreamer
from streamer.base import DatasetStreamerAdapter
from typing import Optional, Tuple, Any
import numpy as np
import cv2
import time
import os


class ShowImagesAdapter(DatasetStreamerAdapter):
    """Shows stereo images automatically like a movie."""

    def __init__(self, frequency: float):
        self.frequency = frequency

        if frequency <= 0:
            raise ValueError("Frequency must be positive")

        self.period = 1.0 / frequency

    def process(
        self,
        input: Optional[
            Tuple[Optional[np.ndarray[Any, Any]], Optional[np.ndarray[Any, Any]]]
        ],
    ) -> None:
        start_time: float = time.time()
        if input is None:
            return
        left, right = input
        if left is None or right is None:
            return
        cv2.imshow("Left", left)
        cv2.imshow("Right", right)
        elapsed_time = time.time() - start_time
        sleep_time = max(0.0, self.period - elapsed_time)
        cv2.waitKey(int(sleep_time * 1000))


if __name__ == "__main__":

    data_root: str = "data/2011_09_26_drive_0001_sync"
    print("Entering ", data_root)

    adapter = ShowImagesAdapter(10)
    streamer: DatasetStreamer = KittiStreamer(data_root, adapter)
    print("Starting KITTI movie playback...")
    print("Pres any key to start...")
    streamer.run()
    cv2.destroyAllWindows()
