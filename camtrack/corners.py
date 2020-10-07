#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class _FrameCornersBuilder:
    def __init__(self, img, number_of_levels):
        self.number_of_levels = number_of_levels
        self.img_pyr = self.get_pyramid(img)
        self.ids_list = [np.zeros((0,), dtype=np.int32) for _ in range(number_of_levels)]
        self.points_list = [np.zeros((0, 2), dtype=np.float32) for _ in range(number_of_levels)]
        self.errs_list = [np.zeros((0, 1), dtype=np.float32) for _ in range(number_of_levels)]

    def get_pyramid(self, img):
        pyramid = [np.around(img * 255).astype(dtype=np.uint8)]
        for i in range(1, self.number_of_levels):
            pyramid.append(cv2.pyrDown(pyramid[i - 1]))
        return pyramid

    def add_new_points(self, points, dist, level, first_id):
        added_points = np.concatenate(self.points_list)
        points = points[[np.all(np.linalg.norm(added_points - point, axis=1) > dist) for point in points]]
        self.points_list[level] = np.concatenate((self.points_list[level], points))
        self.ids_list[level] = np.concatenate((self.ids_list[level], np.arange(first_id, first_id + points.shape[0])))
        return points.shape[0]

    def get_corners(self, corner_size):
        sizes_list = [np.full(points.shape[0], corner_size * 2 ** i) for i, points in enumerate(self.points_list)]
        return FrameCorners(np.concatenate(self.ids_list), np.concatenate(self.points_list), np.concatenate(sizes_list))

    def calculate_optical_flow(self, prev_builder, window_size):
        levels = np.concatenate([np.full(ids.shape[0], i) for i, ids in enumerate(prev_builder.ids_list)])
        ids = np.concatenate(prev_builder.ids_list)
        prev_points = np.concatenate(prev_builder.points_list)
        points, st, errs = cv2.calcOpticalFlowPyrLK(prev_builder.img_pyr[0], self.img_pyr[0],
                                                    prev_points, None,
                                                    winSize=(window_size, window_size),
                                                    maxLevel=2,
                                                    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                    criteria=(
                                                        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))
        filter_by_status = (st == 1).reshape((-1,))
        for i in range(self.number_of_levels):
            filter_by_lvl_and_status = np.logical_and(filter_by_status, levels == i)
            self.ids_list[i] = ids[filter_by_lvl_and_status]
            self.points_list[i] = points[filter_by_lvl_and_status]
            self.errs_list[i] = errs[filter_by_lvl_and_status]

    def filter_corners_in_flow(self, corner_size, quality_level):
        levels_list = [np.full(ids.shape[0], i) for i, ids in enumerate(self.ids_list)]
        min_eigenvals_thr_list = [np.max(errs) * quality_level for errs in self.errs_list]
        errs = np.concatenate(self.errs_list)
        sort_by_err = np.argsort(errs.reshape((-1,)))[::-1]

        errs = errs[sort_by_err]
        ids = np.concatenate(self.ids_list)[sort_by_err]
        points = np.concatenate(self.points_list)[sort_by_err]
        levels = np.concatenate(levels_list)[sort_by_err]

        added_points = np.zeros((0, 2))
        is_added = np.full(points.shape[0], False)
        for i, point in enumerate(points):
            if errs[i] > min_eigenvals_thr_list[levels[i]] * quality_level:
                if np.all(np.linalg.norm(added_points - point, axis=1) > corner_size * 2 ** levels[i] / 4):
                    added_points = np.vstack((added_points, point))
                    is_added[i] = True

        for i in range(self.number_of_levels):
            add_i_lvl = np.logical_and(is_added, levels == i)
            self.ids_list[i] = ids[add_i_lvl]
            self.points_list[i] = points[add_i_lvl]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corners_counter = 0

    # constants
    window_size = max(frame_sequence[0].shape[0] // 16, 3)
    corner_size = max(frame_sequence[0].shape[0] // 80, 3)
    min_dist = corner_size / 2
    number_of_levels = 2
    max_corners = 1500
    quality_level = 0.01

    builder0 = _FrameCornersBuilder(frame_sequence[0], number_of_levels)

    for i in range(number_of_levels):
        points = cv2.goodFeaturesToTrack(builder0.img_pyr[i], max_corners, quality_level * 0.5, min_dist,
                                         useHarrisDetector=False).reshape((-1, 2)) * 2 ** i
        corners_counter += builder0.add_new_points(points, min_dist, i, corners_counter)

    builder.set_corners_at_frame(0, builder0.get_corners(corner_size))

    for frame, img1 in enumerate(frame_sequence[1:], 1):
        builder1 = _FrameCornersBuilder(img1, number_of_levels)
        builder1.calculate_optical_flow(builder0, window_size)
        builder1.filter_corners_in_flow(corner_size, quality_level)

        for i in range(number_of_levels):
            # add new corners
            new_points = cv2.goodFeaturesToTrack(builder1.img_pyr[i], max_corners, quality_level, min_dist,
                                                 useHarrisDetector=False).reshape((-1, 2)) * 2 ** i
            corners_counter += builder1.add_new_points(new_points, min_dist * 2 ** i, i, corners_counter)

        builder0 = builder1
        builder.set_corners_at_frame(frame, builder1.get_corners(corner_size))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
