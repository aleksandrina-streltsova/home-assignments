#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import deque, defaultdict
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp

import frameseq
from _camtrack import *
from _corners import filter_frame_corners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


class Point:
    def __init__(self):
        self.frames = []
        self.poses2d = []
        self.pose3d = None


def add_frame(points_dict, corners, frame):
    for point, point_id in zip(corners.points, corners.ids.flatten()):
        points_dict[point_id].frames.append(frame)
        points_dict[point_id].poses2d.append(point)


def count_inliers_and_mean_error(points2d, proj_mats, point3d, max_reprojection_error):
    point3d = np.append(point3d, 1)
    projected_points2d = np.dot(proj_mats, point3d)
    projected_points2d /= projected_points2d[:, [2]]
    projected_points2d = projected_points2d[:, :2]
    errors = np.linalg.norm(points2d - projected_points2d, axis=1)
    errors_cnt = np.count_nonzero(errors < max_reprojection_error)
    if errors_cnt == 0:
        mean_error = max_reprojection_error
    else:
        mean_error = np.mean(errors[errors < max_reprojection_error])
    return mean_error, errors_cnt


def soften_parameters(parameters: TriangulationParameters):
    return TriangulationParameters(max_reprojection_error=min(8.0, parameters.max_reprojection_error + 2.0),
                                   min_triangulation_angle_deg=max(1.0, parameters.min_triangulation_angle_deg - 0.5),
                                   min_depth=0.1)


def check_points(points, points3d, ids, proj_mats, max_reprojection_error, st):
    for i, point3d in enumerate(points3d):
        point = points[ids[i]]
        if not st[i]:
            continue
        if point.pose3d is None:
            point.pose3d = point3d
            continue
        frames = points[ids[i]].frames
        points2d = np.array(point.poses2d)
        mean_error_before, inliers_before = count_inliers_and_mean_error(points2d, proj_mats[frames],
                                                                         point.pose3d,
                                                                         max_reprojection_error)
        mean_error_after, inliers_after = count_inliers_and_mean_error(points2d, proj_mats[frames],
                                                                       point3d,
                                                                       max_reprojection_error)
        if inliers_before > inliers_after or (
                inliers_before == inliers_after and mean_error_before <= mean_error_after):
            st[i] = False
    return st


def adjust(points, point_cloud_builder, corner_storage, view_mats, processed_view_mats, intrinsic_mat,
           max_reprojection_error):
    frame_inds = np.argwhere(processed_view_mats).flatten()
    ids = point_cloud_builder.ids.flatten()
    print(f"\rSolving bundle adjustment problem...", end="")
    view_mats[processed_view_mats], points3d = bundle_adjustment(corner_storage,
                                                                 point_cloud_builder.points,
                                                                 point_cloud_builder.ids,
                                                                 view_mats[processed_view_mats], frame_inds,
                                                                 intrinsic_mat)
    proj_mats = intrinsic_mat @ view_mats
    st = check_points(points, points3d, ids, proj_mats, max_reprojection_error, np.full(len(points3d), True))
    for point_id, point3d in zip(ids, points3d[st]):
        points[point_id].pose3d = point3d
    point_cloud_builder.add_points(point_cloud_builder.ids[st], points3d[st])
    return view_mats


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    np.random.seed(197)

    triangulation_parameters = TriangulationParameters(max_reprojection_error=4.0,
                                                       min_triangulation_angle_deg=2.0,
                                                       min_depth=0.1)

    RETRIANGULATION_FRAME_COUNT = 10
    RETRIANGULATION_MIN_INLIERS = 6
    MIN_SHARE_OF_INLIERS = 0.6

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    known_view_1, known_view_2 = detect_motion(intrinsic_mat, corner_storage, known_view_1, known_view_2)

    points = defaultdict(Point)
    frame_count = len(corner_storage)

    processed_view_mats = np.full(frame_count, False)
    view_mats = np.zeros((frame_count, 3, 4))
    proj_mats = np.zeros((frame_count, 3, 4))
    times_passed = np.zeros(frame_count, dtype=np.int)

    frame_1, camera_pose_1 = known_view_1
    frame_2, camera_pose_2 = known_view_2

    add_frame(points, corner_storage[frame_1], frame_1)
    add_frame(points, corner_storage[frame_2], frame_2)

    view_mat_1 = pose_to_view_mat3x4(camera_pose_1)
    view_mat_2 = pose_to_view_mat3x4(camera_pose_2)

    view_mats[frame_1] = view_mat_1
    view_mats[frame_2] = view_mat_2
    processed_view_mats[[frame_1, frame_2]] = True

    proj_mats[frame_1] = intrinsic_mat @ view_mat_1
    proj_mats[frame_2] = intrinsic_mat @ view_mat_2

    # определение структуры сцены
    initial_correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    while True:
        points3d, ids, _ = triangulate_correspondences(initial_correspondences,
                                                       view_mat_1, view_mat_2,
                                                       intrinsic_mat,
                                                       triangulation_parameters)
        share_of_inliers = len(ids) / len(initial_correspondences.ids)
        if share_of_inliers >= MIN_SHARE_OF_INLIERS or triangulation_parameters.max_reprojection_error == 8.0:
            break
        triangulation_parameters = soften_parameters(triangulation_parameters)

    point_cloud_builder = PointCloudBuilder(ids, points3d)
    # определение движения относительно точек сцены
    is_changing = True
    processed_frames = 0
    while is_changing:
        is_changing = False
        last_processed_frames = deque([])  # последние RETRIANGULATION_FRAME_COUNT обработанных кадров
        for frame_1 in range(frame_count):
            if processed_view_mats[frame_1]:
                continue
            corners = corner_storage[frame_1]
            intersection, (ids_3d, ids_2d) = snp.intersect(point_cloud_builder.ids.flatten(),
                                                           corners.ids.flatten(),
                                                           indices=True)
            if len(intersection) < 4:
                continue
            succeeded, r_vec, t_vec, inliers = solvePnP(point_cloud_builder.points[ids_3d], corners.points[ids_2d],
                                                        intrinsic_mat, triangulation_parameters.max_reprojection_error)
            if succeeded:
                share_of_inliers = len(inliers) / len(intersection)
                if share_of_inliers < 0.1 and times_passed[frame_1] < 3:
                    times_passed[frame_1] += 1
                    is_changing = True
                    continue
                view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
                view_mats[frame_1] = view_mat
                proj_mats[frame_1] = intrinsic_mat @ view_mat
                processed_view_mats[frame_1] = True
                last_processed_frames.append(frame_1)
                add_frame(points, corner_storage[frame_1], frame_1)
                processed_frames += 1
                print(f"\rProcessing frame {frame_1}, inliers: {len(inliers)}, processed {processed_frames} out"
                      f" of {frame_count} frames, {len(point_cloud_builder.ids)} points in cloud", end="")

                if processed_frames % (frame_count // 3) == 0:
                    view_mats = adjust(points, point_cloud_builder, corner_storage, view_mats, processed_view_mats,
                                       intrinsic_mat, triangulation_parameters.max_reprojection_error)
                    proj_mats = intrinsic_mat @ view_mats
                # дополнение структуры сцены
                corners_1 = filter_frame_corners(corner_storage[frame_1], inliers)
                for frame_2 in range(frame_count):
                    if not processed_view_mats[frame_2]:
                        continue
                    corners_2 = corner_storage[frame_2]
                    correspondences = build_correspondences(corners_1, corners_2, point_cloud_builder.ids)
                    if len(correspondences.ids) == 0:
                        continue
                    points3d, ids, _ = triangulate_correspondences(correspondences,
                                                                   view_mat, view_mats[frame_2],
                                                                   intrinsic_mat,
                                                                   triangulation_parameters)
                    is_changing = True
                    for point_id, point3d in zip(ids, points3d):
                        points[point_id].pose3d = point3d
                    point_cloud_builder.add_points(ids, points3d)

                # ретриангуляция
                if len(last_processed_frames) < RETRIANGULATION_FRAME_COUNT:
                    continue
                view_mat_list = [view_mats[frame] for frame in last_processed_frames]
                ids_retriangulation = corner_storage[last_processed_frames[0]].ids.flatten()
                for frame in last_processed_frames:
                    ids_retriangulation = snp.intersect(ids_retriangulation, corner_storage[frame].ids.flatten())

                if len(ids_retriangulation) > 0:
                    points2d_list = []
                    for frame in last_processed_frames:
                        _, (_, ids) = snp.intersect(ids_retriangulation,
                                                    corner_storage[frame].ids.flatten(),
                                                    indices=True)
                        points2d_list.append(corner_storage[frame].points[ids])

                    points3d, st = retriangulate_points_ransac(np.array(points2d_list),
                                                               np.array(view_mat_list),
                                                               intrinsic_mat,
                                                               RETRIANGULATION_MIN_INLIERS,
                                                               triangulation_parameters)

                    st = check_points(points, points3d, ids_retriangulation, proj_mats,
                                      triangulation_parameters.max_reprojection_error, st)
                    for point_id, point3d in zip(ids_retriangulation[st], points3d[st]):
                        points[point_id].pose3d = point3d
                    point_cloud_builder.add_points(ids_retriangulation[st], points3d[st])
                for i in range(RETRIANGULATION_FRAME_COUNT // 3):
                    last_processed_frames.popleft()
    print(f"\rProcessed {processed_frames + 2} out of {frame_count} frames,"
          f" {len(point_cloud_builder.ids)} points in cloud")

    # если какие-то позиции вычислить не удалось, возьмем ближайшие вычисленные
    view_mats_processed_inds = np.argwhere(processed_view_mats)
    if len(view_mats_processed_inds) == 0:
        print("\rFailed to solve scene")
        exit(0)

    first_processed_view_mat = view_mats[view_mats_processed_inds[0]]
    view_mats[0] = first_processed_view_mat
    for i in range(1, len(view_mats)):
        if not processed_view_mats[i]:
            view_mats[i] = view_mats[i - 1]

    view_mats = adjust(points, point_cloud_builder, corner_storage, view_mats, processed_view_mats, intrinsic_mat,
                       triangulation_parameters.max_reprojection_error)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
