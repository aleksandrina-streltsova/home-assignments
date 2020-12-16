#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp

import frameseq
from _camtrack import *
from _corners import filter_frame_corners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


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


def check_points(point_cloud_builder, points3d, ids, max_reprojection_error, st):
    for i, point3d in enumerate(points3d):
        pointWithPose = point_cloud_builder.points_with_poses[ids[i]]
        if not st[i]:
            continue
        if pointWithPose.pose3d is None:
            pointWithPose.pose3d = point3d
            continue
        frames = pointWithPose.frames
        points2d = np.array(pointWithPose.poses2d)
        mean_error_before, inliers_before = count_inliers_and_mean_error(points2d,
                                                                         point_cloud_builder.proj_mats[frames],
                                                                         pointWithPose.pose3d,
                                                                         max_reprojection_error)
        mean_error_after, inliers_after = count_inliers_and_mean_error(points2d,
                                                                       point_cloud_builder.proj_mats[frames],
                                                                       point3d,
                                                                       max_reprojection_error)
        if inliers_before > inliers_after or (
                inliers_before == inliers_after and mean_error_before <= mean_error_after):
            st[i] = False
    return st


def adjust(point_cloud_builder, corner_storage, max_reprojection_error):
    frame_inds = np.argwhere(point_cloud_builder.processed_view_mats).flatten()
    ids = point_cloud_builder.ids.flatten()
    print(f"\rSolving bundle adjustment problem...", end="")
    point_cloud_builder.view_mats[frame_inds], points3d = bundle_adjustment(corner_storage,
                                                                            point_cloud_builder.points,
                                                                            point_cloud_builder.ids,
                                                                            point_cloud_builder.view_mats[frame_inds],
                                                                            frame_inds,
                                                                            point_cloud_builder.intrinsic_mat)
    point_cloud_builder.proj_mats = point_cloud_builder.intrinsic_mat @ point_cloud_builder.view_mats
    st = check_points(point_cloud_builder, points3d, ids, max_reprojection_error, np.full(len(points3d), True))
    point_cloud_builder.add_points(point_cloud_builder.ids[st], points3d[st])


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
    if known_view_1 is None or known_view_2 is None:
        print("\rFailed to solve scene")
        exit(0)

    frame_count = len(corner_storage)

    frame_1, camera_pose_1 = known_view_1
    frame_2, camera_pose_2 = known_view_2

    view_mat_1 = pose_to_view_mat3x4(camera_pose_1)
    view_mat_2 = pose_to_view_mat3x4(camera_pose_2)

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

    point_cloud_builder = MyPointCloudBuilder(frame_count, intrinsic_mat, ids, points3d)

    point_cloud_builder.add_frame(corner_storage[frame_1], frame_1)
    point_cloud_builder.add_frame(corner_storage[frame_2], frame_2)
    point_cloud_builder.set_view_mat(frame_1, view_mat_1)
    point_cloud_builder.set_view_mat(frame_2, view_mat_2)

    # определение движения относительно точек сцены
    is_changing = True
    first_time = True
    last_processed_frames = deque([])  # последние RETRIANGULATION_FRAME_COUNT обработанных кадров
    while is_changing:
        is_changing = False
        processing_range = range(known_view_1[0], frame_count) if first_time else range(frame_count - 1, -1, -1)
        for frame_1 in processing_range:
            if point_cloud_builder.processed_view_mats[frame_1]:
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
                if share_of_inliers < 0.1 and point_cloud_builder.times_passed[frame_1] < 3:
                    point_cloud_builder.times_passed[frame_1] += 1
                    is_changing = True
                    continue
                if share_of_inliers > 0.1:
                    last_processed_frames.append(frame_1)
                view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
                point_cloud_builder.set_view_mat(frame_1, view_mat)
                point_cloud_builder.add_frame(corner_storage[frame_1], frame_1)
                print(f"\rProcessing frame {frame_1}, inliers: {len(inliers)},"
                      f" processed {point_cloud_builder.processed_frames} out"
                      f" of {frame_count} frames, {len(point_cloud_builder.ids)} points in cloud", end="")

                # дополнение структуры сцены
                corners_1 = filter_frame_corners(corner_storage[frame_1], inliers)
                for frame_2 in range(frame_count):
                    if not point_cloud_builder.processed_view_mats[frame_2]:
                        continue
                    corners_2 = corner_storage[frame_2]
                    correspondences = build_correspondences(corners_1, corners_2, point_cloud_builder.ids)
                    if len(correspondences.ids) == 0:
                        continue
                    points3d, ids, _ = triangulate_correspondences(correspondences, view_mat,
                                                                   point_cloud_builder.view_mats[frame_2],
                                                                   intrinsic_mat,
                                                                   triangulation_parameters)
                    is_changing = True
                    point_cloud_builder.add_points(ids, points3d)

                # ретриангуляция
                if len(last_processed_frames) < RETRIANGULATION_FRAME_COUNT:
                    continue
                view_mat_list = [point_cloud_builder.view_mats[frame] for frame in last_processed_frames]
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

                    st = check_points(point_cloud_builder, points3d, ids_retriangulation,
                                      triangulation_parameters.max_reprojection_error, st)
                    point_cloud_builder.add_points(ids_retriangulation[st], points3d[st])
                for i in range(RETRIANGULATION_FRAME_COUNT // 3):
                    last_processed_frames.popleft()
        first_time = False
    print(f"\rProcessed {point_cloud_builder.processed_frames} out of {frame_count} frames,"
          f" {len(point_cloud_builder.ids)} points in cloud")
    # adjust(point_cloud_builder, corner_storage, triangulation_parameters.max_reprojection_error)
    # если какие-то позиции вычислить не удалось, возьмем ближайшие вычисленные
    view_mats_processed_inds = np.argwhere(point_cloud_builder.processed_view_mats)
    if len(view_mats_processed_inds) == 0:
        print("\rFailed to solve scene")
        exit(0)

    point_cloud_builder.set_view_mat(0, point_cloud_builder.view_mats[view_mats_processed_inds[0]])
    for i in range(1, frame_count):
        if not point_cloud_builder.processed_view_mats[i]:
            point_cloud_builder.set_view_mat(i, point_cloud_builder.view_mats[i - 1])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        point_cloud_builder.view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, point_cloud_builder.view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
