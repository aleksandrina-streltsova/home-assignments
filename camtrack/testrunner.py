#! /usr/bin/env python3

from collections import namedtuple
import contextlib
import os
from os import path

import click
from good import Any, Default, Invalid, Optional, Schema
import numpy as np
import yaml

import camtrack
import cmptrack
import corners
import data3d
import frameseq


FramePair = namedtuple('FramePair', ('frame_1', 'frame_2'))


def _check_frame_pair(frame_pair):
    if (not isinstance(frame_pair, list) or
            len(frame_pair) != 2 or
            not all(isinstance(x, int) for x in frame_pair)):
        raise Invalid('Invalid initial frame pair format')
    return FramePair(*frame_pair)


DATASET_CONFIG_SCHEMA = Schema({
    'tests': {
        Any(str): {
            'camera': str,
            'ground_truth': str,
            'rgb': str,
            Optional('initial_frames'): Any(_check_frame_pair, Default(None))
        }
    }
})


TestInfo = namedtuple('TestInfo', (
    'camera',
    'ground_truth',
    'rgb',
    'initial_frames'
))


def read_config(config_path):
    root = path.dirname(path.abspath(config_path))
    with open(config_path, 'r') as config_file:
        raw_config_data = yaml.load(config_file)
    config_data = DATASET_CONFIG_SCHEMA(raw_config_data)
    config = dict()
    for name, info in config_data['tests'].items():
        config[name] = TestInfo(**{
            k: path.join(root, v) if isinstance(v, str) else v
            for k, v in info.items()
        })
    return config


def _run_and_save_logs(stdout_path, stderr_path, func, *args, **kwargs):
    with open(stdout_path, 'w') as stdout_file:
        with open(stderr_path, 'w') as stderr_file:
            with contextlib.redirect_stdout(stdout_file):
                with contextlib.redirect_stderr(stderr_file):
                    result = func(*args, **kwargs)
    return result


def _make_dir_if_needed(dir_path, indent_level=0):
    if not dir_path:
        return
    if not path.exists(dir_path):
        click.echo("{}make dir '{}'".format('  ' * indent_level, dir_path))
        os.mkdir(dir_path)


def _read_camera_parameters(parameters_path):
    with open(parameters_path, 'r') as camera_file:
        return data3d.read_camera_parameters(camera_file)


def _write_poses(track, track_path):
    with open(track_path, 'w') as track_file:
        data3d.write_poses(track, track_file)


def _write_point_cloud(point_cloud, pc_path):
    with open(pc_path, 'w') as pc_file:
        data3d.write_point_cloud(point_cloud, pc_file)


def _read_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as gt_file:
        return data3d.read_poses(gt_file)


def _write_error_measure(error, dst_path):
    with open(dst_path, 'w') as dst_file:
        yaml.dump({'error_measure': float(error)}, dst_file)


def _calc_corners_path(test_name, corners_dir):
    if corners_dir is None:
        return None
    return path.join(corners_dir, test_name + '.pickle')


def _try_to_load_corners(corners_path):
    if not corners_path:
        return None
    if not path.exists(corners_path):
        return None
    with open(corners_path, 'rb') as corners_file:
        return corners.load(corners_file)


def _try_to_dump_corners(corners_path, corner_storage):
    if not corners_path:
        return None
    with open(corners_path, 'wb') as corners_file:
        return corners.dump(corner_storage, corners_file)


def _load_or_calculate_corners(grayscale_seq, test_name,
                               test_dir, corners_dir):
    corners_path = _calc_corners_path(test_name, corners_dir)
    corner_storage = _try_to_load_corners(corners_path)
    if corner_storage:
        click.echo("  corners are loaded from '{}'".format(corners_path))
        return corner_storage
    try:
        click.echo('  start corners tracking')
        corner_storage = _run_and_save_logs(
            path.join(test_dir, 'corners_stdout.txt'),
            path.join(test_dir, 'corners_stderr.txt'),
            corners.build,
            grayscale_seq,
            False
        )
    except Exception as err:  # pylint:disable=broad-except
        click.echo('  corners tracking failed: {}'.format(err))
        return None
    else:
        click.echo('  corners tracking succeeded')
        if corners_path:
            _try_to_dump_corners(corners_path, corner_storage)
            click.echo("  corners are dumped to '{}'".format(corners_path))
        return corner_storage


def _do_tracking(test_info, ground_truth, corner_storage, test_dir):
    camera_parameters = _read_camera_parameters(test_info.camera)
    if test_info.initial_frames is not None:
        frame_1, frame_2 = test_info.initial_frames
        known_view_1 = (frame_1, ground_truth[frame_1])
        known_view_2 = (frame_2, ground_truth[frame_2])
    else:
        known_view_1 = None
        known_view_2 = None
    try:
        click.echo('  start scene solving')
        track, point_cloud = _run_and_save_logs(
            path.join(test_dir, 'tracking_stdout.txt'),
            path.join(test_dir, 'tracking_stderr.txt'),
            camtrack.track_and_calc_colors,
            camera_parameters,
            corner_storage,
            test_info.rgb,
            known_view_1,
            known_view_2
        )
    except Exception as err:  # pylint:disable=broad-except
        click.echo('  scene solving failed: {}'.format(err))
        return None, None
    else:
        click.echo('  scene solving succeeded')
        return track, point_cloud


def run_tests(config, output_dir, corners_dir):
    # pylint:disable=too-many-locals

    _make_dir_if_needed(output_dir)
    _make_dir_if_needed(corners_dir)

    all_r_errors = []
    all_t_errors = []
    for test_name, test_info in config.items():
        click.echo(test_name)

        test_dir = path.join(output_dir, test_name)
        _make_dir_if_needed(test_dir, 1)

        grayscale_seq = frameseq.read_grayscale_f32(test_info.rgb)

        inf_errors = np.full((len(grayscale_seq),), np.inf)
        all_r_errors.append(inf_errors)
        all_t_errors.append(inf_errors)

        corner_storage = _load_or_calculate_corners(grayscale_seq, test_name,
                                                    test_dir, corners_dir)
        if not corner_storage:
            continue

        ground_truth = _read_ground_truth(test_info.ground_truth)
        track, point_cloud = _do_tracking(test_info, ground_truth,
                                          corner_storage, test_dir)
        if not track:
            continue

        _write_poses(track, path.join(test_dir, 'track.yml'))
        _write_point_cloud(point_cloud, path.join(test_dir, 'point_cloud.yml'))

        r_errors, t_errors = cmptrack.calc_errors(ground_truth, track)
        all_r_errors[-1] = r_errors
        all_t_errors[-1] = t_errors
        click.echo('  error measure: {}'.format(
            cmptrack.calc_vol_under_surface(r_errors, t_errors)
        ))

    all_r_errors = np.concatenate(all_r_errors)
    all_t_errors = np.concatenate(all_t_errors)
    error_measure = cmptrack.calc_vol_under_surface(all_r_errors, all_t_errors)

    click.echo('overall error measure: {}'.format(error_measure))
    _write_error_measure(error_measure,
                         path.join(output_dir, 'error_measure.yml'))


@click.command()
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--corners-dir', type=click.Path(file_okay=False))
def cli(config_path, output_dir, corners_dir):
    config = read_config(config_path)
    run_tests(config, output_dir, corners_dir)


if __name__ == '__main__':
    cli()  # pylint:disable=no-value-for-parameter
