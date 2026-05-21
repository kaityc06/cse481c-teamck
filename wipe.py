#!/usr/bin/env python3

"""Execute a small local wiping motion with a Stretch robot.

The table path planner should position the robot at the next wipe patch, then
call ``wipe_once`` with its existing HelloNode or run this file as a script.
"""

from dataclasses import dataclass
import argparse
import time

import rclpy

import hello_helpers.hello_misc as hm


@dataclass
class WipeConfig:
    """Parameters for one local wipe patch."""

    wrist_extension_m: float = 0.25
    wrist_pitch_rad: float = -1.1
    wrist_roll_rad: float = 0.0
    yaw_center_rad: float = 0.0
    yaw_amplitude_rad: float = 0.8
    strokes: int = 2
    approach_distance_m: float = 0.08
    contact_effort: float = 33.7
    lift_retreat_m: float = 0.04
    settle_s: float = 0.2
    surface_lift_m: float | None = None
    close_gripper: bool = False
    gripper_aperture_m: float = -0.02


def _wait_for_joint_states(node, timeout_s=5.0):
    start_s = time.monotonic()
    while rclpy.ok() and not node.joint_state.name:
        if time.monotonic() - start_s > timeout_s:
            raise RuntimeError('Timed out waiting for /stretch/joint_states')
        time.sleep(0.05)


def _joint_position(node, joint_name):
    _wait_for_joint_states(node)
    try:
        index = node.joint_state.name.index(joint_name)
    except ValueError as exc:
        raise RuntimeError(f'Joint {joint_name!r} was not found in /stretch/joint_states') from exc
    return node.joint_state.position[index]


def _move(node, pose, message, custom_contact_thresholds=False):
    node.get_logger().info(message)
    node.move_to_pose(pose, blocking=True, custom_contact_thresholds=custom_contact_thresholds)


def _sweep_yaw(node, config):
    left_yaw = config.yaw_center_rad + config.yaw_amplitude_rad
    right_yaw = config.yaw_center_rad - config.yaw_amplitude_rad

    for stroke_index in range(config.strokes):
        node.get_logger().info(f'Wipe stroke {stroke_index + 1}/{config.strokes}: left')
        node.move_to_pose({'joint_wrist_yaw': left_yaw}, blocking=True)
        node.get_logger().info(f'Wipe stroke {stroke_index + 1}/{config.strokes}: right')
        node.move_to_pose({'joint_wrist_yaw': right_yaw}, blocking=True)

    _move(node, {'joint_wrist_yaw': config.yaw_center_rad}, 'Centering wrist yaw')


def wipe_once(node, config=None):
    """Run one wipe at the robot's current base/table position.

    The caller is responsible for navigating the robot to the table patch. If
    ``surface_lift_m`` is omitted, this routine lowers from the current lift
    position by ``approach_distance_m`` using a light contact threshold.
    """

    config = config or WipeConfig()
    if config.strokes < 1:
        raise ValueError('strokes must be at least 1')
    if config.approach_distance_m < 0.0:
        raise ValueError('approach_distance_m must be non-negative')
    if config.lift_retreat_m < 0.0:
        raise ValueError('lift_retreat_m must be non-negative')

    _wait_for_joint_states(node)
    start_lift_m = _joint_position(node, 'joint_lift')
    if config.surface_lift_m is None:
        hover_lift_m = start_lift_m + config.lift_retreat_m
    else:
        hover_lift_m = config.surface_lift_m + config.lift_retreat_m

    if config.close_gripper:
        _move(
            node,
            {'gripper_aperture': config.gripper_aperture_m},
            f'Closing gripper to {config.gripper_aperture_m:.3f} m',
        )

    _move(node, {'joint_lift': hover_lift_m}, f'Hovering at lift {hover_lift_m:.3f} m')

    _move(
        node,
        {
            'joint_wrist_yaw': config.yaw_center_rad,
            'joint_wrist_pitch': config.wrist_pitch_rad,
            'joint_wrist_roll': config.wrist_roll_rad,
            'wrist_extension': config.wrist_extension_m,
        },
        'Moving wrist into wiping pose',
    )

    if config.surface_lift_m is None:
        target_lift_m = hover_lift_m - config.approach_distance_m
        _move(
            node,
            {'joint_lift': (target_lift_m, config.contact_effort)},
            (
                f'Lowering up to {config.approach_distance_m:.3f} m '
                f'or until contact effort {config.contact_effort:.1f}'
            ),
            custom_contact_thresholds=True,
        )
    else:
        target_lift_m = config.surface_lift_m
        _move(
            node,
            {'joint_lift': target_lift_m},
            f'Lowering to supplied surface lift {target_lift_m:.3f} m',
        )

    time.sleep(config.settle_s)
    _sweep_yaw(node, config)

    current_lift_m = _joint_position(node, 'joint_lift')
    retreat_lift_m = current_lift_m + config.lift_retreat_m
    _move(node, {'joint_lift': retreat_lift_m}, f'Raising {config.lift_retreat_m:.3f} m off surface')

    return True


def _parse_args():
    parser = argparse.ArgumentParser(description='Run one local Stretch wiping motion.')
    parser.add_argument('--extension', type=float, default=WipeConfig.wrist_extension_m,
                        help='Wrist extension for the wipe patch in meters.')
    parser.add_argument('--pitch', type=float, default=WipeConfig.wrist_pitch_rad,
                        help='Wrist pitch in radians.')
    parser.add_argument('--roll', type=float, default=WipeConfig.wrist_roll_rad,
                        help='Wrist roll in radians.')
    parser.add_argument('--yaw-center', type=float, default=WipeConfig.yaw_center_rad,
                        help='Center wrist yaw angle in radians.')
    parser.add_argument('--yaw-amplitude', type=float, default=WipeConfig.yaw_amplitude_rad,
                        help='Wrist yaw sweep amplitude in radians.')
    parser.add_argument('--strokes', type=int, default=WipeConfig.strokes,
                        help='Number of left/right yaw sweeps.')
    parser.add_argument('--approach-distance', type=float, default=WipeConfig.approach_distance_m,
                        help='Distance to lower from the current lift if --surface-lift is omitted.')
    parser.add_argument('--contact-effort', type=float, default=WipeConfig.contact_effort,
                        help='Lift contact threshold effort used while lowering.')
    parser.add_argument('--surface-lift', type=float, default=None,
                        help='Absolute joint_lift target for the table surface. If omitted, lower by contact.')
    parser.add_argument('--retreat', type=float, default=WipeConfig.lift_retreat_m,
                        help='Distance to raise after wiping.')
    parser.add_argument('--settle', type=float, default=WipeConfig.settle_s,
                        help='Seconds to pause after reaching contact/surface lift.')
    parser.add_argument('--close-gripper', action='store_true',
                        help='Close the gripper before wiping.')
    parser.add_argument('--gripper-aperture', type=float, default=WipeConfig.gripper_aperture_m,
                        help='Gripper aperture used with --close-gripper.')
    return parser.parse_args()


def _config_from_args(args):
    return WipeConfig(
        wrist_extension_m=args.extension,
        wrist_pitch_rad=args.pitch,
        wrist_roll_rad=args.roll,
        yaw_center_rad=args.yaw_center,
        yaw_amplitude_rad=args.yaw_amplitude,
        strokes=args.strokes,
        approach_distance_m=args.approach_distance,
        contact_effort=args.contact_effort,
        lift_retreat_m=args.retreat,
        settle_s=args.settle,
        surface_lift_m=args.surface_lift,
        close_gripper=args.close_gripper,
        gripper_aperture_m=args.gripper_aperture,
    )


def main():
    args = _parse_args()
    config = _config_from_args(args)
    node = hm.HelloNode.quick_create('wipe_node', wait_for_first_pointcloud=False)

    try:
        node.switch_to_position_mode()
        wipe_once(node, config)
        node.get_logger().info('Wipe complete')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
