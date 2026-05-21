#!/usr/bin/env python3

import cv2
import rclpy
import rclpy.action
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

from stretch_deep_perception import object_detect_pytorch as od

# Rotated-frame geometry (cv2.ROTATE_90_CLOCKWISE on a 640x480 image -> 480x640)
FRAME_W = 480
FRAME_H = 640
FRAME_CX = FRAME_W // 2
FRAME_CY = FRAME_H // 2

# Servoing gains and limits
K_ROT = 0.003       # rad per pixel of horizontal error
K_TRANS = 0.002     # m per pixel of vertical error
MAX_DTHETA = 0.15   # rad per iteration
MAX_DX = 0.10       # m per iteration
TOL_X = 10          # px
TOL_Y = 15          # px

DINING_TABLE_LABELS = ('dining table', 'diningtable')


def _clamp(value, limit):
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        self.get_logger().info('Loading YOLOv5s model...')
        self.detector = od.ObjectDetector(confidence_threshold=0.2)
        self.bridge = CvBridge()

        self._action_client = ActionClient(
            self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory'
        )
        self._moving = False
        self.centered = False

        self.create_subscription(Image, '/camera/color/image_raw', self._image_callback, 1)

        self.detections_pub = self.create_publisher(Detection2DArray, '/object_detections/boxes', 10)
        self.annotated_pub = self.create_publisher(Image, '/object_detections/image', 10)

        self.get_logger().info('Object detection node ready')

    def _image_callback(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # rotate to match camera orientation, same as detection_node.py
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

        detections_2d, annotated = self.detector.apply_to_image(bgr, draw_output=True)

        det_array = Detection2DArray()
        det_array.header = msg.header
        for d in detections_2d:
            x_min, y_min, x_max, y_max = d['box']
            det = Detection2D()
            det.header = msg.header
            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = float((x_min + x_max) / 2)
            det.bbox.center.position.y = float((y_min + y_max) / 2)
            det.bbox.size_x = float(x_max - x_min)
            det.bbox.size_y = float(y_max - y_min)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d['label']
            hyp.hypothesis.score = float(d['confidence'])
            det.results.append(hyp)
            det_array.detections.append(det)

        self.detections_pub.publish(det_array)

        if annotated is not None:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

        self._servo_to_dining_table(detections_2d)

    def _servo_to_dining_table(self, detections_2d):
        if self.centered or self._moving:
            return

        target = None
        target_area = 0.0
        for d in detections_2d:
            label = str(d.get('label', '')).lower()
            if label not in DINING_TABLE_LABELS:
                continue
            x_min, y_min, x_max, y_max = d['box']
            area = max(0.0, (x_max - x_min)) * max(0.0, (y_max - y_min))
            if area > target_area:
                target_area = area
                target = d

        if target is None:
            return

        x_min, y_min, x_max, y_max = target['box']
        x_center = 0.5 * (x_min + x_max)
        err_x = x_center - FRAME_CX
        err_y = y_max - FRAME_CY

        self.get_logger().info(
            f'dining_table bbox=({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) '
            f'err_x={err_x:+.1f}px err_y={err_y:+.1f}px'
        )

        if abs(err_x) < TOL_X and abs(err_y) < TOL_Y:
            self.centered = True
            self.get_logger().info('Centered on dining table. Stopping.')
            return

        if abs(err_x) / TOL_X >= abs(err_y) / TOL_Y:
            joint_name = 'rotate_mobile_base'
            position = _clamp(-K_ROT * err_x, MAX_DTHETA)
            self.get_logger().info(f'rotate_by({position:+.3f} rad)')
        else:
            joint_name = 'translate_mobile_base'
            position = _clamp(-K_TRANS * err_y, MAX_DX)
            self.get_logger().info(f'translate_by({position:+.3f} m)')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]

        self._moving = True
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Motion goal rejected')
            self._moving = False
            return
        goal_handle.get_result_async().add_done_callback(
            lambda f: setattr(self, '_moving', False)
        )


def main():
    rclpy.init()
    node = ObjectDetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
