"""ROS 2 Humble image -> ArUco pose; run with /usr/bin/python3, not Isaac Python."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

EDGE = .024


def detect(rgb: np.ndarray, K: np.ndarray, distortion: np.ndarray):
    """Estimate marker pose solely from the received image and CameraInfo."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return None
    half = EDGE / 2
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    for pixels, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) != 0:
            continue
        image_points = pixels.reshape(4, 2).astype(np.float64)
        result = cv2.solvePnPGeneric(object_points, image_points, K, distortion,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            if tvec[2, 0] <= 0:
                continue
            projected = cv2.projectPoints(object_points, rvec, tvec, K, distortion)[0]
            error = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
            candidates.append((error, rvec, tvec))
        if not candidates:
            return None
        error, rvec, tvec = min(candidates, key=lambda item: item[0])
        if error > 2.0:
            return None
        rotation = cv2.Rodrigues(rvec)[0]
        # Convert rotation matrix to quaternion without a SciPy dependency.
        trace = float(np.trace(rotation))
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            quat = ((rotation[2, 1] - rotation[1, 2]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[1, 0] - rotation[0, 1]) / s, s / 4)
        else:
            # OpenCV's Rodrigues vector is also a valid axis-angle source.
            theta = float(np.linalg.norm(rvec))
            axis = rvec.ravel() / theta if theta else np.array([0., 0., 1.])
            quat = (*tuple(axis * np.sin(theta / 2)), np.cos(theta / 2))
        return {"id": 0, "t": tvec.ravel().tolist(), "q": list(map(float, quat)),
                "error_px": error, "corners_px": image_points.tolist()}
    return None


class Detector(Node):
    def __init__(self, output: Path | None, view: bool = False):
        super().__init__("optics_aruco_pose")
        self.info = None
        self.output = output.open("w") if output else None
        self.view = view
        self.pub = self.create_publisher(PoseStamped, "/optics/marker_0/pose", 10)
        self.create_subscription(CameraInfo, "/optics/camera/camera_info", self.on_info, 10)
        self.create_subscription(Image, "/optics/camera/image_raw", self.on_image, 10)

    def on_info(self, msg: CameraInfo):
        if msg.width and msg.height and msg.k[0] > 0 and msg.k[4] > 0:
            self.info = msg

    def on_image(self, msg: Image):
        if self.info is None or (msg.width, msg.height) != (self.info.width, self.info.height):
            return
        if msg.encoding not in ("rgb8", "bgr8") or msg.step < msg.width * 3:
            self.get_logger().warning(f"unsupported image layout: {msg.encoding}")
            return
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        rgb = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        K = np.array(self.info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.array(self.info.d, dtype=np.float64)
        found = detect(rgb, K, distortion)
        record = {"stamp_sec": msg.header.stamp.sec,
                  "stamp_nanosec": msg.header.stamp.nanosec,
                  "frame_id": msg.header.frame_id, "detected": found is not None}
        if found:
            record.update(found)
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = found["t"]
            (pose.pose.orientation.x, pose.pose.orientation.y,
             pose.pose.orientation.z, pose.pose.orientation.w) = found["q"]
            self.pub.publish(pose)
            self.get_logger().info(f"marker 0 t={np.round(found['t'], 4)} error={found['error_px']:.2f}px")
        if self.view:
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if found:
                cv2.polylines(display, [np.array(found["corners_px"], dtype=np.int32)],
                              True, (0, 255, 0), 2)
                cv2.putText(display, "ID 0", tuple(np.array(found["corners_px"][0], dtype=int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            cv2.imshow("Isaac camera / ArUco", display)
            cv2.waitKey(1)
        if self.output:
            self.output.write(json.dumps(record) + "\n")
            self.output.flush()

    def destroy_node(self):
        if self.output:
            self.output.close()
        if self.view:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Detector(args.output, args.view)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
