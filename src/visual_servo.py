#!/usr/bin/env python

import rospy
from rospy import Header
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TwistStamped
from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry 
import torch

from core_trajectory_msgs.msg import FixedTrajectory
from diagnostic_msgs.msg import KeyValue

import tf
from tf.transformations import euler_matrix, translation_matrix, concatenate_matrices


class VisualServo:
    def __init__(self):
        
        # Camera parameters and control constants
        self.fx = 410.9  # Focal length in x
        self.fy = 410.9  # Focal length in y
        self.cx = 640.0  # Optical center x
        self.cy = 540.0  # Optical center y

        # Load the YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/mohammad/workspace/16-667/src/visual_servo_landing/model/yolov5_landing_pad/best.pt')

        # ROS Subscribers and Publishers

        rospy.init_node('bolt_head_follower', anonymous=True)
        odom_sub = rospy.Subscriber('odometry', Odometry, self.odom_callback, queue_size=10)
        rgb_image_sub = Subscriber('camera/color/image_raw', Image)
        depth_image_sub = Subscriber("camera/aligned_depth_to_color/image_raw", Image)
        img_tss = TimeSynchronizer([rgb_image_sub, depth_image_sub], 10)
        img_tss.registerCallback(self.image_callback)

        self.velocity_pub = rospy.Publisher("velocity_setpoint", TwistStamped, queue_size=10)
        self.img_debug_pub = rospy.Publisher("image_debug", Image, queue_size=2)
        self.pose_ctrl_mute_pub = rospy.Publisher("pose_controller/mute_control", Bool, queue_size=100)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=100)

        self.odom = Odometry()
        self.got_odom = False

    def odom_callback(self, odom):
        self.odom = odom
        self.got_odom = True
        
    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert the ROS image messages to OpenCV images
            rgb = CvBridge().imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
        except Exception as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        x, y, z = self.detect_landing_pad(rgb, depth)

        # self.pbvs_control(x, y, z)        
        self.ibvs_control(x, y, z)

    def camera_to_body(self, camera_x, camera_y, camera_z):

        # Camera to robot transformation (rotation followed by translation)
        R_cam_robot = euler_matrix(0, 3.1415, 1.57)  # Rotation matrix
        t_cam_robot = np.array([-0.051, 0, -0.162])  # Translation vector
        T_cam_robot = concatenate_matrices(R_cam_robot, translation_matrix(t_cam_robot))
        # T_cam_robot = np.array([[-1, 0, 0, -0.051], [0, -1, 0, 0], [0, 0, 1,-0.162], [0, 0, 0, 1]])

        # Camera coordinates in homogeneous form
        P_c = np.array([camera_x, camera_y, camera_z, 1])

        # Convert to robot frame
        P_r = np.dot(T_cam_robot, P_c)

        return P_r[0], P_r[1], P_r[2]

    def publish_debug_image(self, image):
        header = Header(stamp=rospy.Time.now())
        img_processed_msg = Image()
        img_processed_msg.data = image.tobytes()
        img_processed_msg.encoding = 'rgb8'
        img_processed_msg.header = header
        img_processed_msg.height = image.shape[0]
        img_processed_msg.width = image.shape[1]                
        img_processed_msg.step = image.shape[1] * image.shape[2]
        self.img_debug_pub.publish(img_processed_msg)    

    def detect_landing_pad(self, image, depth):
        ## Use YOLO model to detect the landing pad
        center_x, center_y, detected_z = None, None, None
        if self.model is not None:
            result = self.model(image)
            debug_image = image.copy()
            detections = result.xyxy[0]
            for x1, y1, x2, y2, conf, cls in detections:
                # Format coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bolt_size = np.abs(y2 - y1) * np.abs(x2 - x1)

                detected_x, detected_y, detected_w, detected_h = x1, y1, x2 - x1, y2 - y1
                    
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.publish_debug_image(debug_image)

            detected_z = depth[int(center_x), int(center_y)].item()
        
        return center_x, center_y, detected_z
    
    def publish_velocity_ibvs(self, vx, vy, vz):
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/base_link_stabilized"
        vel_msg.header.stamp = rospy.get_rostime()
        vel_msg.twist.linear.x = vx
        vel_msg.twist.linear.y = vy
        vel_msg.twist.linear.z = vz

        msg = Bool()
        msg.data = True
        self.pose_ctrl_mute_pub.publish(msg)
        self.velocity_pub.publish(vel_msg)
        rospy.loginfo("velocity update: vx = " + str(vx) + ", vy = " + str(vy))

    def publish_position_pbvs(self, px, py, pz):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z
        tracking_point_mag = 0.5
        mag = np.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2)
        if mag > tracking_point_mag:
            tracking_point_x = x + (px - x)/mag * tracking_point_mag
            tracking_point_y = y + (py - y)/mag * tracking_point_mag
            tracking_point_z = z + (pz - z)/mag * tracking_point_mag/4
        else:
            tracking_point_x = px
            tracking_point_y = py
            tracking_point_z = pz

        traj = FixedTrajectory()
        traj.type = "Point"
        att1 = KeyValue()
        att1.key = "frame_id"
        att1.value = "world"
        traj.attributes.append(att1)
        att2 = KeyValue()
        att2.key = "height"
        att2.value = str(tracking_point_z)
        traj.attributes.append(att2)
        att3 = KeyValue()
        att3.key = "max_acceleration"
        att3.value = str(0.4)
        traj.attributes.append(att3)
        att4 = KeyValue()
        att4.key = "velocity"
        att4.value = str(0.1)
        traj.attributes.append(att4)
        att5 = KeyValue()
        att5.key = "x"
        att5.value = str(tracking_point_x)
        traj.attributes.append(att5)
        att6 = KeyValue()
        att6.key = "y"
        att6.value = str(tracking_point_y)
        traj.attributes.append(att6)

        self.fixed_traj_pub.publish(traj)

    def ibvs_control(self, x, y, z):
        x_gain = 0.0008
        y_gain = 0.0008
        z_gain = 0.8

        # Compute error vector in image plane
        e_u = self.cy - y
        e_v = self.cx - x
        e_z = -z
        e = np.array([[e_u], [e_v]])
        # compute interaction matrix
        L = np.array([
                        [-self.fy/z, 0, y/z],
                        [0, -self.fx/z, x/z]
                    ])
        # Compute the pseudo-inverse of the interaction matrix
        L_inv = np.linalg.pinv(L)

        lambda_gain = 2

        v_c = -lambda_gain * np.dot(L_inv, e)

        print("v_c: ", v_c)
        vx, vy, _ = v_c.flatten()


        vx_pid = (x - self.cx) * x_gain
        vy_pid = (y - self.cy) * y_gain
        vz_pid = (0.5 - z ) * z_gain
        # vx = 0
        # vy = 0
        vz = -0.1
        print(f"vx     = {vx:.2f}, vy     = {vy:.2f}, vz     = {vz:.2f}")
        print(f"vx_pid = {-vy_pid:.2f}, vy_pid = {-vx_pid:.2f}, vz_pid = {vz_pid:.2f}")

        self.publish_velocity_ibvs(vx, vy, vz_pid)

    def image_to_camera(self, image_x, image_y, depth):
        
        X_c = (image_x - self.cx) * depth / self.fx
        Y_c = (image_y - self.cy) * depth / self.fy
        Z_c = depth

        return X_c, Y_c, Z_c
    
    

    def body_to_inertial(self, body_x, body_y, body_z):
        # Convert body frame coordinates to a numpy array
        P_r = np.array([body_x, body_y, body_z, 1])  # 4x1 homogeneous coordinates
        
        # Robot position in the inertial frame
        current_position = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z
        ])

        print("x_curr: %f y_curr: %f z_curr", current_position[0], current_position[1], current_position[2])
        
        # Robot orientation as a quaternion
        current_orientation = self.odom.pose.pose.orientation
        
        # Convert quaternion to Euler angles and then to a 4x4 rotation matrix
        R_robot_inertial = euler_matrix(*tf.transformations.euler_from_quaternion([
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ]))
        
        # Create a 4x4 translation matrix from the current position
        T_translation = translation_matrix(current_position)
        
        # Combine rotation and translation into a single 4x4 transformation matrix
        T_robot_inertial = np.dot(T_translation, R_robot_inertial)

        # Transform the point from the body frame to the inertial frame
        P_i = np.dot(T_robot_inertial, P_r)

        # Return the x, y, z coordinates in the inertial frame
        return P_i[0], P_i[1], P_i[2]
    
    def pbvs_control(self, x, y, z):
        # Transform from the image plane to the camera frame
        x_c, y_c, z_c = self.image_to_camera(x, y, z)
        # Transform from the camera frame to the body frame
        x_b, y_b, z_b = self.camera_to_body(x_c, y_c, z_c)
        # Transform from the body frame to the inertial frame
        x_i, y_i, z_i = self.body_to_inertial(x_b, y_b, z_b)
        # Publish the position command for the PBVS control system
        self.publish_position_pbvs(x_i, y_i, z_i)
        


if __name__ == '__main__':
    follower = VisualServo()
    rospy.spin()