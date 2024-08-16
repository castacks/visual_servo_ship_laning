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

class BoltHeadFollower:
    def __init__(self):
        self.debug = True
        self.last_center = None
        self.last_size = None
        self.max_center_jump = 50  # maximum allowed jump in center position
        self.max_size_jump = 200   # maximum allowed change in size

        # # Load the YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/mohammad/workspace/16-667/src/visual_servo_landing/model/yolov5_landing_pad/best.pt')
        # self.confidence_threshold = 0.5

        # Camera parameters and control constants
        self.fx = 410.9  # Focal length in x
        self.fy = 410.9  # Focal length in y
        self.cx = 640.0  # Optical center x
        self.cy = 540.0  # Optical center y
        self.Kp = np.identity(4)*0.5    # Proportional control gain
        self.Kp[0, 0] = 2
        self.Kp[1, 1] = 2
        self.Kp[2, 2] = 0.5
        self.Kp[3, 3] = 0.1

        self.Ki = np.identity(4)*0.2    # Proportional control gain
        self.Kd = np.identity(4)*0.01    # Proportional control gain

        self.error_integral = 0
        self.prev_error = 0
        self.Zt = 1.2    # Desired depth
        self.desired_size = 12000  # Desired size of the bolt head in the image

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
        self.tracking_point_pub = rospy.Publisher("tracking_point2", Odometry, queue_size=100)
        
        self.output_dir = '/home/dream_reaper/Documents/16-667/yolo_training/train'
        self.img_num = 0
        self.odom = Odometry()
        self.got_odom = False
        self.image_center = None
        self.frame_number = 3973
        self.save_dataset = False
        self.tracked_circle = None
        
        self.mute_pose_controller = False

    def odom_callback(self, odom):
        self.odom = odom
        self.got_odom = True
        
    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert the ROS image messages to OpenCV images
            rgb = CvBridge().imgmsg_to_cv2(rgb_msg, "bgr8")
            self.img_num = self.img_num + 1
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
        except Exception as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        image_center = (rgb.shape[1] // 2, rgb.shape[0] // 2 + 400)  # tip of the end-effector
        self.image_center = image_center
        # bolt_center = self.find_bolt_head(rgb, True)

        # landing_center_x, landing_center_y, landing_center_r = self.circle_detector(rgb)
        self.detect_skewed_rectangles(rgb)
        # self.find_circle_and_draw_bounding_box(rgb)

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
    def detect_skewed_rectangles(self, image):
        ## Use YOLO model to detect the landing pad
        detection_found = False

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
                # print ("area = ", detected_w*detected_h)
                # if detected_w*detected_h < 1000000/(self.odom.pose.pose.position.z*self.odom.pose.pose.position.z*self.odom.pose.pose.position.z):
                #     detection_found = True
                #     # Draw rectangle and circle
                    
                #     # cv2.circle(debug_image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                    
 
                #     self.frame_number += 1
                #     norm_x, norm_y, norm_w, norm_h = center_x / debug_image.shape[1], center_y / debug_image.shape[0], detected_w / debug_image.shape[1], detected_w / debug_image.shape[0]
                #     annotation = f"0 {norm_x} {norm_y} {norm_w} {norm_h}"

                #     # Save the annotated image and annotation
                #     image_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.jpg"
                #     image_filename_annotated = f"{self.output_dir}/annotated_{self.frame_number:03d}.jpg"
                #     annotation_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.txt"

                    
                #     cv2.imwrite(image_filename, image)
                #     cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.imwrite(image_filename_annotated, debug_image)
                #     with open(annotation_filename, 'w') as file:
                #         file.write(annotation + '\n')
                    
                #     self.publish_debug_image(debug_image)
        
        return image
    
    def find_circle_and_draw_bounding_box(self, image):
        alpha = 0.3
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, 100, 200)

        # Use HoughCircles to detect circles with stricter parameters
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_circle = circles[0][0]  # Take the first detected circle

            # Smooth the detected circle with a moving average
            if self.tracked_circle is None:
                self.tracked_circle = detected_circle
            else:
                self.tracked_circle = alpha * detected_circle + (1 - alpha) * self.tracked_circle

            # Get the smoothed center coordinates and radius of the circle
            center_x, center_y, radius = int(self.tracked_circle[0]), int(self.tracked_circle[1]), int(self.tracked_circle[2])
            
            # Calculate the bounding box coordinates
            x = int(center_x - radius)
            y = int(center_y - radius)
            w = int(2 * radius)
            h = int(2 * radius)
                

            # Draw the bounding box around the circle
            
            
            self.frame_number += 1
            norm_x, norm_y, norm_w, norm_h = center_x / image.shape[1], center_y / image.shape[0], w / image.shape[1], w / image.shape[0]
            annotation = f"0 {norm_x} {norm_y} {norm_w} {norm_h}"

            # Save the annotated image and annotation
            image_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.jpg"
            image_filename_annotated = f"{self.output_dir}/annotated_{self.frame_number:03d}.jpg"
            annotation_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.txt"

            
            cv2.imwrite(image_filename, image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(image_filename_annotated, image)
            with open(annotation_filename, 'w') as file:
                file.write(annotation + '\n')
            
            self.publish_debug_image(image)
    def simplest_pid_control(self, center_x, center_y):
        x_gain = 0.0008
        y_gain = 0.0008

        vx = (center_x - self.image_center[0]) * x_gain
        vy = (center_y - self.image_center[1]) * y_gain
        vz = 0
        yaw_rate = 0
        
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/camera"
        vel_msg.header.stamp = rospy.get_rostime()
        vel_msg.twist.linear.x = vz
        vel_msg.twist.linear.y = -vx
        vel_msg.twist.linear.z = vy
        vel_msg.twist.angular.z = yaw_rate

        msg = Bool()
        msg.data = True
        self.pose_ctrl_mute_pub.publish(msg)
        self.velocity_pub.publish(vel_msg)
        rospy.loginfo("velocity update: vx = " + str(vx) + ", vy = " + str(vy))


if __name__ == '__main__':
    follower = BoltHeadFollower()
    rospy.spin()