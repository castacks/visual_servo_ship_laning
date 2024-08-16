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
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/dream_reaper/Airlab/champ_noetic_demo/src/visual_servo_landing/model/yolov5_landing_pad/best.pt')
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
        
        self.output_dir = '/home/dream_reaper/Documents/16-667/yolo_training/test'
        self.img_num = 0
        self.odom = Odometry()
        self.got_odom = False
        self.image_center = None
        self.frame_number = 0
        self.save_dataset = False

        
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
        self.find_smallest_rectangle(rgb)

        # ## Use YOLO model to detect the landing pad
        # if self.model is not None:
        #     result = self.model(rgb)
        #     debug_image = rgb
        #     detections = result.xyxy[0]
        #     for x1, y1, x2, y2, conf, cls in detections:
        #         # Format coordinates
        #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #         center_x = (x1 + x2) / 2
        #         center_y = (y1 + y2) / 2
        #         bolt_size = np.abs(y2 - y1) * np.abs(x2 - x1)

        #         detected_x, detected_y, detected_w, detected_h = x1, y1, x2 - x1, y2 - y1
        #         detection_found = True

        #         # Draw rectangle and circle
        #         cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.circle(debug_image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

        #         header = Header(stamp=rospy.Time.now())
        #         img_processed_msg = Image()
        #         img_processed_msg.data = debug_image.tobytes()
        #         img_processed_msg.encoding = 'rgb8'
        #         img_processed_msg.header = header
        #         img_processed_msg.height = debug_image.shape[0]
        #         img_processed_msg.width = debug_image.shape[1]                
        #         img_processed_msg.step = debug_image.shape[1] * debug_image.shape[2]
        #         self.img_debug_pub.publish(img_processed_msg)
                


            # cv2.circle(debug_image, (image_center[0], image_center[1]), 5, (255, 0, 0), -1)
        # Initialize params with a default value
        center_x, center_y = image_center
        bolt_size = 0
        detection_found = False
        # if landing_center_x is not None:
        #     detection_found = True
        

        # # Calculate visual error
        # visual_error = np.array([center_x - image_center[0], 
        #                         center_y - image_center[1], 
        #                         self.desired_size - bolt_size])  # Include size error
        # # Calculate depth at the end-effector's position
        # Z_cur = depth[int(image_center[1]), int(image_center[0]), 0]

        # if detection_found:
        #     # Calculate control signal
        #     bounding_box = (landing_center_x, landing_center_y, landing_center_r, 0)
        #     end_effector = (image_center[0], image_center[1], Z_cur, 1)
        #     # control_signal = self.calculate_control_signal(bounding_box, depth, end_effector)
        #     # control_signal = self.simple_pid_control(end_effector, bounding_box, depth)
        #     self.simplest_pid_control(landing_center_x, landing_center_y)
        
        

        
            # if control_signal:
            #     rospy.loginfo("control signal published")
            #     self.velocity_pub.publish(control_signal)
            # else:
            #     rospy.loginfo("No control signal calculated.")
            #     print(control_signal)
    
    
    def circle_detector(self, cv_image):
        orig = cv_image
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise and improve detection accuracy
        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=30, minRadius=20, maxRadius=100)
        center_x, center_y = None, None
        # If some circles are detected, proceed to draw them
        biggest_radius = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if i[2] > biggest_radius:
                    biggest_radius = i[2]
                    center_x, center_y = i[0], i[1]
        
        # Draw the outer circle
        # cv2.circle(cv_image, (center_x, center_y), biggest_radius, (0, 255, 0), 2)
        # Draw the center of the circle
        # cv2.circle(cv_image, (center_x, center_y), 2, (0, 0, 255), 3)

        # Find contours in the original image to detect the rectangle
        edged = cv2.Canny(gray_blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            print("rect: ", len(approx))
            
            # Check if the contour is a rectangle (4 sides)
            if len(approx) == 4:
                print('rectangle')
                
                # Use cv2.minAreaRect() to get the rotated rectangle
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Calculate the center of the rotated rectangle
                center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
                print('diff = ', np.abs(center_x - center_x))
                
                # Check if the center of the rectangle is near the expected center
                if np.abs(center_x - center_x) < 80 and np.abs(center_y - center_y) < 80:
                    # Draw the rotated rectangle
                    cv_image = cv2.drawContours(cv_image, [box], 0, (255, 0, 0), 2)
                    detected = True

                    # Convert the processed OpenCV image back to ROS Image message
                    header = Header(stamp=rospy.Time.now())
                    img_processed_msg = Image()
                    img_processed_msg.data = cv_image.tobytes()
                    img_processed_msg.encoding = 'rgb8'
                    img_processed_msg.header = header
                    img_processed_msg.height = cv_image.shape[0]
                    img_processed_msg.width = cv_image.shape[1]                
                    img_processed_msg.step = cv_image.shape[1] * cv_image.shape[2]
                    self.img_debug_pub.publish(img_processed_msg)

        
        # if detected:
        #     self.frame_number += 1
        #     center_x, center_y = x + w/2, y + h/2
        #     norm_x, norm_y, norm_w, norm_h = center_x / cv_image.shape[1], center_y / cv_image.shape[0], w / cv_image.shape[1], h / cv_image.shape[0]
        #     annotation = f"0 {norm_x} {norm_y} {norm_w} {norm_h}"

        #     # Save the annotated image and annotation
        #     image_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.jpg"
        #     image_filename_annotated = f"{self.output_dir}/annotated_{self.frame_number:03d}.jpg"
        #     annotation_filename = f"{self.output_dir}/frame_{self.frame_number:03d}.txt"

        #     cv2.imwrite(image_filename_annotated, cv_image)
        #     cv2.imwrite(image_filename, orig)
        #     with open(annotation_filename, 'w') as file:
        #         file.write(annotation + '\n')


        return center_x, center_y, biggest_radius

    def find_smallest_rectangle(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (optional)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection to find edges in the image
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edged image
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize variables to store the smallest rectangle
        min_area = float('inf')
        min_rect = None
        
        # Loop over all contours
        for contour in contours:
            # Find the minimum area bounding rectangle for the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate the area of the rectangle
            width = rect[1][0]
            height = rect[1][1]
            area = width * height
            
            # If this rectangle has a smaller area than the previous ones, update min_rect
            if area < min_area:
                min_area = area
                min_rect = box
        
        # If a rectangle was found, draw it on the image
        if min_rect is not None:
            cv2.drawContours(image, [min_rect], 0, (0, 255, 0), 2)
            # Convert the processed OpenCV image back to ROS Image message
            header = Header(stamp=rospy.Time.now())
            img_processed_msg = Image()
            img_processed_msg.data = image.tobytes()
            img_processed_msg.encoding = 'rgb8'
            img_processed_msg.header = header
            img_processed_msg.height = image.shape[0]
            img_processed_msg.width = image.shape[1]                
            img_processed_msg.step = image.shape[1] * image.shape[2]
            self.img_debug_pub.publish(img_processed_msg)                       
        
        # Return the image with the smallest rectangle drawn
        return image, min_rect
    def find_bolt_head_and_save(self, rgb, frame_number):
        image = rgb.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_center = np.array([image.shape[1]/2, image.shape[0]/2])  # (center_x, center_y)
        top_15_percent = image.shape[0] * 0.15
        closest_contour = None
        min_distance = float('inf')

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 6:
                x, y, w, h = cv2.boundingRect(approx)
                center_x, center_y = x + w/2, y + h/2

                # Reject if in the top 15% of the image
                if center_y < top_15_percent:
                    continue

                distance = np.linalg.norm(np.array([center_x, center_y]) - image_center)

                if distance < min_distance:
                    min_distance = distance
                    closest_contour = approx

        if closest_contour is not None:
            x, y, w, h = cv2.boundingRect(closest_contour)
            center_x, center_y = x + w/2, y + h/2
            norm_x, norm_y, norm_w, norm_h = center_x / image.shape[1], center_y / image.shape[0], w / image.shape[1], h / image.shape[0]
            annotation = f"0 {norm_x} {norm_y} {norm_w} {norm_h}"

            if self.debug:
                cv2.drawContours(image, [closest_contour], -1, (0, 255, 0), 2)
                cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

            # Save the annotated image and annotation
            image_filename = f"{self.output_dir}/frame_{frame_number:03d}.jpg"
            image_filename_annotated = f"{self.output_dir}/annotated_{frame_number:03d}.jpg"
            annotation_filename = f"{self.output_dir}/frame_{frame_number:03d}.txt"

            cv2.imwrite(image_filename_annotated, image)
            cv2.imwrite(image_filename, rgb)
            with open(annotation_filename, 'w') as file:
                file.write(annotation + '\n')

            return annotation

        return None
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
    def simple_pid_control(self, end_effector, target, depth):
        x, y, w, h = target
        x_ee, y_ee, depth_ee, dummy = end_effector
        depth_target = depth[int((y + h) / 2), int((x + w) / 2)][0]
        x_center = x + w / 2
        y_center = y + h / 2

        dr = depth[int(y + h / 2), int(x + w)]
        dl = depth[int(y + h / 2), int(x)]
        d = (dr - dl)[0]

        x_gain = 0.004
        y_gain = 0.005
        z_gain = 20
        yaw_gain = 1

        print("error x = ", (x_ee - x_center))
        print("error y = ", (y_ee - y_center))
        vx = (x_ee - x_center) * x_gain
        vy = (y_ee - y_center) * y_gain
        vz = (depth_target - depth_ee) * z_gain
        # yaw_rate = d * yaw_gain
        yaw_rate = 0

        # max_actuation = 0.1
        # if (vx > max_actuation):
        #     vx = max_actuation
        # if (vx < -max_actuation):
        #     vx = -max_actuation

        # if (vy > max_actuation):
        #     vy = max_actuation
        # if (vy < -max_actuation):
        #     vy = -max_actuation

        # if (vz > 10*max_actuation):
        #     vz = 10*max_actuation
        # if (vz < -10*max_actuation):
        #     vz = -10*max_actuation
        
        
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/camera"
        vel_msg.header.stamp = rospy.get_rostime()
        # vel_msg.twist.linear.x = -vx
        vel_msg.twist.linear.x = -vz
        # vel_msg.twist.linear.x = -0.1
        vel_msg.twist.linear.y = -vy
        # vel_msg.twist.linear.z = vz
        vel_msg.twist.linear.z = vx
        # vel_msg.twist.linear.z = 0.1
        vel_msg.twist.angular.z = yaw_rate
        # vel_msg.twist.angular.z = 0

        print("vx = " ,vel_msg.twist.linear.x)
        print("vy = " ,vel_msg.twist.linear.y)
        print("vz = " ,vel_msg.twist.linear.z)
        print("yaw_rate = " ,vel_msg.twist.angular.z)

        # if not self.mute_pose_controller:
        #     self.mute_pose_controller = True
        msg = Bool()
        msg.data = True
        self.pose_ctrl_mute_pub.publish(msg)
        self.velocity_pub.publish(vel_msg)

    def calculate_control_signal(self, bounding_box, depth, target):
        """
        Calculate the control signal for the drone based on visual servoing.

        Parameters:
        - bounding_box: Tuple of (x, y, width, height) for the detected object.
        - depth: Depth array.
        - target: Tuple of target properties (tx, ty, tw, th).

        Returns:
        - TwistStamped message representing the velocity control signal.
        """

        x, y, w, h = bounding_box
        tx, ty, tw, th = target

        print("SIZE = ", w*h)

        # Calculate depth difference and current depth at the center of the object
        dr = depth[int(y + h / 2), int(x + w)]
        dl = depth[int(y + h / 2), int(x)]
        d = (dr - dl)[0]
        Z_cur = tw
        Zt = depth[int((y + h) / 2), int((x + w) / 2)][0]

        # Center coordinates of the bounding box
        x_center = x + w / 2
        y_center = y + h / 2

        # Convert to image plane coordinates
        x = (x_center - self.cx) / self.fx
        y = (y_center - self.cy) / self.fy
        w = w / self.fx
        h = h / self.fy

        print("Zt = ", Zt)
        visual_measure = np.array([[x, y, Zt]]).transpose()
        Lc = np.array([[-1/Zt,       0,             x/Zt,       y ],
                       [0,          -1/Zt,          y/Zt,       -x],  
                       [0,           0,             -1,         0 ]  ])
        Lc_inv = np.linalg.pinv(Lc)

        # Convert target to image plane coordinates
        target_depth = depth[int(ty + th/2), int(tx + tw/2)][0]
        tx = (tx - self.cx) / self.fx
        ty = (ty - self.cy) / self.fy
        tw = tw / self.fx
        th = th / self.fy
        
        visual_target = np.array([[tx, ty, target_depth]]).transpose()

        # Control gains and error calculation
        error = visual_target - visual_measure
        self.error_integral += error  # Integral of the error
        error_derivative = error - self.prev_error  # Derivative of the error
        self.prev_error = error  # Update previous error

        # Adjust the error computation with sqrt and sign
        error = np.sign(error) * np.sqrt(np.abs(error))
        integral_error = np.sign(self.error_integral) * np.sqrt(np.abs(self.error_integral))
        derivative_error = np.sign(error_derivative) * np.sqrt(np.abs(error_derivative))

        # Control gains and error calculation
        error = visual_target - visual_measure
        error = np.sign(error) * np.sqrt(np.abs(error))
        # P Control
        # vel_control = self.Kp @ Lc_inv @ error
        # PID Control
        vel_control = self.Kp @ Lc_inv @ error + self.Ki @ Lc_inv @ integral_error + self.Kd @ Lc_inv @ derivative_error

        # Depth and yaw rate control
        # vz = np.array([[0, 0, (self.desired_size - (w*h)) * 0.5, 0]]).transpose()
        yaw_rate = np.array([[0, 0, 0, (d) * 50.0]]).transpose()
        # vel_control += vz + yaw_rate
        vel_control += yaw_rate

        # rospy.loginfo_throttle(0.1, f"error:\n{error.transpose()}{d}")
        # rospy.loginfo_throttle(0.1, f"vel_control:\n{vel_control.transpose()}")

        # Creating the TwistStamped message
        vx, vy, vz, yaw_rate = vel_control[:, 0]
        max_actuation = 0.5
        if (vx > max_actuation):
            vx = max_actuation
        if (vx < -max_actuation):
            vx = -max_actuation

        if (vy > max_actuation):
            vy = max_actuation
        if (vy < -max_actuation):
            vy = -max_actuation

        if (vz > max_actuation):
            vz = max_actuation
        if (vz < -max_actuation):
            vz = -max_actuation
        # yaw_rate = 0
        # vz = 0
        print("vx2 = ", vx)
        print("vy2 = ", vy)
        print("vz2 = ", vz)
        print("yaw_rate2 = ", yaw_rate)
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/camera"
        vel_msg.header.stamp = rospy.get_rostime()
        vel_msg.twist.linear.x = vx
        vel_msg.twist.linear.y = vy
        # vel_msg.twist.linear.z = vz
        vel_msg.twist.linear.z = 0.0
        # vel_msg.twist.angular.z = yaw_rate
        vel_msg.twist.angular.z = 0
    
        msg = Bool()
        msg.data = True
        # self.pose_ctrl_mute_pub.publish(msg) 
        # self.velocity_pub.publish(vel_msg)
        tracking_point = Odometry()
        tracking_point.header.stamp = rospy.Time.now()
        tracking_point.header.frame_id = 'map'
        tracking_point.child_frame_id = 'map'
        tracking_point.pose.pose.position.x = self.odom.pose.pose.position.x - 0.3*vy 
        tracking_point.pose.pose.position.y = self.odom.pose.pose.position.y - 0.3*vz 
        tracking_point.pose.pose.position.z = self.odom.pose.pose.position.z + 0.3*vx
        tracking_point.pose.pose.orientation.w = 1
        self.tracking_point_pub.publish(tracking_point)


if __name__ == '__main__':
    follower = BoltHeadFollower()
    rospy.spin()