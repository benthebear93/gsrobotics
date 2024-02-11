

import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import Pose2D
import numpy as np
import cv2
from sklearn.decomposition import PCA
from cv_bridge import CvBridge
import math
from typing import Tuple


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def rotate_vector(vector, angle_rad):
    # Rotate a 2D vector counter-clockwise by the given angle
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(rotation_matrix, vector)

def erode(img, ksize=10):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def dilate(img, ksize=3, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

class PosePublisher2D:
    def __init__(self) -> None:
        self.debugging_flag = False
        self._dimg_l = None
        self._depth_threshold = 240
        self._pos_l = [0.0, 0.0]
        self._ori_l = 0.0
        self._bridge = CvBridge()
        self.centroid = (0, 0)
        self.normal1 = self.normal2 = [0, 0]
        self.ori = [0]
        self.legnth = [0, 0]
        self._sub_raw_img = rospy.Subscriber("/gsmini_rawimg_0", Image, callback=self.raw_img_show, queue_size=10) 
        self._sub_pc_l = rospy.Subscriber("/gsmini_depth_img_l", Image, callback=self.left_depth_callback, queue_size=1)
        self._pub_2d_l = rospy.Publisher("/gsmini_2d_pose", Pose2D, queue_size=1)

    def left_depth_callback(self, data: Image) -> None:
        if data is not None:
            self._dimg_l = self._bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
        self.centroid, self.normal1, self.normal2 = self.get_2d_pose(self.erode_dimg_l)

        self.pose2d = Pose2D()
        self.pose2d.theta = 0
        self.pose2d.x, self.pose2d.y = self.centroid[0], self.centroid[1]
        self._pub_2d_l.publish(self.pose2d)

    def raw_img_show(self, data: Image) -> None:
        self._raw_img = self._bridge.imgmsg_to_cv2(data)
        self._raw_img = np.array(self._raw_img)
        self._raw_img_to_gray_scale = cv2.cvtColor(self._raw_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(self._raw_img_to_gray_scale, 70, 255, 0)

        cv2.circle(self._raw_img, self.centroid, 5, (0, 255, 0)) 
        if self.debugging_flag == True:
            if self._raw_img is None or self._dimg_l is None:
                print("Image waiting...")
            elif self._raw_img.size == 0 or self._dimg_l.size == 0:
                print("Error: One of the images is empty.")
            else:
                print(f"raw img : {self._raw_img.shape} thres img : {self._dimg_l.shape}")
        else:
            pass
        self.erode_dimg_l = erode(self._dimg_l, 20)
        self.dilate_dimg_l = dilate(self.erode_dimg_l, 10)

        self.draw_pose(self._raw_img, self.centroid)
        cv2.imshow("_raw_img", self._raw_img)
        cv2.imshow("thres", self._dimg_l)
        cv2.imshow("erode_dimg_l", self.erode_dimg_l)
        cv2.waitKey(1)

    def get_2d_pose(self, dimg: np.ndarray) -> Tuple[np.ndarray, float]:

        _, thresholded_image = cv2.threshold(dimg, self._depth_threshold, 255, cv2.THRESH_BINARY)
        self.image = thresholded_image
        # # Find the coordinates of pixels below the threshold
        coordinates = np.column_stack(np.where(thresholded_image > self._depth_threshold))
        if coordinates.size < 10:
            # print("Error: No coordinates found. The condition was not met by any element.")
            centroid = [0, 0]
        else:
            print("Coordinates found and processing can continue.")
            print("Size : ", coordinates.size)
            # Instantiate the PCA model
            pca = PCA(n_components=2)
            # Fit the model to your data
            pca.fit(coordinates)
            eig_vec = pca.components_
            self.legnth = np.sqrt(pca.explained_variance_)
            # self.lenght = pca.explained_variance_
            self.normal1 = eig_vec[0, :]
            self.normal2 = eig_vec[1, :]
            centroid = np.mean(coordinates, axis=0)
            centroid = [int(centroid[1]), int(centroid[0])]
            print("centroid:", centroid)
            print("eig_vec", eig_vec)
            
            print(f"normal1 : {self.normal1}")
            print(f"normal2 : {self.normal2}")
        return tuple(centroid), self.normal1, self.normal2

    def draw_pose(self,img, pos):
        cv2.circle(img, pos, 5, (0, 255, 0), -1)  # Green color, thickness=-1 to fill the circle

        # Calculate the arrow end point based on orientation
        arrow_length = 2
        if(self.legnth[1] > self.legnth[0]):
            a = 40
            b = 20
        else:
            a = 20
            b = 40
        
        first_component = (
            int(pos[0] + b * self.normal1[1]),
            int(pos[1] + b * self.normal1[0])
        )

        second_component = (
            int(pos[0] + a * self.normal2[1]),
            int(pos[1] + a * self.normal2[0])
        )
        # Draw the arrow lines
        cv2.arrowedLine(img, pos, first_component, (0, 0, 255), 2)  # Red color, thickness=2
        cv2.arrowedLine(img, pos, second_component, (255, 0, 0), 2)  # Blue color, thickness=2

        return img

def main():
    rospy.init_node("pose_publisher_2D")
    rospy.loginfo("Pose estimator start")
    pp2d = PosePublisher2D()
    rospy.spin()

if __name__ == "__main__":
    main()
