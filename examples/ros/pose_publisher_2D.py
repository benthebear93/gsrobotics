

import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray
import numpy as np
import cv2
from sklearn.decomposition import PCA
from cv_bridge import CvBridge
import math
from typing import Tuple

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
        self.vecotr1 = [0, 0]
        self.vecotr2 = [0, 0]
        
        self._sub_raw_img = rospy.Subscriber("/gsmini_rawimg_0", Image, callback=self.raw_img_show, queue_size=10) 
        self._sub_pc_l = rospy.Subscriber("/gsmini_depth_img_l", Image, callback=self.left_depth_callback, queue_size=1)
        self._pub_2d_pose = rospy.Publisher("/left_gs_pose", Float64MultiArray, queue_size = 10)
        self.pose_data = Float64MultiArray()
        
    def left_depth_callback(self, data: Image) -> None:
        
        self._dimg_l = self._bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.erode_dimg_l = erode(self._dimg_l, 20)
        if self._dimg_l is None:
            rospy.loginfo("Depth Image waiting...")
        else:
            # self.dilate_dimg_l = dilate(self.erode_dimg_l, 10)
            self.get_2d_pose(self.erode_dimg_l)
            
            magnitude1 = self.vector_magnitude(self.vecotr1)
            magnitude2 = self.vector_magnitude(self.vecotr2)

            if magnitude1 > magnitude2:
                larger_vector = self.vecotr1
            else:
                larger_vector = self.vecotr2
                
                self.pose_data.data = larger_vector
                self._pub_2d_pose.publish(self.pose_data)

    def raw_img_show(self, data: Image) -> None:
        self._raw_img = self._bridge.imgmsg_to_cv2(data)
        self._raw_img = np.array(self._raw_img)
        # self._raw_img_to_gray_scale = cv2.cvtColor(self._raw_img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(self._raw_img_to_gray_scale, 70, 255, 0)
        
        if self._raw_img is None:
            rospy.loginfo("Image waiting...")
        elif self._raw_img.size == 0:
            rospy.logerr("Error: images is empty.")
        else:
            # rospy.loginfo(f"raw img : {self._raw_img.shape}"
            self.draw_pose(self._raw_img, self.centroid)
            cv2.imshow("_raw_img", self._raw_img)
            cv2.imshow("thres", self._dimg_l)
            cv2.imshow("erode_dimg_l", self.erode_dimg_l)
            cv2.waitKey(1)

    def get_2d_pose(self, dimg: np.ndarray) -> Tuple[np.ndarray, list, list]:

        _, thresholded_image = cv2.threshold(dimg, self._depth_threshold, 255, cv2.THRESH_BINARY)
        self.image = thresholded_image
        # # Find the coordinates of pixels below the threshold
        coordinates = np.column_stack(np.where(thresholded_image > self._depth_threshold))
        if coordinates.size < 100:
            pass
        else:
            rospy.loginfo("Coordinates found and processing can continue.")
            rospy.loginfo("Size : {}".format(coordinates.size))
            # Instantiate the PCA model
            pca = PCA(n_components=2)
            # Fit the model to your data
            pca.fit(coordinates)
            eig_vec = pca.components_
            self.legnth = np.sqrt(pca.explained_variance_)
            # self.lenght = pca.explained_variance_
            self.normal1 = eig_vec[0, :]
            self.normal2 = eig_vec[1, :]
            self.centroid = np.mean(coordinates, axis=0)
            self.centroid = [int(self.centroid[1]), int(self.centroid[0])]
            rospy.loginfo("centroid:{}".format(self.centroid))
            # rospy.loginfo( "eig_vec{}".format(eig_vec))
            
            rospy.loginfo(f"normal1 : {self.normal1}")
            rospy.loginfo(f"normal2 : {self.normal2}")
            rospy.loginfo("==========================")
            if(self.legnth[1] > self.legnth[0]):
                a = 40
                b = 20
            else:
                a = 20
                b = 40
            
            self.vecotr1 = [int(self.centroid[0] + b * self.normal1[1]), int(self.centroid[1] + b * self.normal1[0])]
            self.vecotr2 = [int(self.centroid[0] + a * self.normal2[1]), int(self.centroid[1] + a * self.normal2[0])]
            
        return tuple(self.centroid), self.vecotr1, self.vecotr2

    def vector_magnitude(self, vector):
        return math.sqrt(vector[0]**2 + vector[1]**2)
    
    def draw_pose(self,img, pos):
        cv2.circle(img, pos, 5, (0, 255, 0), -1)  # Green color, thickness=-1 to fill the circle
        cv2.arrowedLine(img, pos, tuple(self.vecotr1), (0, 0, 255), 2)  # Red color, thickness=2
        cv2.arrowedLine(img, pos, tuple(self.vecotr2), (255, 0, 0), 2)  # Blue color, thickness=2

        return img

def main():
    rospy.init_node("pose_publisher_2D")
    rospy.loginfo("Pose estimator start")
    pp2d = PosePublisher2D()
    rospy.spin()

if __name__ == "__main__":
    main()
