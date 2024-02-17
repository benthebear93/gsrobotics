import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Float32
import numpy as np
import cv2
from sklearn.decomposition import PCA
from cv_bridge import CvBridge
import math
from typing import Tuple
np.set_printoptions(precision=4, suppress=True)

def dilate(img, ksize=3, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

class PosePublisher2D:
    def __init__(self) -> None:
        self.debugging_flag = False
        self._dimg_l = None
        self._depth_threshold = 240
        self._pos_l = [0.0, 0.0]
        self._ori_l = 0.
        self._bridge = CvBridge()
        self._centroid = (0, 0)
        self.normal1 = self.normal2 = [0, 0]
        self.ori = [0]
        self.legnth = [0, 0]
        self._vecotr1 = [0, 0]
        self._vecotr2 = [0, 0]
        
        self._sub_raw_img = rospy.Subscriber("/left_gs/rgb/image_raw", Image, callback=self.raw_img_show, queue_size=10) 
        self._sub_pca = rospy.Subscriber("/left_gs/pca/value", Float32MultiArray, callback=self.pca_callback, queue_size=10) 
        # self._sub_pc_l = rospy.Subscriber("/gsmini_depth_img_l", Image, callback=self.depth_callback, queue_size=1)
        self._pub_object_orietation = rospy.Publisher("/sensor_object_theta", Float32, queue_size = 10)
        self._pose_data = Float32()
        
    # def depth_callback(self, data: Image) -> None:
        
    #     self._dimg_l = self._bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    #     # TODO make ksize and iteration as input of preprocess function.
    #     ksize = 20
    #     kernel = np.ones((ksize, ksize), np.uint8)
    #     self._erode_dimg_l = cv2.erode(self._dimg_l, kernel, 1)
        
    #     if self._dimg_l is None:
    #         rospy.loginfo("Depth Image waiting...")
    #     else:
    #         self.get_2d_pose(self._erode_dimg_l)
    #         if self.vecotr1[0] !=0 and self.vecotr1[1] != 0:
    #             self._pose_data.data = [self._centroid[0], self._centroid[1], -self._vecotr1[0]/abs(self._vecotr1[0]), self._vecotr1[1]/abs(self._vecotr1[1])]
    #         else:
    #             self._pose_data.data = [self._centroid[0], self._centroid[1], self._vecotr1[0], self._vecotr1[1]]
    #         self._pub_object_orietation.publish(self._pose_data)

    def pca_callback(self, data):
        print("callback2")
        self.vectors = data.data
        
    def raw_img_show(self, data: Image) -> None:
        print("call back 1")
        self._raw_img = self._bridge.imgmsg_to_cv2(data)
        self._raw_img = np.array(self._raw_img)
        
        if self._raw_img is None:
            rospy.loginfo("Image waiting...")
        elif self._raw_img.size == 0:
            rospy.logerr("Error: images is empty.")
        else:
            self.draw_pose(self._raw_img)
            cv2.imshow("_raw_img", self._raw_img)
            cv2.waitKey(1)

    # def get_2d_pose(self, dimg: np.ndarray) -> Tuple[np.ndarray, list, list]:

    #     _, thresholded_image = cv2.threshold(dimg, self._depth_threshold, 255, cv2.THRESH_BINARY)
    #     coordinates = np.column_stack(np.where(thresholded_image > self._depth_threshold))
        
    #     if coordinates.size < 100:
    #         pass
    #     else:
    #         rospy.loginfo("Coordinates found and processing can continue.")
    #         rospy.loginfo("Size : {}".format(coordinates.size))
            
    #         # Instantiate the PCA model
    #         pca = PCA(n_components=2)
    #         # Fit the model to your data
    #         pca.fit(coordinates)
    #         eig_vec = pca.components_
    #         self.legnth = np.sqrt(pca.explained_variance_)
    #         # self.lenght = pca.explained_variance_
    #         self.normal1 = eig_vec[0, :]
    #         self.normal2 = eig_vec[1, :]
    #         self.centroid = np.mean(coordinates, axis=0)
            
    #         # FIXME : Currently it is Y, X but somethings not right...
    #         self.centroid = [int(self.centroid[1]), int(self.centroid[0])] # y, x 
    #         rospy.loginfo("centroid:{}".format(self.centroid))
    #         # rospy.loginfo( "eig_vec{}".format(eig_vec))
    #         if(self.legnth[1] > self.legnth[0]):
    #             a = 40
    #             b = 20
    #         else:
    #             a = 20
    #             b = 40
            
    #         # FIXME : Drawing vector seesm little weird, Maybe something wrong with pca result? of centroid? 
    #         # Looks okay, but way they are calculated is weird
    #         self.vecotr1 = [int(self.centroid[0] + b * self.normal1[1]), int(self.centroid[1] + b * self.normal1[0])]
    #         self.vecotr2 = [int(self.centroid[0] + a * self.normal2[1]), int(self.centroid[1] + a * self.normal2[0])]
            
    #         rospy.loginfo(f"normal1 : {self.normal1}")
    #         rospy.loginfo(f"vector1 : {self.vecotr1}")
    #         rospy.loginfo(f"vector2 : {self.vecotr2}")
    #         rospy.loginfo(" ")
            
    #     return tuple(self.centroid), self.vecotr1, self.vecotr2
    
    def draw_pose(self,img):
        print("draw pose?")
        RAD2DEG = 180 / math.pi
        DEG2RAD = math.pi / 180
        mm2m = 1/1000
        arrow_lenght = 30
        goal_theta = 45
        goal_x = 0
        goal_y = 0
        if goal_theta > 90:
            goal_x = -arrow_lenght*math.cos(goal_theta*DEG2RAD)
            goal_y = arrow_lenght*math.cos(goal_theta*DEG2RAD)
        else:
            goal_x = arrow_lenght*math.cos(goal_theta*DEG2RAD)
            goal_y = arrow_lenght*math.cos(goal_theta*DEG2RAD)
        
        
        print("vectors", int(self.vectors[0]), int(self.vectors[1]), int(self.vectors[2])
              , int(self.vectors[3]), int(self.vectors[4]), int(self.vectors[5]))
        # print("goal x , goal_y", goal_x, goal_y)
        cv2.circle(img, (int(self.vectors[0]), int(self.vectors[1])), 5, (0, 0, 255), -1)  # red big X
        cv2.circle(img, (int(self.vectors[2]), int(self.vectors[3])), 5, (255, 0, 0), -1)  # blue Small X
        cv2.circle(img, (int(self.vectors[4]), int(self.vectors[5])), 5, (0, 255, 0), -1)  # green center
        
        cv2.circle(img, (int(self.vectors[4] +goal_x), int(self.vectors[5] + goal_y)), 5, (255, 255, 255), -1)  # green center
        cv2.circle(img, (int(self.vectors[4]), int(self.vectors[5])), 5, (255, 255, 255), -1)  # green center
        
        cv2.line(img, (int(self.vectors[4] + goal_x), int(self.vectors[5] + goal_y)), (int(self.vectors[4]), int(self.vectors[5])), (255, 255, 255), thickness=2)
        
        red = [int(self.vectors[0]), int(self.vectors[1])]
        blue = [int(self.vectors[2]), int(self.vectors[3])]
        center = [int(self.vectors[4]), int(self.vectors[5])]

        if blue[1] < center[1]: # theta2
            print("blus y : ", blue[1], "center y : ", center[1])
            print("theta2")
            theta = 90* DEG2RAD + math.atan2((blue[0] - red[0]), (red[1] - blue[1])) 
        else:
            print("theta1")
            print("dx", blue[0]-red[0])
            print("dy", blue[1]-red[1])
            theta = math.atan2((blue[1] - red[1]), (blue[0] - red[0]))
        print("Sensor data " , theta, "[RAD]")
        self._pose_data.data = theta*RAD2DEG
        self._pub_object_orietation.publish(self._pose_data)
        print("Sensor data " , theta*RAD2DEG, "[DEG]")
        # cv2.arrowedLine(img, pos, tuple(self.vecotr1), (0, 0, 255), 2)  
        # cv2.arrowedLine(img, pos, tuple(self.vecotr2), (255, 0, 0), 2) 

        return img

def main():
    rospy.init_node("pose_publisher_2D")
    rospy.loginfo("Pose estimator start")
    pp2d = PosePublisher2D()
    rospy.spin()

if __name__ == "__main__":
    main()
