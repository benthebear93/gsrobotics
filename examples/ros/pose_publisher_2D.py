

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

class PosePublisher2D:
    def __init__(self) -> None:
        self._dimg_l = None
        # self._dimg_r = None
        self._depth_threshold = 240
        self._pos_l = [0.0, 0.0]
        self._ori_l = 0.0
        self._bridge = CvBridge()

        self._sub_pc_l = rospy.Subscriber("/gsmini_depth_img_l", Image, callback=self.sub_cb_l, queue_size=1)
        # self._sub_pc_r = rospy.Subscriber("/gsmini_depth_img_r", Image, callback=self.sub_cb_r, queue_size=1)s
        self._sub_2d_pose_pub = rospy.Subscriber("/gsmini_depth_img_l", Image, callback=self.pub_cb, queue_size=1)
        
        self._pub_2d_l = rospy.Publisher("/gsmini_2d_pose", Pose2D, queue_size=1)
        # self._pub_2d_r = rospy.Publisher("/gsmini_2d_pose", Pose2D)

    def sub_cb_l(self, data: Image) -> None:
        if data is not None:
            self._dimg_l = self._bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            _, self._dimg_l = cv2.threshold(self._dimg_l, self._depth_threshold, 255, cv2.THRESH_BINARY)

    # def sub_cb_r(self, data: Image) -> None: 
    #     self._dimg_r = self._bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def _got_img(self):
        return True if self._dimg_l is not None else False

    def _in_contact(self):
        coordinates = np.column_stack(np.where(self._dimg_l > self._depth_threshold))
        return True if len(coordinates) != 0 else False

    def pub_cb(self, data: Image) -> None:

        if not self._got_img():
            return

        if not self._in_contact():
            return

        # if  self._dimg_r is None and self._dimg_r is None:
        if self._dimg_l.flags.writeable is False:
            self._dimg_l = np.copy(self._dimg_l)

        # test = self.get_2d_pose(self._dimg_l)
        pos, ori = self.get_2d_pose(self._dimg_l)

        pose2d = Pose2D()
        pose2d.theta = ori
        pose2d.x, pose2d.y = pos[0], pos[1]
        self._pub_2d_l.publish(pose2d)

        # cv2.imwrite("test_0.png", self._dimg_l)
        # self._dimg_l = cv2.cvtColor(self._dimg_l, cv2.COLOR_GRAY2BGR)
        # self._dimg_l = self.draw_pose(self._dimg_l, pos, ori)
        # cv2.waitKey(100)
        # cv2.imshow("test", self._dimg_l)
        # cv2.imwrite("test_1.png", self._dimg_l)

    def get_2d_pose(self, dimg: np.ndarray) -> Tuple[np.ndarray, float]:

        _, thresholded_image = cv2.threshold(dimg, self._depth_threshold, 255, cv2.THRESH_BINARY)

        # Find the coordinates of pixels below the threshold
        coordinates = np.column_stack(np.where(thresholded_image > self._depth_threshold))

        pos = [ int(np.mean(coordinates[:,0])), int(np.mean(coordinates[:,1])) ]
        # Instantiate the PCA model
        pca = PCA(n_components=2)
    
        # Fit the model to your data
        pca.fit(coordinates)
    
        arrow_length = 40

        # compute position and orientation
        first_component = pca.components_[0]
        d = [int(pos[0] + first_component[0]), int(pos[1] + first_component[1])]
        x_axis = (
            int(pos[0] + arrow_length ),
            int(pos[1] )
        )
        ori = -angle(v1 = x_axis, v2 = d)
        # print(pos, x_axis, d, first_component,ori)

        return tuple(pos), ori

    # def draw_pose(self,img, pos, ori):
    #     cv2.circle(img, pos, 5, (0, 255, 0), -1)  # Green color, thickness=-1 to fill the circle

    #     # Calculate the arrow end point based on orientation
    #     arrow_length = 40
    #     # -52.4114929 deg
    #     # 1.57
    #     arrow_angle = -ori  # Reverse the angle for counter-clockwise direction
    #     # arrow_angle = -0.785398163  # Reverse the angle for counter-clockwise direction
    #     # arrow_angle = -0.914753116278168  # Reverse the angle for counter-clockwise direction
    #     pc = (
    #         int(pos[0] + arrow_length * np.cos(arrow_angle)),
    #         int(pos[1] + arrow_length * np.sin(arrow_angle))
    #     )

    #     x_axis = (
    #         int(pos[0] + arrow_length ),
    #         int(pos[1] )
    #     )

    #     # Draw the arrow lines
    #     cv2.arrowedLine(img, pos, pc, (0, 0, 255), 2)  # Red color, thickness=2
    #     cv2.arrowedLine(img, pos, x_axis, (255, 0, 0), 2)  # Blue color, thickness=2

    #     return img

def main():
    rospy.init_node("pose_publisher_2D")
    pp2d = PosePublisher2D()
    rospy.spin()

if __name__ == "__main__":
    main()
