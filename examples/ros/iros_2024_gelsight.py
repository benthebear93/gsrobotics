import sys
import numpy as np
import cv2
import os
import open3d
import copy
# from gelsight import gsdevice
# from gelsight import gs3drecon
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import Image

import gsdevice
import gs3drecon
import ros_numpy

from threading import Thread, Lock
from cv_bridge import CvBridge, CvBridgeError

def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5

class WebcamVideoStream :
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

def main(argv):

    rospy.init_node('showmini3dros', anonymous=True)

    #2d image

    NUM_SENSORS = 1

    gs = {}
    gs['img'] = [0] * 2
    gs['gsmini_pub'] = [0] * 2
    gs['vs'] = [0] * 2
    gs['img_msg'] = [0] * 2

    # Set flags
    SAVE_VIDEO_FLAG = True
    GPU = False
    MASK_MARKERS_FLAG = True
    USE_ROI = False
    PUBLISH_ROS_PC = True
    SHOW_3D_NOW = False
    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.

    # the device ID can change after chaning the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")
    net_file_path = '../nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()

    #for i in range(NUM_SENSORS):
    #    gs['gsmini_pub'][i] = rospy.Publisher("/gsmini_rawimg_{}".format(i), Image, queue_size=1)
    #    gs['vs'][i] = WebcamVideoStream(src=2*i).start()  # make sure the id numbers of the cameras are ones recognized by the computer. Default, 2 and 4


    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1],f0.shape[0]), isColor=True)

    if PUBLISH_ROS_PC:
        ''' ros point cloud initialization '''
        x = np.arange(dev.imgh) * mpp
        y = np.arange(dev.imgw) * mpp
        X, Y = np.meshgrid(x, y)
        points = np.zeros([dev.imgw * dev.imgh, 3])
        points[:, 0] = np.ndarray.flatten(X)
        points[:, 1] = np.ndarray.flatten(Y)
        Z = np.zeros((dev.imgh, dev.imgw))  # initialize points array with zero depth values
        points[:, 2] = np.ndarray.flatten(Z)
        gelpcd = open3d.geometry.PointCloud()
        gelpcd.points = open3d.utility.Vector3dVector(points)
        gelpcd_pub = rospy.Publisher("/gsmini_pcd", PointCloud2, queue_size=10)

    if USE_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('roi = ', roi)

    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    if SHOW_3D_NOW:
        vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)

    try:
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():

            #for i in range(NUM_SENSORS):
            #    gs['img'][i] = gs['vs'][i]
            #    gs['img'][i] = cv2.resize(gs['img'][i], (dev.imgw, dev.imgh))
                # gs['img'][i] = chop_border_resize(gs['img'][i])
            #    cv2.imshow('gsmini{}'.format(i), gs['img'][i])

            # get the roi image
            f1 = dev.get_image()
            if USE_ROI:
                f1 = f1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            #cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            gs['img'][0] = cv2.resize(f0, (dev.imgw, dev.imgh))

            cv2.imshow('Img', gs['img'][0])

            ''' Display the results '''
            if SHOW_3D_NOW:
                vis3d.update(dm)

            if PUBLISH_ROS_PC:
                print ('publishing ros point cloud')
                dm_ros = copy.deepcopy(dm) * mpp
                ''' publish point clouds '''
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'gs_mini'
                points[:, 2] = np.ndarray.flatten(dm_ros)
                gelpcd.points = open3d.utility.Vector3dVector(points)
                gelpcdros = pcl2.create_cloud_xyz32(header, np.asarray(gelpcd.points))
                gelpcd_pub.publish(gelpcdros)


                xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(gelpcdros)
                np.shape(xyz_array)

                print(np.shape(xyz_array))


            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

            rate.sleep()

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
