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

import setting
import time
import marker_detection
import find_marker

from std_msgs.msg import Float64

def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5

def resize_crop_mini(img, imgw, imgh):
    # resize, crop and resize back
    img = cv2.resize(img, (895, 672))  # size suggested by janos to maintain aspect ratio
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    return img

def trim(img):
    img[img<0] = 0
    img[img>255] = 255



def compute_tracker_gel_stats(thresh):
    numcircles = 9 * 7;
    mmpp = .063;
    true_radius_mm = .5;
    true_radius_pixels = true_radius_mm / mmpp;
    circles = np.where(thresh)[0].shape[0]
    circlearea = circles / numcircles;
    radius = np.sqrt(circlearea / np.pi);
    radius_in_mm = radius * mmpp;
    percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2);
    return radius_in_mm, percent_coverage*100.


def main(argv):

    rospy.init_node('showmini3dros', anonymous=True)

    #2d image
    NUM_SENSORS = 1

    gs = {}
    gs['img'] = [0] * 2
    gs['gsmini_pub'] = [0] * 2
    gs['vs'] = [0] * 2
    gs['img_msg'] = [0] * 2

    gs['gsmini_maker_img_pub'] = [0] * 1
    gs['maker_img_msg'] = [0] * 1

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

    cvbridge = CvBridge()
    for i in range(NUM_SENSORS):
        gs['gsmini_pub'][i] = rospy.Publisher("/gsmini_rawimg_{}".format(i), Image, queue_size=1)

    gs['gsmini_maker_img_pub'][0] = rospy.Publisher("/gsmini_rawimg_maker_img_", Image, queue_size=1)

    gs_max_z_pub = rospy.Publisher("/gsmini_max_z", Float64, queue_size=1)
    gs_max_z_msg = 0
    #Marker tracking
    # Resize scale for faster image processing
    setting.init()
    frame0 = None
    counter = 0

    print('DONE')
    while 1:
        if counter<50:
            ret,frame = dev.get_image_two()
            print ('flush black imgs')

            if counter == 48:
                ret, frame = dev.get_image_two()
                ##########################
                frame = resize_crop_mini(frame, dev.imgw, dev.imgh)
                ### find marker masks
                mask = marker_detection.find_marker(frame)
                ### find marker centers
                mc = marker_detection.marker_center(mask, frame)
                break

            counter += 1

    counter = 0
    mccopy = mc
    mc_sorted1 = mc[mc[:,0].argsort()]
    mc1 = mc_sorted1[:setting.N_]
    mc1 = mc1[mc1[:,1].argsort()]

    mc_sorted2 = mc[mc[:,1].argsort()]
    mc2 = mc_sorted2[:setting.M_]
    mc2 = mc2[mc2[:,0].argsort()]

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker
    dx_, dy_: the horizontal and vertical interval between adjacent markers
    """
    N_= setting.N_
    M_= setting.M_
    fps_ = setting.fps_
    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    print ('x0:',x0_,'\n', 'y0:', y0_,'\n', 'dx:',dx_,'\n', 'dy:', dy_)
    radius, coverage = compute_tracker_gel_stats(mask)
    m = find_marker.Matching(N_, M_, fps_, x0_, y0_, dx_, dy_)
    frameno = 0

    calibrate = False

    ''' use this to plot just the 3d '''
    if SHOW_3D_NOW:
        vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)

    try:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # get the roi image
            f1 = dev.get_image()

            ret, frame = dev.get_image_two()
            if not(ret):
                break

            '''maker tracking '''
            frame = resize_crop_mini(frame, dev.imgw, dev.imgh)
            raw_img = copy.deepcopy(frame)

            ''' EXTRINSIC calibration ... 
            ... the order of points [x_i,y_i] | i=[1,2,3,4], are same 
            as they appear in plt.imshow() image window. Put them in 
            clockwise order starting from the topleft corner'''
            # frame = warp_perspective(frame, [[35, 15], [320, 15], [290, 360], [65, 360]], output_sz=frame.shape[:2])   # params for small dots
            # frame = warp_perspective(frame, [[180, 130], [880, 130], [800, 900], [260, 900]], output_sz=(640,480)) # org. img size (1080x1080)

            ### find marker masks
            mask = marker_detection.find_marker(frame)

            ### find marker centers
            mc = marker_detection.marker_center(mask, frame)

            if calibrate == False:
                tm = time.time()
                ### matching init
                m.init(mc)
                ### matching
                m.run()
                #print(time.time() - tm)

                ### matching result
                """
                output: (Ox, Oy, Cx, Cy, Occupied) = flow
                    Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
                    Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
                    Occupied: N*M matrix, the index of the marker at each position, -1 means inferred.
                        e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
                """
                flow = m.get_flow()

                if frame0 is None:
                    frame0 = frame.copy()
                    frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

                # diff = (frame * 1.0 - frame0) * 4 + 127
                # trim(diff)

                # # draw flow
                marker_detection.draw_flow(frame, flow)

                frameno = frameno + 1

                # if SAVE_DATA_FLAG:
                #     Ox, Oy, Cx, Cy, Occupied = flow
                #     for i in range(len(Ox)):
                #         for j in range(len(Ox[i])):
                #             datafile.write(
                #                f"{frameno}, {i}, {j}, {Ox[i][j]:.2f}, {Oy[i][j]:.2f}, {Cx[i][j]:.2f}, {Cy[i][j]:.2f}\n")

            #mask_img = mask.astype(frame[0].dtype)
            mask_img = np.asarray(mask)
            bigframe_marker = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3))
            #cv2.imshow('frame', bigframe_marker)
            #bigmask = cv2.resize(mask_img * 255, (mask_img.shape[1] * 3, mask_img.shape[0] * 3))
            #cv2.imshow('mask', bigmask)

            ''' publish maker image to ros '''
            for i in range(NUM_SENSORS):
                gs['maker_img_msg'][i] = cvbridge.cv2_to_imgmsg(bigframe_marker, encoding="passthrough")
                gs['maker_img_msg'][i].header.stamp = rospy.Time.now()
                gs['maker_img_msg'][i].header.frame_id = 'map'
                gs['gsmini_maker_img_pub'][i].publish(gs['maker_img_msg'][i])


            ''' publish image to ros '''
            bigframe = cv2.resize(f1, (dev.imgw,dev.imgh))
            f_stream = f1
            f_stream = cv2.resize(f_stream, (dev.imgw,dev.imgh))
            #cv2.imshow('Image', f_stream)

            for i in range(NUM_SENSORS):
                gs['img_msg'][i] = cvbridge.cv2_to_imgmsg(f_stream, encoding="passthrough")
                gs['img_msg'][i].header.stamp = rospy.Time.now()
                gs['img_msg'][i].header.frame_id = 'map'
                gs['gsmini_pub'][i].publish(gs['img_msg'][i])

            if USE_ROI:
                f1 = f1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            #compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            ''' Display the results '''
            if SHOW_3D_NOW:
                vis3d.update(dm)

            if PUBLISH_ROS_PC:
                #print ('publishing ros point cloud')
                dm_ros = copy.deepcopy(dm) * mpp
                ''' publish point clouds '''
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'gs_mini'
                points[:, 2] = np.ndarray.flatten(dm_ros)
                gelpcd.points = open3d.utility.Vector3dVector(points)
                gelpcdros = pcl2.create_cloud_xyz32(header, np.asarray(gelpcd.points))
                gelpcd_pub.publish(gelpcdros)

                # to xyz array
                xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(gelpcdros)
                gs_max_z_msg = np.max(xyz_array[:,2])
                gs_max_z_pub.publish(gs_max_z_msg)

                #print(type(xyz_array))
                #print('MAX Z:: ',np.max(xyz_array[:,2]))
                #np.shape(xyz_array)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

            #rate.sleep()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
