import sys
sys.path.append("/home/haegu/catkin_ws/src/gsrobotics/examples/ros/")
sys.path.append("/home/haegu/catkin_ws/src/digit-depth/")

import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import hydra
from digit_depth.handlers import find_recent_model
from digit_depth.third_party.vis_utils import ContactArea

import gsdevice

# depth image
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from pathlib import Path
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model
from digit_depth.third_party.vis_utils import ContactArea

import setting
import time
import marker_detection
import find_marker

seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "/home/haegu/catkin_ws/src/gsrobotics/examples/ros"
f0 = []

gs = {}
gs['img'] = [0] * 2
gs['gsmini_pub'] = [0] * 2
gs['vs'] = [0] * 2
gs['img_msg'] = [0] * 2

gs['gsmini_maker_img_pub'] = [0] * 1
gs['maker_img_msg'] = [0] * 1

gs['gsmini_depth_img_pub_l'] = [0] * 1
gs['gsmini_depth_img_msg_l'] = [0] * 1

# Set flags
SAVE_VIDEO_FLAG = True
GPU = True
MASK_MARKERS_FLAG = True
USE_ROI = False
PUBLISH_ROS_PC = True
SHOW_3D_NOW = False
# Path to 3d model
path = '.'
calibrate = False

setting.init()
frame0 = None

cvbridge = CvBridge()

flag2 = False

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

def get_depth_values(cfg,model, img_np):
    """
    Calculate the depth values for an image using the given model.

    Parameters:
    - model: PyTorch model for calculating depth values
    - img_np: NumPy array representing the image

    Returns:
    - img_depth: NumPy array of depth values for the image
    """
    img_np = preproc_mlp(img_np)
    img_np = model(img_np).detach().cpu().numpy()
    img_np, _ = post_proc_mlp(img_np)

    gradx_img, grady_img = geom_utils._normal_to_grad_depth(
        img_normal=img_np, gel_width=cfg.sensor.gel_width,
        gel_height=cfg.sensor.gel_height, bg_mask=None
    )
    img_depth = geom_utils._integrate_grad_depth(
        gradx_img, grady_img, boundary=None, bg_mask=None, max_depth=cfg.max_depth
    )
    img_depth = img_depth.detach().cpu().numpy().flatten()
    return img_depth

def sensor_1(model, video, cfg, pub, pub_img, pub_left_pca, flag, mask, mc):
    if flag == 1:
        #while not rospy.is_shutdown():
        frame = video.get_image()

        if frame is None:
            print("No fame")
        else:
            """Marker Tracking"""
            # mccopy = mc
            # mc_sorted1 = mc[mc[:, 0].argsort()]
            # mc1 = mc_sorted1[:setting.N_]
            # mc1 = mc1[mc1[:, 1].argsort()]
            #
            # mc_sorted2 = mc[mc[:, 1].argsort()]
            # mc2 = mc_sorted2[:setting.M_]
            # mc2 = mc2[mc2[:, 0].argsort()]
            #
            # """
            # N_, M_: the row and column of the marker array
            # x0_, y0_: the coordinate of upper-left marker
            # dx_, dy_: the horizontal and vertical interval between adjacent markers
            # """
            # N_ = setting.N_
            # M_ = setting.M_
            # fps_ = setting.fps_
            # x0_ = np.round(mc1[0][0])
            # y0_ = np.round(mc1[0][1])
            # dx_ = mc2[1, 0] - mc2[0, 0]
            # dy_ = mc1[1, 1] - mc1[0, 1]
            #
            # #print('x0:', x0_, '\n', 'y0:', y0_, '\n', 'dx:', dx_, '\n', 'dy:', dy_)
            # radius, coverage = compute_tracker_gel_stats(mask)
            # m = find_marker.Matching(N_, M_, fps_, x0_, y0_, dx_, dy_)
            # frameno = 0
            #
            # calibrate = False
            # ############################################################################################################
            # ret, frame_maker = video.get_image_two()
            #
            # '''maker tracking '''
            # frame_maker = resize_crop_mini(frame_maker, video.imgw, video.imgh)
            # raw_img = copy.deepcopy(frame_maker)
            #
            # ''' EXTRINSIC calibration ...
            # ... the order of points [x_i,y_i] | i=[1,2,3,4], are same
            # as they appear in plt.imshow() image window. Put them in
            # clockwise order starting from the topleft corner'''
            # # frame = warp_perspective(frame, [[35, 15], [320, 15], [290, 360], [65, 360]], output_sz=frame.shape[:2])   # params for small dots
            # # frame = warp_perspective(frame, [[180, 130], [880, 130], [800, 900], [260, 900]], output_sz=(640,480)) # org. img size (1080x1080)
            #
            # ### find marker masks
            # mask = marker_detection.find_marker(frame_maker)
            #
            # ### find marker centers
            # mc = marker_detection.marker_center(mask, frame_maker)
            #
            # if calibrate == False:
            #     tm = time.time()
            #     ### matching init
            #     m.init(mc)
            #     ### matching
            #     m.run()
            #     # print(time.time() - tm)
            #
            #     ### matching result
            #     """
            #     output: (Ox, Oy, Cx, Cy, Occupied) = flow
            #         Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            #         Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            #         Occupied: N*M matrix, the index of the marker at each position, -1 means inferred.
            #             e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
            #     """
            #     flow = m.get_flow()
            #     frame0 = None
            #
            #     if frame0 is None:
            #         frame0 = frame_maker.copy()
            #         frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)
            #
            #     # diff = (frame * 1.0 - frame0) * 4 + 127
            #     # trim(diff)
            #
            #     # # draw flow
            #     marker_detection.draw_flow(frame_maker, flow)
            #
            #     frameno = frameno + 1
            #
            #     # if SAVE_DATA_FLAG:
            #     #     Ox, Oy, Cx, Cy, Occupied = flow
            #     #     for i in range(len(Ox)):
            #     #         for j in range(len(Ox[i])):
            #     #             datafile.write(
            #     #                f"{frameno}, {i}, {j}, {Ox[i][j]:.2f}, {Oy[i][j]:.2f}, {Cx[i][j]:.2f}, {Cy[i][j]:.2f}\n")
            #
            # # mask_img = mask.astype(frame[0].dtype)
            # mask_img = np.asarray(mask)
            # # print(np.shape(mask))
            # bigframe_marker = cv2.resize(frame_maker, (frame_maker.shape[1] * 3, frame_maker.shape[0] * 3))
            # cv2.imshow('frame_maker', bigframe_marker)
            # cv2.waitKey(1)
            # # bigmask = cv2.resize(mask_img * 255, (mask_img.shape[1] * 3, mask_img.shape[0] * 3))
            # # cv2.imshow('mask', bigmask)
            #
            # ''' publish maker image to ros '''
            # for i in range(1):
            #     gs['maker_img_msg'][i] = cvbridge.cv2_to_imgmsg(bigframe_marker, encoding="passthrough")
            #     gs['maker_img_msg'][i].header.stamp = rospy.Time.now()
            #     gs['maker_img_msg'][i].header.frame_id = 'map'
            #     gs['gsmini_maker_img_pub'][i].publish(gs['maker_img_msg'][i])
            ############################################################################################################
            """Depth value/ PCA"""
            # dp_zero is the "background" depth value.
            if sensor_1.dp_zero_counter < 100:
                img_depth = get_depth_values(cfg, model, frame)
                sensor_1.dp_zero += np.min(img_depth)
                sensor_1.dp_zero_counter += 1

            else:
                if sensor_1.dp_zero_counter == 100:
                    sensor_1.dp_zero = sensor_1.dp_zero / 100
                    sensor_1.dp_zero_counter += 1

                # depth value
                img_depth = get_depth_values(cfg, model, frame)
                max_deformation = np.min(img_depth)
                #print(f"Max deformation 1 : {max_deformation}")
                actual_deformation = np.abs((max_deformation - sensor_1.dp_zero))
                pub.publish(Float32(actual_deformation * 1000))  # convert to mm

                # raw image
                img_np = preproc_mlp(frame)

                img_np = model(img_np).detach().cpu().numpy()
                img_np, _ = post_proc_mlp(img_np)

                # get gradx and grady
                gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np,
                                                                        gel_width=cfg.sensor.gel_width,
                                                                        gel_height=cfg.sensor.gel_height, bg_mask=f0)


                #img_depth = geom_utils.depth_to_depth(img_depth, f0, boundary=None, params=None)
                # reconstruct depth
                img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,
                                                             max_depth=cfg.max_depth)

                #img_depth = geom_utils.depth_to_depth(img_depth, f0, boundary=None, params=None)

                img_depth = img_depth.detach().cpu().numpy()  # final depth image for current image
                # # Get the first 50 frames and average them to get the zero depth

                br = CvBridge()

                if sensor_1.dm_zero_counter < 50:
                    sensor_1.dm_zero += img_depth
                    sensor_1.dm_zero_counter += 1
                else:
                    if sensor_1.dm_zero_counter == 50:
                        sensor_1.dm_zero = sensor_1.dm_zero/50
                        sensor_1.dm_zero_counter += 1

                    # remove the zero depth
                    diff = img_depth - sensor_1.dm_zero
                    diff = diff * 255
                    diff = diff * -1


                    ret, thresh4 = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)

                    if cfg.visualize.ellipse:
                        img = thresh4
                        pt = ContactArea()
                        axis_data= pt.__call__(target=thresh4, raw_img=frame)
                        #print(center)
                        #print(np.center))
                        if axis_data is not None:
                            #print("Sensor 1 --------------")
                            #print(center[0])
                            #print("--------------")
                        # poly, major_axis, major_axis_end, minor_axis, minor_axis_end
                            #print("major_axis",axis_data[0])
                            #print("major_axis_end", axis_data[1])
                            #print("center", axis_data[2])
                            msg_pca_data = Float32MultiArray()  # the data to be sent, initialise the array

                            # thing = MultiArrayDimension()
                            # thing.label = "x"
                            # thing.size = 6
                            # thing.stride = 1
                            # msg_pca_data.layout.dim.append(thing)

                            msg_pca_data.data = [axis_data[0][0],axis_data[0][1],axis_data[1][0]
                                                 ,axis_data[1][1], axis_data[2][0], axis_data[2][1]]  # assign the array with the value you want to send

                            #print("!!!!!!!!!!!!!!!!")
                            pub_left_pca.publish(msg_pca_data)


                        msg = br.cv2_to_imgmsg(frame, encoding="rgb8")
                        msg.header.stamp = rospy.Time.now()
                        pub_img.publish(msg)

                        f_stream = cv2.resize(frame, (video.imgw*3, video.imgh*3))
                        # cv2.imshow('Depth Image1', f_stream)
                        # cv2.waitKey(1)

                    else:
                        msg = br.cv2_to_imgmsg(thresh4, encoding="passthrough")
                        msg.header.stamp = rospy.Time.now()
                        pub_img.publish(msg)
                    #now = rospy.get_rostime()
                    #rospy.loginfo("published depth image at {}".format(now))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("terminate")

                    #rospy.loginfo(f"Published msg at {rospy.get_time()}")
            #print("sensor_1.dp_zero",sensor_1.dp_zero)
            #print("sensor_1.dp_zero_counter", sensor_1.dp_zero_counter)
    else:
        video.stop_video()

sensor_1.dp_zero = 0
sensor_1.dp_zero_counter = 0
sensor_1.dm_zero_counter = 0
sensor_1.dm_zero = 0

@hydra.main(config_path=f"{base_path}/config", config_name="gelsight.yaml", version_base=None)
def main(cfg):
    rospy.init_node('depth', anonymous=True)

    pub_left = rospy.Publisher('left_gs/depth/value', Float32, queue_size=1)
    pub_left_pca = rospy.Publisher('left_gs/pca/value', Float32MultiArray, queue_size=1)
    #pub_right = rospy.Publisher('right_gs/depth/value', Float32, queue_size=1)
    image_pub_left = rospy.Publisher("left_gs/rgb/image_raw/", Image, queue_size=10)
    gs['gsmini_maker_img_pub'][0] = rospy.Publisher("/gsmini_rawimg_maker_img_", Image, queue_size=1)
    #image_pub_right = rospy.Publisher("right_gs/rgb/image_raw/", Image, queue_size=10)

    model_path = find_recent_model(f"{base_path}/config/")
    model = torch.load(model_path).to(device)

    model.eval()

    #2d image
    NUM_SENSORS = 1

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.
    # the device ID can change after chaning the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")

    dev.connect()
    f0 = dev.get_raw_image()
    cv2.waitKey(0)

    flag = 1
    counter = 0
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        try:
            flag = 1

            while counter < 49:
                ret, frame = dev.get_image_two()
                print('flush black imgs')

                if counter == 48:
                    ret, frame = dev.get_image_two()
                    ##########################
                    frame = resize_crop_mini(frame, dev.imgw, dev.imgh)
                    ### find marker masks
                    mask = marker_detection.find_marker(frame)
                    ### find marker centers
                    mc = marker_detection.marker_center(mask, frame)
                    counter += 1
                    break
                counter +=1

            if counter > 50:
                counter = 50

            sensor_1(model, dev, cfg, pub_left, image_pub_left, pub_left_pca, flag, mask, mc)
        except KeyboardInterrupt:
            flag = 0

if __name__ == '__main__':
    rospy.loginfo("starting...")
    main()
