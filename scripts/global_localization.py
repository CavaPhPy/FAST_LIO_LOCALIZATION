#!/usr/bin/python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import os
from utils.submap_manager import SubmapManager
from utils.rtk_converter import RTKConverter, ConvertRTKData

from slam_utils.msg import RTKData
import rospkg

import copy
import _thread
import time

import open3d as o3d
import rospy
import ros_numpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf
import tf.transformations

global_map = None
initialized = False
T_map_to_odom = np.eye(4)
cur_odom = None
cur_scan = None

# 融合子地图与RTK数据
submap_manager = None
latest_rtk_data = None
current_submaps = []  # 记录当前加载的子地图

def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc


def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale), voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )

    return result_icp.transformation, result_icp.fitness


def inverse_se3(trans):
    trans_inverse = np.eye(4)
    # R
    trans_inverse[:3, :3] = trans[:3, :3].T
    # t
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):
    # 当前scan原点的位姿
    T_odom_to_base_link = pose_to_mat(cur_odom)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)

    # 把地图转换到lidar系下
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

    # 将视角内的地图点提取出来
    if FOV > 3.14:
        # 环状lidar 仅过滤距离
        indices = np.where(
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else:
        # 非环状lidar 保前视范围
        # FOV_FAR>x>0 且角度小于FOV
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

    # 发布fov内点云
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])

    return global_map_in_FOV


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom
    # 用icp配准
    # print(global_map, cur_scan, T_map_to_odom)
    rospy.loginfo('Global localization by scan-to-map matching......')

    # TODO 这里注意线程安全
    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)

    # 粗配准
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)

    # 精配准
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation,
                                                    scale=1)
    toc = time.time()
    rospy.loginfo('Time: {}'.format(toc - tic))
    rospy.loginfo('')

    # 当全局定位成功时才更新map2odom
    if fitness > LOCALIZATION_TH:
        # T_map_to_odom = np.matmul(transformation, pose_estimation)
        T_map_to_odom = transformation

        # 发布map_to_odom
        map_to_odom = Odometry()
        xyz = tf.transformations.translation_from_matrix(T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        return True
    else:
        rospy.logwarn('Not match!!!!')
        rospy.logwarn('{}'.format(transformation))
        rospy.logwarn('fitness score:{}'.format(fitness))
        return False


def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


# def initialize_global_map(pc_msg):
#     global global_map

#     global_map = o3d.geometry.PointCloud()
#     global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
#     global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
#     rospy.loginfo('Global map received.')

# 融合RTK后初始化代码
def initialize_global_map_with_rtk(rtk_data):
    global global_map, submap_manager, current_submaps
    
    # 根据RTK位置获取候选子地图
    candidates = submap_manager.get_candidate_submaps(rtk_data)
    
    if not candidates:
        rospy.logwarn("No candidate submaps found for position: {}, {}".format(rtk_data.latitude, rtk_data.longitude))
        return False

    # 保存当前加载的子地图信息
    current_submaps = candidates
    
    # 加载候选子地图并合并
    global_map = o3d.geometry.PointCloud()
    for candidate in candidates:
        pcd_file = os.path.join(submap_manager.map_directory, candidate.file_path)
        if os.path.exists(pcd_file):
            try:
                submap = o3d.io.read_point_cloud(pcd_file)
                global_map += submap
                rospy.loginfo("Loaded submap: {}".format(pcd_file))
            except Exception as e:
                rospy.logwarn("Failed to load submap {}: {}".format(pcd_file, str(e)))
        else:
            rospy.logwarn("Submap file not found: {}".format(pcd_file))
    
    if len(global_map.points) > 0:
        # 降采样用于实际匹配计算
        global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
        rospy.loginfo('Global map initialized with {} submaps, total points: {}'.format(
            len(candidates), len(global_map.points)))
        # 发布当前使用的子地图点云（原来的代码要发布点云，但是现在看来好像只是用来展示，所以先注释掉）
        # publish_current_submap_points()
        return True
    else:
        rospy.logwarn('Failed to initialize global map')
        return False

# 发布当前使用的子地图点云
def publish_current_submap_points():
    global pub_global_map, global_map
    
    # 直接使用已经在内存中的global_map
    if global_map and len(global_map.points) > 0:
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        publish_point_cloud(pub_global_map, header, np.array(global_map.points))
        rospy.loginfo('Published global map points, total points: {}'.format(len(global_map.points)))

# 添加RTK数据订阅（回调函数）
def cb_rtk_data(rtk_msg):
    global latest_rtk_data
    latest_rtk_data = rtk_msg

def get_current_rtk_data():
    """
    返回当前有效的RTK数据对象，如果数据无效或过期则返回None
    这样调用者可以直接访问所有需要的属性
    """
    global latest_rtk_data
    # 使用局部变量减少临界区时间
    local_rtk_data = latest_rtk_data
    if local_rtk_data and (rospy.Time.now() - local_rtk_data.header.stamp).to_sec() < 3.0:
        return local_rtk_data
    return None

# 智能定位
def smart_global_localization():
    global global_map, cur_scan, cur_odom, T_map_to_odom, submap_manager
    
    # 获取当前RTK位置
    current_rtk_data = get_current_rtk_data()
    if current_rtk_data is None:
        rospy.logwarn("No current RTK data available")
        return False
        
    # 获取候选子地图
    candidates = submap_manager.get_candidate_submaps(current_rtk_data)

    # 检查是否需要更新地图（当候选子地图与当前加载的子地图不同时）
    if set([c.file_path for c in candidates]) != set([c.file_path for c in current_submaps]):
        rospy.loginfo("Updating map with new submaps")
        # 更新全局地图
        if not initialize_global_map_with_rtk(current_rtk_data):
            rospy.logwarn("Failed to update global map")
            # 如果更新失败，继续使用现有地图进行定位
            return global_localization(T_map_to_odom)
        return True
    
    # 首先尝试候选子地图
    rospy.loginfo("Trying candidate submaps: {}".format(len(candidates)))
    if try_localization_with_submaps(candidates):
        return True
    
    # 如果失败，尝试所有子地图
    rospy.logwarn("Candidate submaps failed, trying all submaps")
    all_submaps = submap_manager.submaps_metadata
    return try_localization_with_submaps(all_submaps)

def try_localization_with_submaps(submap_list):
    global global_map
    
    for i, submap_meta in enumerate(submap_list):
        try:
            # 加载单个子地图
            pcd_file = os.path.join(submap_manager.map_directory, submap_meta.file_path)
            if os.path.exists(pcd_file):
                temp_map = o3d.io.read_point_cloud(pcd_file)
                temp_map = voxel_down_sample(temp_map, MAP_VOXEL_SIZE)
                
                # 临时设置为全局地图进行定位尝试
                original_map = global_map
                global_map = temp_map
                
                success = global_localization(T_map_to_odom)
                
                if success:
                    rospy.loginfo("Localization succeeded with submap: {}".format(submap_meta.file_path))
                    return True
                else:
                    # 恢复原地图
                    global_map = original_map
            else:
                rospy.logwarn("Submap file not found: {}".format(pcd_file))
                
        except Exception as e:
            rospy.logwarn("Failed to try submap {}: {}".format(submap_meta.file_path, str(e)))
    
    return False

def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan
    # 注意这里fastlio直接将scan转到odom系下了 不是lidar局部系
    pc_msg.header.frame_id = 'camera_init'
    pc_msg.header.stamp = rospy.Time().now()
    pub_pc_in_map.publish(pc_msg)

    # 转换为pcd
    # fastlio给的field有问题 处理一下
    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)

    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])


def thread_localization():
    global T_map_to_odom
    while True:
        # 每隔一段时间进行全局定位
        rospy.sleep(1 / FREQ_LOCALIZATION)
        # TODO 由于这里Fast lio发布的scan是已经转换到odom系下了 所以每次全局定位的初始解就是上一次的map2odom 不需要再拿odom了
        # global_localization(T_map_to_odom)
        # 使用智能定位流程
        smart_global_localization()

def send_initial_pose_from_rtk():
    """
    基于RTK坐标计算并发送初始位姿，包括位置和姿态信息
    """
    global submap_manager, rtk_converter
    
    # 获取当前RTK位置
    current_rtk_data = get_current_rtk_data()
    if current_rtk_data is None:
        rospy.logwarn("No current RTK data available")
        return False

    # 检查是否有原点RTK信息
    if not submap_manager.origin_rtk:
        rospy.logwarn("No origin RTK data available")
        return False
    
    try:
        # 获取实时 RTK 数据
        RTK_CURRENT_DATA = ConvertRTKData(
            lat=current_rtk_data.latitude,
            lon=current_rtk_data.longitude,
            alt=current_rtk_data.altitude, 
            hdg=current_rtk_data.heading,
            pit=current_rtk_data.pitch, 
            rol=-current_rtk_data.roll
        )
        # 6. 转换并获取最终的世界坐标位姿
        x, y, z, roll, pitch, yaw = rtk_converter.convert_to_world_pose(RTK_CURRENT_DATA)
        send_initial_pose(x, y, z, roll, pitch, yaw)
        rospy.loginfo(f"Auto Sent initial pose based on RTK: pos=({x:.2f}, {y:.2f}, {z:.2f}), att=({roll:.2f}, {pitch:.2f}, {yaw:.2f})")
        return True
    except Exception as e:
        rospy.logwarn(f"Failed to compute initial pose from RTK: {e}")
        import traceback
        rospy.logwarn(traceback.format_exc())
        return False

def send_initial_pose(x, y, z, roll, pitch, yaw):
    """
    发送初始位姿到 /initialpose topic
    """
    # 创建发布者
    pub_pose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
    
    # 等待订阅者连接
    rospy.sleep(0.1)
    
    # 转换为pose
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    xyz = [x, y, z]

    initial_pose = PoseWithCovarianceStamped()
    initial_pose.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
    initial_pose.header.stamp = rospy.Time().now()
    initial_pose.header.frame_id = 'map'
    
    rospy.loginfo('Sending Initial Pose: {} {} {} {} {} {}'.format(
        x, y, z, roll, pitch, yaw))
    pub_pose.publish(initial_pose)
    
    # 确保消息被发送
    rospy.sleep(0.1)

if __name__ == '__main__':
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = 0.5

    # The threshold of global localization,
    # only those scan2map-matching with higher fitness than LOCALIZATION_TH will be taken
    LOCALIZATION_TH = 0.95

    # FOV(rad), modify this according to your LiDAR type
    FOV = 6.28

    # The farthest distance(meters) within FOV
    FOV_FAR = 100

    rospy.init_node('fast_lio_localization')
    rospy.loginfo('Localization Node Inited...')

    # 初始化SubmapManager
    rospack = rospkg.RosPack()
    try:
        fast_lio_path = rospack.get_path('fast_lio')
        default_map_base_path = os.path.join(fast_lio_path, 'PCD')
    except rospkg.ResourceNotFound:
        default_map_base_path = '/tmp/PCD'  # fallback路径
    map_base_path = rospy.get_param('/map_base_path', default_map_base_path)
    map_name = rospy.get_param('/map_name', 'default_map')
    print("global_localization init params: ",map_base_path, map_name)
    submap_manager = SubmapManager(map_base_path, map_name)

    # publisher
    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)
     # 添加发布全局地图点云的publisher
    pub_global_map = rospy.Publisher('/map', PointCloud2, queue_size=1)

    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)
    # RTK订阅
    rospy.Subscriber('/rtk_data', RTKData, cb_rtk_data, queue_size=1)

    # 初始化全局地图（现在取消全局地图方式的初始化）
    # rospy.logwarn('Waiting for global map......')
    # initialize_global_map(rospy.wait_for_message('/map', PointCloud2))


    # 实例化转换器（注意这里面有个T_RTK_TO_BASELINK_OFFSET需要根据实际情况修改！！！）
    global rtk_converter
    rtk_converter = RTKConverter()

    # 在上方SubmapManager初始化的时候，理论上origin_rtk就有值了。
    map_origin_rtk = submap_manager.origin_rtk
    if not map_origin_rtk:
        rospy.logwarn("No origin RTK data available")
        exit(1)

    RTK_START_DATA = ConvertRTKData(
        lat=map_origin_rtk['latitude'],
        lon=map_origin_rtk['longitude'],
        alt=map_origin_rtk['altitude'],
        hdg=map_origin_rtk['heading'],
        pit=map_origin_rtk['pitch'],
        rol=map_origin_rtk['roll'],
        world_map_pos=[
            map_origin_rtk['world_map']['position']['x'],
            map_origin_rtk['world_map']['position']['y'],
            map_origin_rtk['world_map']['position']['z']
        ] if 'world_map' in map_origin_rtk else None,
        world_map_quat=[
            map_origin_rtk['world_map']['orientation']['x'],
            map_origin_rtk['world_map']['orientation']['y'],
            map_origin_rtk['world_map']['orientation']['z'],
            map_origin_rtk['world_map']['orientation']['w']
        ] if 'world_map' in map_origin_rtk else None
    )

    # 初始化坐标系对齐
    rtk_converter.initialize_alignment(RTK_START_DATA)
    rospy.loginfo("转换器已初始化。地理坐标系已对齐到世界坐标系。")


    # 等待RTK数据并初始化地图 (设置超时)
    rospy.logwarn('Waiting for RTK data to initialize global map...')
    rtk_initialized = False
    start_time = time.time()
    timeout = 30.0  # 30秒超时
    
    while not rtk_initialized and not rospy.is_shutdown() and (time.time() - start_time) < timeout:
        rtk_data = get_current_rtk_data()
        if rtk_data is not None:
            if initialize_global_map_with_rtk(rtk_data):
                rtk_initialized = True
                break
        rospy.sleep(0.1)
        rospy.loginfo_throttle(5, "Waiting for RTK data...")

    if not rtk_initialized:
        rospy.logerr("Failed to initialize with RTK data within timeout")
        exit(1)

    # 初始化
    attempt_count = 0
    max_attempts = 300  # 最大尝试次数，避免无限循环
    while not initialized:
        rospy.logwarn(f'Waiting for initial pose.... (Attempt {attempt_count + 1}/{max_attempts})')

        # 每次循环都发送基于最新RTK数据的初始位姿
        if send_initial_pose_from_rtk():
            rospy.loginfo("Successfully sent initial pose based on RTK coordinates")
        else:
            rospy.logwarn("Failed to send initial pose from RTK, will retry")

        # 等待初始位姿
        pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
        initial_pose = pose_to_mat(pose_msg)
        if cur_scan:
            initialized = global_localization(initial_pose)
            if not initialized:
                attempt_count += 1
        else:
            rospy.logwarn('First scan not received!!!!!')
            attempt_count += 1

    rospy.loginfo('')
    rospy.loginfo('Initialize successfully!!!!!!')
    rospy.loginfo('')
    # 开始定期全局定位
    _thread.start_new_thread(thread_localization, ())

    rospy.spin()
