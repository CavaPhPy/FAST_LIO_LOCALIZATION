import math
import numpy as np
from geodesy import utm
from tf.transformations import (
    euler_matrix, 
    euler_from_matrix, 
    concatenate_matrices, 
    inverse_matrix,
    quaternion_matrix 
)
from tf.transformations import quaternion_matrix, euler_matrix 

# --- 1. 数据结构定义 (保持不变) ---
class ConvertRTKData:
    """用于内部处理的 RTK 数据结构，包含地理和姿态信息。"""
    def __init__(self, lat, lon, alt, hdg, pit, rol, world_map_pos=None, world_map_quat=None):
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt
        self.heading = hdg  # 基线逆时针到真北的夹角 (度)
        self.pitch = pit
        self.roll = rol
        self.world_map_position = world_map_pos 
        self.world_map_quaternion = world_map_quat # [x, y, z, w]

# --- 2. 配置参数 ---
'''
！！！！！非常重要，在不同的地方上安装后，要调这个的值。根据雷达和RTK设备的位置！！！！！

T_{RTK→BASELINK_OFFSET​}=(x,y,z,roll,pitch,yaw)
x: RTK 天线中心 相对于 base_link 原点 在 base_link X 轴 (通常是前进方向) 上的偏移量，单位：米
y: RTK 天线中心 相对于 base_link 原点 在 base_link Y 轴 (通常是左侧方向) 上的偏移量，单位：米
z: RTK 天线中心 相对于 base_link 原点 在 base_link Z 轴 (通常是上侧方向) 上的偏移量，单位：米
roll: RTK 坐标系 相对于 base_link 坐标系 绕 base_link X 轴 (滚转) 的旋转角度。单位：度 (Degrees)，0∼360的角度制
pitch: RTK 坐标系 相对于 base_link 坐标系 绕 base_link Y 轴 (俯仰) 的旋转角度。单位：度 (Degrees)，0∼360的角度制
yaw: RTK 坐标系 相对于 base_link 坐标系 绕 base_link Z 轴 (偏航) 的旋转角度。单位：度 (Degrees)，0∼360的角度制
'''
T_RTK_TO_BASELINK_OFFSET = (0.075, 0.05, 0.0, 0.0, 5.0, 90.0) 

# --- 3. 转换器核心类 ---
class RTKConverter:
    def __init__(self):
        self.T_base_to_rtk_mat = self._euler_to_matrix(*T_RTK_TO_BASELINK_OFFSET, use_degrees=True)
        self.T_rtk_to_base_mat = inverse_matrix(self.T_base_to_rtk_mat)
        self.T_map_to_geo_ref = None
        self.reference_utm = None

    def _euler_to_matrix(self, x, y, z, roll, pitch, yaw, use_degrees=False) -> np.ndarray:
        """欧拉角和平移转为 4x4 齐次矩阵"""
        if use_degrees:
            roll, pitch, yaw = map(math.radians, [roll, pitch, yaw])
            
        T = np.array([x, y, z])
        R = euler_matrix(roll, pitch, yaw, 'sxyz') 
        M = np.copy(R)
        M[:3, 3] = T
        return M

    def _rtk_to_matrix(self, rtk_data: ConvertRTKData, is_reference=False) -> np.ndarray:
        """
        修正 V4.0: 使用 ROS 惯例 (X=Northing差, Y=-Easting差, Z=高程差)。
        """
        if self.reference_utm is None and not is_reference:
            raise Exception("RTKConverter 尚未初始化。")
        
        current_utm = utm.fromLatLong(rtk_data.latitude, rtk_data.longitude, rtk_data.altitude)
        
        # 1. 位置：使用 ROS 映射 (X=Northing差, Y=-Easting差, Z=高程差)
        if is_reference:
            x, y, z = 0.0, 0.0, 0.0
        else:
            easting_diff = current_utm.easting - self.reference_utm.easting
            northing_diff = current_utm.northing - self.reference_utm.northing
            
            x = northing_diff  # Geo X (北向)
            y = -easting_diff  # Geo Y (西向)
            z = current_utm.altitude - self.reference_utm.altitude 

        # 2. 姿态： RTK (基线逆时针到北) -> ROS Yaw (X=北=0, 逆时针为正)
        roll_rad = math.radians(rtk_data.roll)
        pitch_rad = math.radians(rtk_data.pitch)
        
        # ROS Yaw = -RTK Heading (弧度)
        ros_yaw_rad = -math.radians(rtk_data.heading)
        ros_yaw_rad = math.atan2(math.sin(ros_yaw_rad), math.cos(ros_yaw_rad))
        
        return self._euler_to_matrix(x, y, z, roll_rad, pitch_rad, ros_yaw_rad, use_degrees=False)


    def initialize_alignment(self, ref_rtk: ConvertRTKData):
        """
        使用绑定了 SLAM 坐标的启动 RTK 数据初始化对齐矩阵 T_map_to_geo_ref。
        """
        if ref_rtk.world_map_position is None or ref_rtk.world_map_quaternion is None:
            raise ValueError("初始 RTK 数据必须包含绑定的 SLAM (world_map) 位姿信息！")
            
        # 1. 设置地理参考点 (UTM)
        self.reference_utm = utm.fromLatLong(
            ref_rtk.latitude, 
            ref_rtk.longitude, 
            ref_rtk.altitude
        )
        
        # --- 2. 构造 T_map_to_base_start (SLAM 初始位姿) ---
        pos = np.array(ref_rtk.world_map_position)
        quat = np.array(ref_rtk.world_map_quaternion)
        T_map_to_base_start = quaternion_matrix(quat)
        T_map_to_base_start[:3, 3] = pos
        
        # --- 3. 构造 T_geo_ref_rtk_link (RTK 传感器在 UTM 原点上的位姿) ---
        # 此时 is_reference=True，保证 x, y, z=0
        T_geo_ref_rtk_link_mat = self._rtk_to_matrix(ref_rtk, is_reference=True)

        # --- 4. 计算 T_map_to_geo_ref (核心对齐矩阵) ---
        
        # T_geo_ref_base_start = T_geo_ref_rtk_link * T_rtk_link_base_link
        T_geo_ref_base_start = concatenate_matrices(T_geo_ref_rtk_link_mat, self.T_rtk_to_base_mat)
        
        # T_map_to_geo_ref = T_map_to_base_start * inv(T_geo_ref_base_start)
        self.T_map_to_geo_ref = concatenate_matrices(T_map_to_base_start, 
                                                     inverse_matrix(T_geo_ref_base_start))

    def convert_to_world_pose(self, current_rtk: ConvertRTKData) -> tuple:
        """
        将实时 RTK 数据转换为世界坐标系 (map) 下的 x, y, z, roll, pitch, yaw。
        """
        if self.T_map_to_geo_ref is None:
            raise Exception("RTKConverter 尚未初始化。")
            
        # 1. 实时计算 T_geo_ref_rtk_link
        T_geo_ref_rtk_link_mat = self._rtk_to_matrix(current_rtk, is_reference=False)

        # 2. 执行完整的坐标转换链: T_map_to_base_link
        # T_map_to_base = T_map_to_geo_ref * T_geo_ref_rtk_link * T_rtk_link_base_link
        T_map_to_base_link_mat = concatenate_matrices(
            self.T_map_to_geo_ref,
            T_geo_ref_rtk_link_mat,
            self.T_rtk_to_base_mat
        )
        
        # 3. 提取结果
        x = T_map_to_base_link_mat[0, 3]
        y = T_map_to_base_link_mat[1, 3]
        z = T_map_to_base_link_mat[2, 3]
        
        # 姿态：使用 ZYX 顺序 (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_matrix(T_map_to_base_link_mat, 'sxyz')
        
        return (x, y, z, roll, pitch, yaw)

if __name__ == "__main__":
    print("**************** 启动 RTK 转换测试 ****************")
    rtk_converter = RTKConverter()

    # --- 模拟 map_origin 初始数据 ---
    initial_position = [0.002087940255250292, -0.001732927803450829, -0.01084360587469013]
    initial_orientation = [0.01420823261530488, 0.003864915807885692, 0.01102814440430194, 0.9998307699719714]
    
    # RTK 初始地理数据
    origin = ConvertRTKData(
        lat=31.53498082,
        lon=104.70193732, 
        alt=478.9721, 
        hdg=237.45,
        pit=0.54, 
        rol=0.0,
        world_map_pos=initial_position,        
        world_map_quat=initial_orientation     
    )
    
    print("--- 初始化对齐 ---")
    rtk_converter.initialize_alignment(origin)
    print("初始化成功。T_map_to_geo_ref 已计算。")

    # --- 模拟新的 RTK 数据 (now) ---
    now = ConvertRTKData(
        lat=31.53506856,
        lon=104.7016149, 
        alt=481.8264, 
        hdg=0.0,
        pit=0.0, 
        rol=0.0
    )

    print("\n--- 转换新数据 ---")
    x, y, z, roll, pitch, yaw = rtk_converter.convert_to_world_pose(now)
    
    print(f"转换结果 (x, y, z): ({x:.6f}, {y:.6f}, {z:.6f})")
    print(f"转换姿态 (roll, pitch, yaw): ({roll:.6f}, {pitch:.6f}, {yaw:.6f})")

    # 验证原点转换 (理论上应该接近 SLAM 初始位姿)
    x_o, y_o, z_o, r_o, p_o, yaw_o = rtk_converter.convert_to_world_pose(origin)
    print("\n--- 验证原点转换结果 (应接近 SLAM 初始位姿) ---")
    print(f"理论原点 (x, y, z): ({x_o:.6f}, {y_o:.6f}, {z_o:.6f})")

    # 您的 IMU 目标结果 (位置)
    print("\n--- 目标 IMU 位置 ---")
    print("目标 (x, y, z): (23.050734, 23.040324, 0.666398)")