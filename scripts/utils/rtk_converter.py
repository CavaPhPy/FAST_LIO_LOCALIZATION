import math
import numpy as np
# 修正：LatLon 模块不存在。根据文件内容，我们应导入 geodesy.utm
from geodesy import utm
# 导入 tf.transformations 库用于矩阵和欧拉角操作
# 注意：在 ROS Noetic (ROS 1) 环境中，应使用 tf.transformations
from tf.transformations import euler_matrix, euler_from_matrix, concatenate_matrices, inverse_matrix

# --- 1. 数据结构定义（请确保与您的实际 RTK 消息字段匹配） ---

class ConvertRTKData:
    """用于内部处理的 RTK 数据结构，包含地理和姿态信息。"""
    def __init__(self, lat, lon, alt, hdg, pit, rol):
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt
        self.heading = hdg  # 基线逆时针与真北夹角 (度)
        self.pitch = pit
        self.roll = rol

# --- 2. 配置参数 (请根据您的实际设备进行修改!) ---

# 2.1 物理结构：RTK 传感器相对于 世界坐标系零点 的固定偏移
# 格式: (x, y, z, roll, pitch, yaw) - 单位: 米和度
# !! 请精确测量您的设备并修改此值 !!
T_RTK_TO_BASELINK_OFFSET = (0.15, 0.0, 0.40, 0.0, 0.0, 0.0) 

# 2.2 逻辑目标：世界坐标系下的目标初始位姿 (T_map_to_base_link_target)
# 默认为单位矩阵 (即启动点是 map 的 (0, 0, 0, 0, 0, 0))
# 如果您想启动在 map 的 (10, 5, 0) 且朝向 90 度，请修改 T_MAP_TO_BASE_TARGET_MAT
T_MAP_TO_BASE_TARGET_MAT = np.eye(4) 
# 示例：如果需要修改，使用 self._euler_to_matrix(10.0, 5.0, 0.0, 0.0, 0.0, math.radians(90.0), use_degrees=False) 

# --- 3. 转换器核心类 ---

class RTKConverter:
    """
    将大地坐标系下的 RTK 数据转换为雷达世界坐标系下的位姿。
    """
    def __init__(self):
        
        # 静态转换矩阵：T_base_link_rtk_link 及其逆矩阵 T_rtk_link_base_link
        self.T_base_to_rtk_mat = self._euler_to_matrix(*T_RTK_TO_BASELINK_OFFSET, use_degrees=True)
        self.T_rtk_to_base_mat = inverse_matrix(self.T_base_to_rtk_mat)

        # 逻辑目标：地图启动目标 (通常是单位矩阵)
        self.T_map_to_base_target = T_MAP_TO_BASE_TARGET_MAT

        # 动态对齐矩阵：T_map_to_geo_ref (启动时计算)
        self.T_map_to_geo_ref = None
        
        # 地理参考：原点 UTM 坐标
        self.reference_utm = None

    def _euler_to_matrix(self, x, y, z, roll, pitch, yaw, use_degrees=False) -> np.ndarray:
        """欧拉角和平移转为 4x4 齐次矩阵"""
        if use_degrees:
            roll, pitch, yaw = map(math.radians, [roll, pitch, yaw])
            
        T = np.array([x, y, z])
        # 'sxyz' 是 ROS 坐标系的常用顺序 (Z-Y-X，即先滚再俯仰再偏航)
        R = euler_matrix(roll, pitch, yaw, 'sxyz') 
        
        M = np.copy(R)
        M[:3, 3] = T
        return M

    def initialize_alignment(self, ref_rtk: ConvertRTKData):
        """
        使用启动时的 RTK 数据初始化坐标系对齐 (T_map_to_geo_ref)。
        必须在第一次调用 convert_to_world_pose 之前执行。
        """
        # 1. 设置地理参考点 (UTM)，修正：使用 utm.fromLatLong
        self.reference_utm = utm.fromLatLong(
            ref_rtk.latitude, 
            ref_rtk.longitude, 
            ref_rtk.altitude
        )
        
        # 2. 计算 T_geo_ref_rtk_link (启动时位姿)
        T_geo_ref_rtk_link_mat = self._rtk_to_matrix(ref_rtk)

        # 3. 计算 T_geo_ref_base_link_start
        # T_geo_ref_base_link_start = T_geo_ref_rtk_link * T_rtk_link_base_link
        T_geo_ref_base_link_start = concatenate_matrices(T_geo_ref_rtk_link_mat, self.T_rtk_to_base_mat)
        
        # 4. 计算对齐矩阵 T_map_to_geo_ref
        # T_map_to_geo_ref = T_map_to_base_target * inv(T_geo_ref_base_link_start)
        self.T_map_to_geo_ref = concatenate_matrices(self.T_map_to_base_target, 
                                                     inverse_matrix(T_geo_ref_base_link_start))

    def _rtk_to_matrix(self, rtk_data: ConvertRTKData) -> np.ndarray:
        """
        将 RTK 数据转换为相对于 self.reference_utm 的 T_geo_ref_rtk_link 矩阵。
        """
        if self.reference_utm is None:
            raise Exception("RTKConverter 尚未初始化，请先调用 initialize_alignment()。")
            
        # 1. 位置 (X=东, Y=北, Z=上)
        # 修正：使用 utm.fromLatLong 函数
        current_utm = utm.fromLatLong(
            rtk_data.latitude, 
            rtk_data.longitude, 
            rtk_data.altitude
        )
        
        # UTM 坐标差值
        x = current_utm.easting - self.reference_utm.easting
        y = current_utm.northing - self.reference_utm.northing
        
        # 高度差值 (如果 UTMPoint 包含 altitude 属性)
        z = current_utm.altitude - self.reference_utm.altitude

        # 2. 姿态 (Roll, Pitch, Yaw)
        roll_rad = math.radians(rtk_data.roll)
        pitch_rad = math.radians(rtk_data.pitch)
        
        # RTK Heading (真北=0, 逆时针为正) -> ROS Yaw (X轴=0, 逆时针为正)
        # 转换公式: ROS Yaw = 90度 + Heading_RTK (这是因为 ROS Y 轴通常指向北)
        ros_yaw_rad = math.radians(90.0) + math.radians(rtk_data.heading)
        
        return self._euler_to_matrix(x, y, z, roll_rad, pitch_rad, ros_yaw_rad, use_degrees=False)

    def convert_to_world_pose(self, current_rtk: ConvertRTKData) -> tuple:
        """
        将实时 RTK 数据转换为世界坐标系 (map) 下的 x, y, z, roll, pitch, yaw。
        
        Returns:
            tuple: (x, y, z, roll, pitch, yaw)，其中姿态为弧度。
        """
        if self.T_map_to_geo_ref is None:
            raise Exception("RTKConverter 尚未初始化，请先调用 initialize_alignment()。")
            
        # 1. 实时计算 T_geo_ref_rtk_link
        T_geo_ref_rtk_link_mat = self._rtk_to_matrix(current_rtk)

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
        
        # 提取欧拉角 (roll, pitch, yaw) 弧度
        roll, pitch, yaw = euler_from_matrix(T_map_to_base_link_mat, 'sxyz')
        
        return (x, y, z, roll, pitch, yaw)