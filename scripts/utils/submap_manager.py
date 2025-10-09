#!/usr/bin/python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import os
import yaml
import numpy as np
from collections import namedtuple

# 定义子地图元数据结构
SubmapMetadata = namedtuple('SubmapMetadata', [
    'submap_id', 'map_name', 'position', 'orientation', 
    'aabb_min', 'aabb_max', 'timestamp', 'file_path', 'point_count'
])

class SubmapManager:
    def __init__(self, map_base_path, map_name):
        self.map_base_path = map_base_path
        self.map_name = map_name
        self.map_directory = os.path.join(map_base_path, map_name)
        self.submaps_metadata = []
        self.origin_rtk = None
        
        # 确保路径以/结尾
        if not self.map_base_path.endswith('/'):
            self.map_base_path += '/'
            
        self.map_directory = os.path.join(self.map_base_path, self.map_name)
        
        # 加载元数据和原点信息
        self.load_all_metadata()
        self.load_origin_rtk()
    
    def load_all_metadata(self):
        """加载所有子地图元数据"""
        self.submaps_metadata = []
        
        if not os.path.exists(self.map_directory):
            print(f"警告：目录不存在 {self.map_directory}")
            return
            
        for filename in os.listdir(self.map_directory):
            if filename.endswith('.yaml') and filename.startswith('submap_'):
                filepath = os.path.join(self.map_directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        config = yaml.safe_load(f)
                        
                    position = np.array([
                        config['world_pose']['position']['x'],
                        config['world_pose']['position']['y'],
                        config['world_pose']['position']['z']
                    ])

                    orientation = np.array([
                        config['world_pose']['orientation']['x'],
                        config['world_pose']['orientation']['y'],
                        config['world_pose']['orientation']['z'],
                        config['world_pose']['orientation']['w']
                    ])

                    aabb_min = np.array([
                        config['aabb']['min']['x'],
                        config['aabb']['min']['y'],
                        config['aabb']['min']['z']
                    ])

                    aabb_max = np.array([
                        config['aabb']['max']['x'],
                        config['aabb']['max']['y'],
                        config['aabb']['max']['z']
                    ])

                    metadata = SubmapMetadata(
                        submap_id=config['submap_id'],
                        map_name=config['map_name'],
                        position=position,
                        orientation=orientation,
                        aabb_min=aabb_min,
                        aabb_max=aabb_max,
                        timestamp=config['timestamp'],
                        file_path=config['file_path'],
                        point_count=config['point_count']
                    )
                    
                    self.submaps_metadata.append(metadata)
                except Exception as e:
                    print(f"解析YAML文件失败 {filepath}: {e}")
        
        print(f"加载了 {len(self.submaps_metadata)} 个子地图元数据文件")
    
    def load_origin_rtk(self):
        """加载地图原点RTK信息"""
        rtk_file_path = os.path.join(self.map_directory, "map_origin_rtk.yaml")
        
        try:
            with open(rtk_file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'map_origin' in config:
                self.origin_rtk = config['map_origin']
                print(f"加载RTK原点: lat={self.origin_rtk['latitude']}, lon={self.origin_rtk['longitude']}")
            else:
                print(f"RTK原点未在 {rtk_file_path} 中找到")
        except Exception as e:
            print(f"加载RTK原点失败 {rtk_file_path}: {e}")
    
    def get_candidate_submaps(self, current_lat, current_lon):
        """根据当前位置获取候选子地图"""
        if not self.origin_rtk:
            print("没有RTK原点信息，返回所有子地图")
            return self.submaps_metadata
            
        # 转换当前RTK位置到地图坐标系
        x, y = self.latlon_to_map_xy(
            current_lat, current_lon,
            self.origin_rtk['latitude'], 
            self.origin_rtk['longitude']
        )
        
        # 根据AABB包围盒筛选候选子地图
        candidates = []
        for metadata in self.submaps_metadata:
            if (metadata.aabb_min[0] <= x <= metadata.aabb_max[0] and
                metadata.aabb_min[1] <= y <= metadata.aabb_max[1]):
                candidates.append(metadata)
        
        print(f"找到 {len(candidates)} 个候选子地图，位置: ({current_lat}, {current_lon}) -> ({x}, {y})")
        return candidates
    
    @staticmethod
    def latlon_to_map_xy(lat, lon, origin_lat, origin_lon):
        """经纬度到地图坐标的转换"""
        # 简化的转换，实际应使用更精确的投影转换
        EARTH_RADIUS = 6378137.0
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        origin_lat_rad = np.radians(origin_lat)
        origin_lon_rad = np.radians(origin_lon)
        
        dlat = lat_rad - origin_lat_rad
        dlon = lon_rad - origin_lon_rad
        
        x = dlon * EARTH_RADIUS * np.cos(origin_lat_rad)
        y = dlat * EARTH_RADIUS
        
        return x, y

if __name__ == '__main__':
    # 测试代码
    x, y = SubmapManager.latlon_to_map_xy(31.53481520,104.70077521,31.53484415,104.70077093)
    print(x,y)