#!/bin/bash

# 保存地图
echo "正在保存地图..."
rosrun map_server map_saver map:=/projected_map -f $1

# 等待地图文件生成
sleep 1

# 定义 YAML 文件路径
YAML_FILE="$1.yaml"

# 检查 YAML 文件是否存在
if [ ! -f "$YAML_FILE" ]; then
    echo "错误: YAML 文件未找到: $YAML_FILE"
    exit 1
fi

# 修复 YAML 文件中的 NaN 值
echo "正在修复 YAML 文件中的 NaN 值..."

# 备份原始文件
cp "$YAML_FILE" "$YAML_FILE.backup"

# 使用 sed 命令直接替换 NaN 值为 0.000000
# 先替换 -nan，再替换 nan
sed -i 's/-\?nan/0.000000/g' "$YAML_FILE"

# 验证修复是否成功
if grep -q "nan" "$YAML_FILE"; then
    echo "警告: 可能仍有 NaN 值存在，尝试替代方法..."
    # 尝试更简单的方法
    sed -i 's/nan/0.000000/g' "$YAML_FILE"
    sed -i 's/-nan/0.000000/g' "$YAML_FILE"
fi

echo "地图保存和修复完成"