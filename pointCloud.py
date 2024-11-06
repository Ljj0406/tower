import ezdxf
import numpy as np
import pandas as pd


def extract_points_from_cad(dxf_path):
    """
    从CAD文件中提取点信息（如直线、圆等几何图形的关键点），
    并将其转换为点云格式的数据。

    参数:
    - dxf_path: CAD文件路径 (DXF格式)

    返回:
    - points: numpy数组，包含点云的(x, y, z)坐标信息
    """
    # 读取CAD文件
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    points = []

    # 遍历CAD文件中的所有实体，提取点信息
    for entity in msp:
        # 提取直线的起点和终点
        if entity.dxftype() == 'LINE':
            start_point = entity.dxf.start.xyz  # 获取起点坐标
            end_point = entity.dxf.end.xyz  # 获取终点坐标
            points.append(start_point)
            points.append(end_point)

        # 提取圆形的中心点，并在圆周上均匀取样
        elif entity.dxftype() == 'CIRCLE':
            center = np.array(entity.dxf.center)
            radius = entity.dxf.radius
            # 在圆周上生成多个采样点
            for angle in np.linspace(0, 2 * np.pi, num=20):
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = center[2]  # Z坐标保持不变
                points.append((x, y, z))

        # 提取多段线的顶点
        elif entity.dxftype() == 'LWPOLYLINE':
            polyline_points = [(point[0], point[1], entity.dxf.elevation) for point in entity]
            points.extend(polyline_points)

        # 提取多边形 (POLYLINE) 的顶点
        elif entity.dxftype() == 'POLYLINE':
            polyline_points = [(vertex.dxf.x, vertex.dxf.y, vertex.dxf.z) for vertex in entity.vertices]
            points.extend(polyline_points)

    # 将点数据转换为numpy数组格式
    points = np.array(points)
    return points


def save_points_to_csv(points, csv_path):
    """
    将点数据保存为CSV文件，用于训练模型。

    参数:
    - points: numpy数组，点云的(x, y, z)坐标信息
    - csv_path: CSV文件的保存路径
    """
    # 转换为DataFrame
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.to_csv(csv_path, index=False)


# 设置文件路径
#你这边得替换为你自己电脑里的路径
dxf_path = 'path/to/your/file.dxf'  # 替换为实际的DXF文件路径
csv_path = 'point_cloud_data.csv'  # 输出的CSV文件路径

# 从CAD文件提取点信息并保存
points = extract_points_from_cad(dxf_path)
save_points_to_csv(points, csv_path)

print("点云数据已成功保存为CSV格式:", csv_path)
