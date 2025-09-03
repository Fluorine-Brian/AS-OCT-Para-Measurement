# 批量测量LT LT LT LT LT LT LT LT LT LT LT LT LT ，最高点法
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv
from PIL import Image


def find_intersection(mask_path, point1, point2):
    # 像素点最高点
    # 找到晶状体最下角顶点 因为在分割掩码中白色为255即最大
    def get_highest_point(image_path):
        # 打开图像
        image = Image.open(image_path)
        # 转换为灰度图像
        image_gray = image.convert("L")
        # 获取图像的宽度和高度
        width, height = image_gray.size
        # 初始化最高点的坐标
        highest_point = None
        highest_pixel_value = 0
        # 遍历图像的每个像素
        for yy in range(height):
            y = height - yy - 1
            for x in range(width):
                # 获取像素值
                pixel_value = image_gray.getpixel((x, y))
                # 如果当前像素值大于0且大于最高像素值，则更新最高像素值和最高点坐标
                if pixel_value > 0 and pixel_value > highest_pixel_value:
                    highest_pixel_value = pixel_value
                    highest_point = (x, y)
                    # print("now highest_point: ",highest_point,'  (x,y)',(x,y))
        return highest_point

    # 计算垂直平分线的两个点
    def perpendicular_points(mid, p1, p2):
        vector = np.array(p2) - np.array(p1)
        perp_vector = np.array([-vector[1], vector[0]])
        midpoint_vector = np.array(mid)
        pt1 = midpoint_vector + perp_vector
        pt2 = midpoint_vector - perp_vector
        return pt1, pt2

    # 平行垂直平分线 且通过最高像素点
    def parallel_line_through_point(a, b, c):
        # 计算AB向量的分量
        ab_x = b[0] - a[0]
        ab_y = b[1] - a[1]

        # 平行线向量的分量与AB向量的分量一致
        cd_x = ab_x
        cd_y = ab_y

        d = (c[0] + cd_x, c[1] + cd_y)
        return d

    # 读取分割掩码图
    mask = cv2.imread(mask_path, 0)  # 以灰度模式读取图像
    height, width = mask.shape

    # 计算线段的中点
    midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    # 找到最大白色部分的边界
    _, thresholded = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # 计算垂直平分线的两个端点
    pt1, pt2 = perpendicular_points(midpoint, point1, point2)
    # 过晶状体最高点且平行于垂直平分线的直线
    heightest_point = get_highest_point(mask_path)
    print("heightest_point: ", heightest_point)
    pt3 = parallel_line_through_point(pt1, pt2, heightest_point)
    print(pt3, heightest_point)
    line = LineString([pt3, heightest_point])

    points_reshaped = [point[0] for point in largest_contour]  # 移除额外的维度
    polygon = Polygon(points_reshaped)

    # 可视化线段、白色部分的边界和交点
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')  # 线段
    plt.plot([pt3[0], heightest_point[0]], [pt3[1], heightest_point[1]], 'b-')  # 经过最高点且 平行于“垂直平分线
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.show()
    # 如果直线与多边形相交
    if line.intersects(polygon):
        # 计算相交部分
        intersection = line.intersection(polygon)
        # 检查相交结果的几何类型
        if intersection.geom_type == 'LineString':
            # 如果相交结果是线段，计算其长度
            intersection_length = intersection.length
            print("相交线段的长度:", intersection_length)
        else:
            # 如果相交结果不是线段，输出提示信息并设置长度为0
            print("相交部分不是线段。")
            intersection_length = 0
    else:
        # 如果直线与多边形不相交，输出提示信息并设置长度为0
        print("直线与多边形没有相交。")
        intersection_length = 0

    return intersection_length


def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open(out_csv_path, 'w', newline='') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)
        fieldnames = csv_reader.fieldnames + ['Intersection Length']
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)

        # 写入 CSV 文件的标题行
        csv_writer.writeheader()
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    # 从文件名中提取图像名称
                    image_name = os.path.splitext(file)[0]

                    # 查找匹配图像名称的行
                    for row in csv_reader:
                        if image_name in row['文件名']:
                            print('文件名--', row['文件名'], '   image_name--', image_name)
                            # 提取坐标点

                            img = cv2.imread(image_path)
                            height, width, channels = img.shape
                            point1 = (float(row['左眼X']) * width, float(row['左眼Y']) * height)
                            point2 = (float(row['右眼X']) * width, float(row['右眼Y']) * height)
                            print('point1', point1, 'point2', point2)

                            # 计算交点长度
                            intersection_length = find_intersection(image_path, point1, point2)

                            # 更新行数据
                            row['Intersection Length'] = intersection_length

                            # 写入更新后的行到新的 CSV 文件
                            csv_writer.writerow(row)

                            break  # 停止在 CSV 文件中继续查找匹配的行

                    # 将原始 CSV 读取器的位置重置为文件开头，以便下一次迭代时重新开始读取
                    csvfile.seek(0)

        print("New CSV file created: new_output.csv")


# 调用函数处理图像文件并更新 CSV 文件
# root_directory = '/home/liujia/project/measurement/baineizhang/晶状体/jingzhuangti'  # 根文件夹路径
# csv_path = '/home/liujia/project/measurement/cataract.csv'  # CSV 文件路径

root_directory = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_img_2_jzt'  # 根文件夹路径
csv_path = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_1011_all.csv'  # CSV 文件路径
out_csv_path = 'MyCSV/LT.csv'
process_images(root_directory, csv_path)