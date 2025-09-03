# 批量测量LV LV LV LV LV LV LV
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv
from PIL import Image


# 点x0，y0到直线的最短距离
def calculate_distance_to_line(x1, y1, x2, y2, x0, y0):
    distance = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# 计算两点与矩阵的交点
def calculate_intersection(pt1, pt2, pt3, pt4):
    # 直线两点
    x1, y1 = pt1
    x2, y2 = pt2

    # 矩形左上角和右下角
    x3, y3 = pt3
    x4, y4 = pt4

    # 直线方程参数
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    intercept = y1 - slope * x1

    # 定义函数来判断交点
    def check_intersection(x, y, xmin, xmax, ymin, ymax):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
        return False

        # 计算与左边相交

    if slope != float('inf'):
        y_left = slope * x3 + intercept
        if check_intersection(x3, y_left, x3, x3, y3, y4):
            yield (x3, y_left)

            # 计算与右边相交
    if slope != float('inf'):
        y_right = slope * x4 + intercept
        if check_intersection(x4, y_right, x4, x4, y3, y4):
            yield (x4, y_right)

            # 计算与上边相交
    if slope != 0:
        x_top = (y3 - intercept) / slope
        if check_intersection(x_top, y3, x3, x4, y3, y3):
            yield (x_top, y3)

            # 计算与下边相交
    if slope != 0:
        x_bottom = (y4 - intercept) / slope
        if check_intersection(x_bottom, y4, x3, x4, y4, y4):
            yield (x_bottom, y4)

        # 得到晶状体与中线交点


def find_intersection(mask_path, point1, point2):
    # 计算垂直平分线的两个点
    def perpendicular_points(mid, p1, p2):
        vector = np.array(p2) - np.array(p1)
        perp_vector = np.array([-vector[1], vector[0]])
        midpoint_vector = np.array(mid)
        pt1 = midpoint_vector + perp_vector
        pt2 = midpoint_vector - perp_vector
        return pt1, pt2

    # 像素点最高点
    # 找到晶状体最上角顶点 因为在分割掩码中白色为255即最大
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
        for y in range(height):
            for x in range(width):
                # 获取像素值
                pixel_value = image_gray.getpixel((x, y))
                # 如果当前像素值大于0且大于最高像素值，则更新最高像素值和最高点坐标
                if pixel_value > 0 and pixel_value > highest_pixel_value:
                    highest_pixel_value = pixel_value
                    highest_point = (x, y)
                    # print("now highest_point: ",highest_point,'  (x,y)',(x,y))
        return highest_point

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

    points_reshaped = [point[0] for point in largest_contour]  # 移除额外的维度
    polygon = Polygon(points_reshaped)
    # for i in points_reshaped:
    #     print('+++++++ points_reshaped +++++++',i)
    intersections = list(calculate_intersection(pt1, pt2, (1, 1), (width - 10, height - 10)))
    pt3, pt4 = intersections
    # print("直线和矩形的交点的坐标:", intersections)

    line = LineString([pt3, pt4])
    point = (66, 66)
    if line.intersects(polygon):
        intersection = line.intersection(polygon)

        if intersection.is_empty:
            point = (66, 66)
            print("没有交点")
        elif intersection.geom_type == 'Point':
            # 如果只有一个交点
            point = intersection.coords[0]
            print("交点坐标:", intersection.coords[0])
        elif intersection.geom_type == 'MultiPoint':
            # 如果有多个交点
            print("交点坐标:")
            point = intersection.coords[0]
            for point in intersection.coords:
                print(point)
        elif intersection.geom_type == 'LineString':

            point = intersection.coords[0]
            print("交点坐标:", intersection.coords[0])
        else:
            # 理论上不应该发生，因为直线与多边形的交集只可能是点或多点
            # point=(66,66)
            print("未知的交集类型:", intersection.geom_type)
            # point=intersection.coords[0]
            point = get_highest_point(mask_path)
            # for point in intersection.coords:
            #     print('the val is',point)
    else:
        print("The line does not intersect with the polygon.")
        point = get_highest_point(mask_path)
        # point=(66,66)
    return point, pt3, pt4


def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open(out_csv_path, 'w', newline='') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)
        fieldnames = csv_reader.fieldnames + ['Intersection Length']
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)
        img_num, now_img_num = 400, 0  # 可视乎的图片数量
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
                            # print('point1',point1,'point2',point2)
                            # 计算LV
                            point0, pt1, pt2 = find_intersection(image_path, point1, point2)
                            # print('pt1:',pt1,'--- pt2:', pt2)
                            LV = calculate_distance_to_line(point1[0], point1[1], point2[0], point2[1], point0[0],
                                                            point0[1])

                            if (now_img_num < img_num):
                                # 可视化线段、白色部分的边界和交点
                                now_img_num += 1
                                plt.figure(figsize=(8, 6))
                                plt.imshow(img, cmap='gray')
                                plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')  # 线段
                                # plt.plot([point0[0], point2[0]], [point0[1], point2[1]], 'bo-')  # 线段
                                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'go-')  # 线段

                                plt.plot([point0[0], 1 + point0[0]], [point0[1], 1 + point0[1]], 'y*-',
                                         linewidth=1.5)  # 点
                                plt.plot([pt1[0], 1 + pt1[0]], [pt1[1], 1 + pt1[1]], 'r*-')  # 点
                                plt.plot([pt2[0], 1 + pt2[0]], [pt2[1], 1 + pt2[1]], 'r*-')  # 点
                                # plt.scatter(pt1[0], pt1[1], color='green', label='Point 1')  # 画出pt1，颜色为green
                                # plt.scatter(pt2[0], pt2[1], color='blue', label='Point 2') # 画出pt2，颜色为蓝色
                                # ax = plt.gca()
                                # ax.scatter([point0[0]], [point0[1]], color='red', marker='*',linewidth=1.25) # 画出pt3，颜色为red色
                                # #ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
                                plt.xlim(0, width)
                                plt.ylim(height, 0)
                                plt.show()

                            # 更新行数据
                            row['Intersection Length'] = LV

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
out_csv_path = 'MyCSV/LV.csv'
process_images(root_directory, csv_path)