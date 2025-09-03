import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv
from math import sqrt, isclose


# --- 辅助几何计算函数 ---

def _perpendicular_points(mid, p1, p2, line_extension_factor):
    """
    计算通过中点且垂直于p1-p2线段的延长线的两个端点。
    Args:
        mid (tuple): 线段p1-p2的中点 $(x, y)$。
        p1 (tuple): 线段的第一个端点 $(x, y)$。
        p2 (tuple): 线段的第二个端点 $(x, y)$。
        line_extension_factor (float): 延长线的长度因子，确保穿过整个图像。
    Returns:
        tuple: $(pt1, pt2)$ 垂直延长线的两个端点 (numpy array)。
    """
    vector = np.array(p2) - np.array(p1)
    perp_vector = np.array([-vector[1], vector[0]])  # 垂直向量
    midpoint_vector = np.array(mid)
    pt1 = midpoint_vector + perp_vector * line_extension_factor
    pt2 = midpoint_vector - perp_vector * line_extension_factor
    return pt1, pt2


def get_highest_point_from_mask(mask_np_array):
    """
    找到掩码图像中白色区域（像素值>0）的最高点（最小Y坐标）。
    Args:
        mask_np_array (numpy.ndarray): 灰度掩码图像 (0-255)。
    Returns:
        tuple: $(x, y)$ 像素坐标，如果未找到白色像素则返回 None。
    """
    white_pixels = np.argwhere(mask_np_array > 0)
    if white_pixels.size == 0:
        return None
    highest_y = np.min(white_pixels[:, 0])
    highest_y_pixels = white_pixels[white_pixels[:, 0] == highest_y]
    highest_point_yx = highest_y_pixels[np.argmin(highest_y_pixels[:, 1])]  # Smallest x among highest y
    return (highest_point_yx[1], highest_point_yx[0])  # Return $(x, y)$


# --- ACD 测量函数 ---
def calculate_acd_and_visualization_points(mask_image_path, point1_px, point2_px):
    """
    计算前房深度（ACD）并返回用于可视化的交点和垂直线。
    """
    mask = cv2.imread(mask_image_path, 0)
    if mask is None:
        # print(f"错误：无法加载前房掩码图像：{mask_image_path}")
        return 0, None, None, None

    height, width = mask.shape
    midpoint = ((point1_px[0] + point2_px[0]) / 2, (point1_px[1] + point2_px[1]) / 2)
    line_extension_factor = max(width, height) * 2
    pt1_perp, pt2_perp = _perpendicular_points(midpoint, point1_px, point2_px, line_extension_factor)

    line_perpendicular = LineString([pt1_perp, pt2_perp])
    perpendicular_line_points = [(int(pt1_perp[0]), int(pt1_perp[1])), (int(pt2_perp[0]), int(pt2_perp[1]))]

    _, thresholded = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("在前房掩码中未找到任何轮廓。")
        return 0, None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    points_reshaped = [point[0] for point in largest_contour]
    polygon = Polygon(points_reshaped)

    acd_length = 0
    acd_start_point = None
    acd_end_point = None

    if line_perpendicular.intersects(polygon):
        intersection = line_perpendicular.intersection(polygon)
        if intersection.geom_type == 'LineString':
            acd_length = intersection.length
            coords = list(intersection.coords)
            if len(coords) == 2:
                acd_start_point = (int(coords[0][0]), int(coords[0][1]))
                acd_end_point = (int(coords[1][0]), int(coords[1][1]))
        # else:
        # print("ACD交点不是线段（例如，一个点或多个线段）。")
    # else:
    # print("ACD测量线未与前房多边形相交。")

    return acd_length, acd_start_point, acd_end_point, perpendicular_line_points


# --- LT 测量函数 ---
def calculate_lt_and_visualization_points(lens_mask_path, point1_px, point2_px):
    """
    计算晶状体深度（LT）并返回用于可视化的交点和平行线。
    """
    mask = cv2.imread(lens_mask_path, 0)
    if mask is None:
        # print(f"错误：无法加载晶状体掩码图像：{lens_mask_path}")
        return 0, None, None, None, None

    height, width = mask.shape
    highest_point = get_highest_point_from_mask(mask)
    if highest_point is None:
        # print("在晶状体掩码中未找到白色像素（晶状体）。")
        return 0, None, None, None, None

    midpoint_ss = ((point1_px[0] + point2_px[0]) / 2, (point1_px[1] + point2_px[1]) / 2)
    line_extension_factor = max(width, height) * 2
    perp_line_p1_ref, perp_line_p2_ref = _perpendicular_points(midpoint_ss, point1_px, point2_px, line_extension_factor)

    perp_direction_vector = np.array(perp_line_p2_ref) - np.array(perp_line_p1_ref)
    norm_perp_direction_vector = np.linalg.norm(perp_direction_vector)

    lt_line_p1 = np.array(highest_point) - perp_direction_vector * line_extension_factor
    lt_line_p2 = np.array(highest_point) + perp_direction_vector * line_extension_factor
    if norm_perp_direction_vector != 0:  # Normalize only if not zero vector
        lt_line_p1 = np.array(highest_point) - (
                perp_direction_vector / norm_perp_direction_vector) * line_extension_factor
        lt_line_p2 = np.array(highest_point) + (
                perp_direction_vector / norm_perp_direction_vector) * line_extension_factor

    line_parallel_to_perp = LineString([lt_line_p1, lt_line_p2])
    parallel_line_points = [(int(lt_line_p1[0]), int(lt_line_p1[1])), (int(lt_line_p2[0]), int(lt_line_p2[1]))]

    _, thresholded = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("在晶状体掩码中未找到任何轮廓。")
        return 0, None, None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    points_reshaped = [point[0] for point in largest_contour]
    polygon = Polygon(points_reshaped)

    lt_length = 0
    lt_start_point = None
    lt_end_point = None

    if line_parallel_to_perp.intersects(polygon):
        intersection = line_parallel_to_perp.intersection(polygon)
        if intersection.geom_type == 'LineString':
            lt_length = intersection.length
            coords = list(intersection.coords)
            if len(coords) == 2:
                lt_start_point = (int(coords[0][0]), int(coords[0][1]))
                lt_end_point = (int(coords[1][0]), int(coords[1][1]))
        # else:
        # print("LT交点不是线段（例如，一个点或多个线段）。")
    # else:
    # print("LT测量线未与晶状体多边形相交。")

    return lt_length, lt_start_point, lt_end_point, parallel_line_points, highest_point


# --- ACA 测量函数 (改编自用户提供的代码) ---
def calculate_aca_and_visualization_data(mask_image_path, ss_point_px, radius_px, epsilon=8):
    """
    计算前房角面积（ACA）并返回相关可视化数据。
    Args:
        mask_image_path (str): 前房掩码图像的路径。
        ss_point_px (tuple): 巩膜突点 $(x, y)$ 作为圆心。
        radius_px (int): 圆的半径。
        epsilon (int): 圆周交点查找的容差。
    Returns:
        tuple: $(aca\_area, b\_point, bottom\_most\_point, perp\_line\_endpoints, region\_contour, circle\_center, circle\_radius)$
               如果计算失败，返回 $(0, None, None, None, None, None, None)$。
    """
    mask_np_array = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_np_array is None:
        # print(f"错误：无法加载ACA掩码图像：{mask_image_path}")
        return 0, None, None, None, None, None, None

    height, width = mask_np_array.shape
    center = ss_point_px

    # 1. 找到圆周与白色区域的交点 (b_point)
    b_point = None
    for y_scan in range(max(0, center[1] - radius_px), min(height, center[1] + radius_px + 1)):
        for x_scan in range(max(0, center[0] - radius_px), min(width, center[0] + radius_px + 1)):
            if abs((x_scan - center[0]) ** 2 + (y_scan - center[1]) ** 2 - radius_px ** 2) <= epsilon ** 2 and \
                    mask_np_array[y_scan, x_scan] == 255:
                b_point = (x_scan, y_scan)
                break
        if b_point:
            break

    if b_point is None:
        # print(f"ACA: 未找到圆周与掩码的交点 (b_point) for center {center}, radius {radius_px}.")
        return 0, None, None, None, None, None, None

    # 2. 计算过 b_point 且垂直于 center-b_point 的直线
    dx_cb = b_point[0] - center[0]
    dy_cb = b_point[1] - center[1]

    perp_vec_x_dir = dy_cb
    perp_vec_y_dir = -dx_cb

    norm_perp_vec_dir = np.linalg.norm([perp_vec_x_dir, perp_vec_y_dir])
    if norm_perp_vec_dir == 0:
        # print(f"ACA: 垂直向量长度为零 for center {center}.")
        return 0, None, None, None, None, None, None
    perp_vec_x_dir /= norm_perp_vec_dir
    perp_vec_y_dir /= norm_perp_vec_dir

    line_length_for_drawing = max(width, height)
    perp_line_endpoints = [
        (int(b_point[0] + perp_vec_x_dir * line_length_for_drawing),
         int(b_point[1] + perp_vec_y_dir * line_length_for_drawing)),
        (int(b_point[0] - perp_vec_x_dir * line_length_for_drawing),
         int(b_point[1] - perp_vec_y_dir * line_length_for_drawing))
    ]

    # 3. 找到垂线与白色区域的最底部交点 (bottom_most_point)
    bottom_most_point = (0, 0)
    white_mask = (mask_np_array == 255)
    y_coords, x_coords = np.indices(mask_np_array.shape)

    A_line = perp_vec_y_dir
    B_line = -perp_vec_x_dir
    C_line = -A_line * b_point[0] - B_line * b_point[1]

    line_pixels_mask = np.abs(A_line * x_coords + B_line * y_coords + C_line) < 2
    valid_points_on_line = line_pixels_mask & white_mask

    if np.any(valid_points_on_line):
        bottom_most_y = np.max(y_coords[valid_points_on_line])
        corresponding_x = x_coords[valid_points_on_line][np.argmax(y_coords[valid_points_on_line])]
        bottom_most_point = (int(corresponding_x), int(bottom_most_y))
    # else:
    # print(f"ACA: 未找到垂线与掩码的最底部交点 for center {center}.")
    # return 0, None, None, None, None, None, None

    # 4. 计算 ACA 区域及其轮廓
    # ACA区域是前房掩码中，位于垂线“外侧”（远离巩膜突中心）的部分
    region_mask = (A_line * x_coords + B_line * y_coords + C_line > 0) & white_mask

    aca_area = np.sum(region_mask)
    region_mask_uint8 = (region_mask * 255).astype(np.uint8)
    contours_aca, _ = cv2.findContours(region_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_contour = None
    if contours_aca:
        region_contour = max(contours_aca, key=cv2.contourArea)
    # else:
    # print(f"ACA: 未找到区域轮廓 for center {center}.")
    # return 0, None, None, None, None, None, None

    return aca_area, b_point, bottom_most_point, perp_line_endpoints, region_contour, center, radius_px


# --- 联合可视化函数 ---
def visualize_combined_measurements(
        oct_image_path,
        anterior_chamber_mask_path,
        lens_mask_path,
        point1_norm,  # Left SS
        point2_norm,  # Right SS
        aca_radius_px=105  # Default to 0.75mm radius for ACA
):
    """
    执行ACD、LT和ACA测量，并在AS-OCT图像上进行可视化。

    Args:
        oct_image_path (str): 原始AS-OCT图像的路径。
        anterior_chamber_mask_path (str): 前房掩码图像的路径（用于ACD和ACA）。
        lens_mask_path (str): 晶状体掩码图像的路径（用于LT）。
        point1_norm (tuple): 第一个参考点的 $(x_{norm}, y_{norm})$ 归一化坐标（0-1范围）。
        point2_norm (tuple): 第二个参考点的 $(x_{norm}, y_{norm})$ 归一化坐标（0-1范围）。
        aca_radius_px (int): ACA测量时圆的半径（像素）。

    Returns:
        numpy.ndarray: 包含所有可视化结果的图像数据 (BGR格式)。
        tuple: $(acd\_length, lt\_length, left\_aca\_area, right\_aca\_area)$.
               如果测量失败，返回0。
    """
    oct_img = cv2.imread(oct_image_path)
    if oct_img is None:
        print(f"错误：无法加载AS-OCT图像：{oct_image_path}")
        return None, (0, 0, 0, 0)

    height, width = oct_img.shape[:2]
    image_dims = (height, width)  # Keep image_dims for potential future use, though not used by current functions

    # 将归一化坐标转换为像素坐标
    point1_px = (int(point1_norm[0] * width), int(point1_norm[1] * height))
    point2_px = (int(point2_norm[0] * width), int(point2_norm[1] * height))

    # --- ACD 计算和可视化点获取 ---
    acd_length, acd_start_point, acd_end_point, acd_perp_line_points = \
        calculate_acd_and_visualization_points(anterior_chamber_mask_path, point1_px, point2_px)

    # --- LT 计算和可视化点获取 ---
    lt_length, lt_start_point, lt_end_point, lt_parallel_line_points, highest_point_lt = \
        calculate_lt_and_visualization_points(lens_mask_path, point1_px, point2_px)

    # --- ACA 计算和可视化数据获取 ---
    left_aca_area, left_b_point, left_bottom_most_point, left_aca_perp_line_endpoints, left_aca_region_contour, left_aca_circle_center, left_aca_circle_radius = \
        calculate_aca_and_visualization_data(anterior_chamber_mask_path, point1_px, aca_radius_px)
    right_aca_area, right_b_point, right_bottom_most_point, right_aca_perp_line_endpoints, right_aca_region_contour, right_aca_circle_center, right_aca_circle_radius = \
        calculate_aca_and_visualization_data(anterior_chamber_mask_path, point2_px, aca_radius_px)

    # 创建原始OCT图像的副本用于绘图
    display_img = oct_img.copy()
    # 创建一个用于半透明填充的叠加层
    overlay = display_img.copy()
    alpha = 0.3  # 半透明度

    # --- 可视化设置 ---
    REF_POINT_COLOR = (0, 255, 255)  # 黄色 (巩膜突点)
    REF_LINE_COLOR = (0, 255, 255)  # 黄色 (巩膜突连线)
    PERP_LINE_COLOR = (255, 0, 0)  # 蓝色 (ACD的垂直辅助线)
    ACD_COLOR = (255, 0, 255)  # 洋红色 (ACD测量线)
    LT_COLOR = (0, 255, 0)  # 绿色 (LT测量线)
    ACA_COLOR = (0, 165, 255)  # 橙色 (ACA相关)
    LT_AUX_POINT_COLOR = (0, 255, 255)  # 黄色 (晶状体最高点 for LT)
    LT_AUX_LINE_COLOR = (0, 255, 255)  # 黄色 (LT的平行辅助线)
    TEXT_COLOR = (255, 255, 255)  # 白色 (在OCT图像上对比度更好)
    THICKNESS_MAIN = 3  # 主要测量线（ACD, LT, ACA轮廓）的粗细
    THICKNESS_AUX = 1  # 辅助线（参考点、参考线、垂直线、圆）的粗细
    ARROW_SIZE = 10  # 箭头大小
    FONT_SCALE = 0.7  # 文本大小

    # --- 绘制所有辅助线和主要测量线 (不包括填充区域) ---

    # 1. 绘制巩膜突点
    cv2.circle(display_img, point1_px, 5, REF_POINT_COLOR, -1)
    cv2.circle(display_img, point2_px, 5, REF_POINT_COLOR, -1)

    # 2. 绘制巩膜突连线 (作为参考线)
    cv2.line(display_img, point1_px, point2_px, REF_LINE_COLOR, THICKNESS_AUX)

    # 3. 绘制ACD的完整的垂直线（提供上下文）
    if acd_perp_line_points:
        cv2.line(display_img, acd_perp_line_points[0], acd_perp_line_points[1], PERP_LINE_COLOR, THICKNESS_AUX,
                 cv2.LINE_AA)

    # 4. 绘制ACD测量线和双箭头
    if acd_start_point and acd_end_point:
        cv2.line(display_img, acd_start_point, acd_end_point, ACD_COLOR, THICKNESS_MAIN, cv2.LINE_AA)
        dx, dy = acd_end_point[0] - acd_start_point[0], acd_end_point[1] - acd_start_point[1]
        angle = np.arctan2(dy, dx)
        for p_end, p_start in [(acd_end_point, acd_start_point), (acd_start_point, acd_end_point)]:
            if p_end == acd_end_point:
                p1_x = int(p_end[0] - ARROW_SIZE * np.cos(angle - np.pi / 6))
                p1_y = int(p_end[1] - ARROW_SIZE * np.sin(angle - np.pi / 6))
                p2_x = int(p_end[0] - ARROW_SIZE * np.cos(angle + np.pi / 6))
                p2_y = int(p_end[1] - ARROW_SIZE * np.sin(angle + np.pi / 6))
            else:
                p1_x = int(p_end[0] + ARROW_SIZE * np.cos(angle - np.pi / 6))
                p1_y = int(p_end[1] + ARROW_SIZE * np.sin(angle - np.pi / 6))
                p2_x = int(p_end[0] + ARROW_SIZE * np.cos(angle + np.pi / 6))
                p2_y = int(p_end[1] + ARROW_SIZE * np.sin(angle + np.pi / 6))
            cv2.line(display_img, p_end, (p1_x, p1_y), ACD_COLOR, THICKNESS_MAIN, cv2.LINE_AA)
            cv2.line(display_img, p_end, (p2_x, p2_y), ACD_COLOR, THICKNESS_MAIN, cv2.LINE_AA)

    # 5. 绘制LT的辅助线 (晶状体最高点和平行线)
    if highest_point_lt:
        cv2.circle(display_img, highest_point_lt, 5, LT_AUX_POINT_COLOR, -1)
    if lt_parallel_line_points:
        cv2.line(display_img, lt_parallel_line_points[0], lt_parallel_line_points[1], LT_AUX_LINE_COLOR, THICKNESS_AUX,
                 cv2.LINE_AA)

    # 6. 绘制LT测量线和双箭头
    if lt_start_point and lt_end_point:
        cv2.line(display_img, lt_start_point, lt_end_point, LT_COLOR, THICKNESS_MAIN, cv2.LINE_AA)
        dx, dy = lt_end_point[0] - lt_start_point[0], lt_end_point[1] - lt_start_point[1]
        angle = np.arctan2(dy, dx)
        for p_end, p_start in [(lt_end_point, lt_start_point), (lt_start_point, lt_end_point)]:
            if p_end == lt_end_point:
                p1_x = int(p_end[0] - ARROW_SIZE * np.cos(angle - np.pi / 6))
                p1_y = int(p_end[1] - ARROW_SIZE * np.sin(angle - np.pi / 6))
                p2_x = int(p_end[0] - ARROW_SIZE * np.cos(angle + np.pi / 6))
                p2_y = int(p_end[1] - ARROW_SIZE * np.sin(angle + np.pi / 6))
            else:
                p1_x = int(p_end[0] + ARROW_SIZE * np.cos(angle - np.pi / 6))
                p1_y = int(p_end[1] + ARROW_SIZE * np.sin(angle - np.pi / 6))
                p2_x = int(p_end[0] + ARROW_SIZE * np.cos(angle + np.pi / 6))
                p2_y = int(p_end[1] + ARROW_SIZE * np.sin(angle + np.pi / 6))
            cv2.line(display_img, p_end, (p1_x, p1_y), LT_COLOR, THICKNESS_MAIN, cv2.LINE_AA)
            cv2.line(display_img, p_end, (p2_x, p2_y), LT_COLOR, THICKNESS_MAIN, cv2.LINE_AA)

    # 7. 绘制ACA (左侧) 的辅助线和填充区域
    if left_aca_area > 0 and left_aca_circle_center and left_aca_circle_radius:
        cv2.circle(display_img, left_aca_circle_center, left_aca_circle_radius, ACA_COLOR, THICKNESS_AUX, cv2.LINE_AA)
        if left_b_point:
            cv2.circle(display_img, left_b_point, 5, ACA_COLOR, -1)
        if left_aca_perp_line_endpoints:
            cv2.line(display_img, left_aca_perp_line_endpoints[0], left_aca_perp_line_endpoints[1], ACA_COLOR,
                     THICKNESS_AUX, cv2.LINE_AA)
        if left_bottom_most_point:
            cv2.circle(display_img, left_bottom_most_point, 5, ACA_COLOR, -1)
        if left_aca_region_contour is not None:
            cv2.fillPoly(overlay, [left_aca_region_contour], ACA_COLOR, cv2.LINE_AA)

    # 8. 绘制ACA (右侧) 的辅助线和填充区域
    if right_aca_area > 0 and right_aca_circle_center and right_aca_circle_radius:
        cv2.circle(display_img, right_aca_circle_center, right_aca_circle_radius, ACA_COLOR, THICKNESS_AUX, cv2.LINE_AA)
        if right_b_point:
            cv2.circle(display_img, right_b_point, 5, ACA_COLOR, -1)
        if right_aca_perp_line_endpoints:
            cv2.line(display_img, right_aca_perp_line_endpoints[0], right_aca_perp_line_endpoints[1], ACA_COLOR,
                     THICKNESS_AUX, cv2.LINE_AA)
        if right_bottom_most_point:
            cv2.circle(display_img, right_bottom_most_point, 5, ACA_COLOR, -1)
        if right_aca_region_contour is not None:
            cv2.fillPoly(overlay, [right_aca_region_contour], ACA_COLOR, cv2.LINE_AA)

    # 将半透明叠加层与原始图像混合
    display_img = cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0)

    # 在混合后的图像上绘制ACA区域的轮廓，确保其在填充区域之上
    if left_aca_area > 0 and left_aca_region_contour is not None:
        cv2.drawContours(display_img, [left_aca_region_contour], -1, ACA_COLOR, THICKNESS_MAIN, cv2.LINE_AA)
    if right_aca_area > 0 and right_aca_region_contour is not None:
        cv2.drawContours(display_img, [right_aca_region_contour], -1, ACA_COLOR, THICKNESS_MAIN, cv2.LINE_AA)

    # --- 绘制所有文本 ---
    # ACD文本
    if acd_start_point and acd_end_point:
        text_acd = f"ACD: {acd_length:.2f} px"
        text_acd_x = int((acd_start_point[0] + acd_end_point[0]) / 2) + 10
        text_acd_y = int((acd_start_point[1] + acd_end_point[1]) / 2) - 10
        cv2.putText(display_img, text_acd, (text_acd_x, text_acd_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR,
                    2, cv2.LINE_AA)

    # LT文本
    if lt_start_point and lt_end_point:
        text_lt = f"LT: {lt_length:.2f} px"
        text_lt_x = int((lt_start_point[0] + lt_end_point[0]) / 2) + 10
        text_lt_y = int((lt_start_point[1] + lt_end_point[1]) / 2) + 20
        cv2.putText(display_img, text_lt, (text_lt_x, text_lt_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 2,
                    cv2.LINE_AA)

    # 左侧ACA文本 (无单位)
    if left_aca_area > 0 and left_aca_circle_center:
        text_aca_left = f"L-ACA: {left_aca_area:.2f}"
        cv2.putText(display_img, text_aca_left, (left_aca_circle_center[0] + 10, left_aca_circle_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)

    # 右侧ACA文本 (无单位)
    if right_aca_area > 0 and right_aca_circle_center:
        text_aca_right = f"R-ACA: {right_aca_area:.2f}"
        cv2.putText(display_img, text_aca_right, (right_aca_circle_center[0] - 100, right_aca_circle_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)

    # 返回BGR格式的图像和所有测量结果
    return display_img, (acd_length, lt_length, left_aca_area, right_aca_area)


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 定义数据集根目录和输出目录 ---
    IMAGE_DATASET_ROOT = r"C:\srp_OCT\mask_compare\yjh_20"  # 你的图像数据集根目录
    MASK_DATASET_ROOT = r"C:\srp_OCT\mask_compare\mask\yjh_mask"  # 你的掩码数据集根目录
    OUTPUT_ROOT_DIR = r"C:\srp_OCT\mask_compare\visualization_results\yjh"  # 结果保存目录

    # ACA测量半径 (0.75mm 对应 105 像素，根据你的代码)
    ACA_RADIUS_750_PX = 105

    # 创建输出根目录（如果不存在）
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    # 遍历图像数据集中的每个疾病文件夹
    for disease_folder in os.listdir(IMAGE_DATASET_ROOT):
        disease_image_path = os.path.join(IMAGE_DATASET_ROOT, disease_folder)

        # 根据图像文件夹名，推断对应的掩码文件夹名
        mask_subfolder_name = disease_folder
        if disease_folder.lower().endswith('x5'):
            mask_subfolder_name = disease_folder[:-2]  # 移除 'X5'

        disease_mask_path = os.path.join(MASK_DATASET_ROOT, mask_subfolder_name)
        disease_output_path = os.path.join(OUTPUT_ROOT_DIR, disease_folder)

        if not os.path.isdir(disease_image_path):
            continue  # 跳过非文件夹项

        os.makedirs(disease_output_path, exist_ok=True)  # 为当前疾病创建输出文件夹

        print(f"\n--- 正在处理疾病类别: {disease_folder} (对应掩码文件夹: {mask_subfolder_name}) ---")

        # 构建当前疾病的巩膜突CSV路径
        ss_locate_csv_path = os.path.join(disease_mask_path, "ss_locate.csv")
        if not os.path.exists(ss_locate_csv_path):
            print(f"警告：未找到 '{mask_subfolder_name}' 的巩膜突CSV文件：{ss_locate_csv_path}。跳过此疾病类别。")
            continue

        # 读取巩膜突CSV文件到内存，方便查找
        ss_data = {}
        try:
            with open(ss_locate_csv_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    filename_key = row['filename'].strip()
                    ss_data[filename_key] = {
                        'leftX': float(row['leftX']),
                        'leftY': float(row['leftY']),
                        'rightX': float(row['rightX']),
                        'rightY': float(row['rightY'])
                    }
        except Exception as e:
            print(f"错误：读取 '{ss_locate_csv_path}' 时发生问题：{e}。跳过此疾病类别。")
            continue

        # 遍历当前疾病文件夹中的所有图像文件
        for image_file in os.listdir(disease_image_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                oct_image_full_path = os.path.join(disease_image_path, image_file)
                image_basename = os.path.splitext(image_file)[0]

                print(f"  正在处理图像: {image_file}")

                # 查找巩膜突坐标
                if image_basename not in ss_data:
                    print(f"    警告：图像 '{image_file}' 在CSV中没有巩膜突坐标。跳过。")
                    continue

                ss_coords = ss_data[image_basename]
                point1_normalized = (ss_coords['leftX'], ss_coords['leftY'])
                point2_normalized = (ss_coords['rightX'], ss_coords['rightY'])

                # 检查巩膜突坐标是否有效
                if not all(isinstance(coord, (int, float)) for coord in point1_normalized + point2_normalized):
                    print(f"    警告：图像 '{image_file}' 的巩膜突坐标无效。跳过。")
                    continue

                # 构建掩码文件路径
                anterior_chamber_mask_file = os.path.join(disease_mask_path, "qianfang", image_basename + ".png")
                lens_mask_file = os.path.join(disease_mask_path, "jingzhuangti", image_basename + ".png")

                if not os.path.exists(anterior_chamber_mask_file):
                    print(f"    警告：未找到前房掩码：{anterior_chamber_mask_file}。跳过图像。")
                    continue
                if not os.path.exists(lens_mask_file):
                    print(f"    警告：未找到晶状体掩码：{lens_mask_file}。跳过图像。")
                    continue

                # 调用联合可视化函数
                visualized_img, results = visualize_combined_measurements(
                    oct_image_full_path,
                    anterior_chamber_mask_file,
                    lens_mask_file,
                    point1_normalized,
                    point2_normalized,
                    aca_radius_px=ACA_RADIUS_750_PX
                )

                if visualized_img is not None:
                    # 保存可视化图像
                    output_image_name = f"{image_basename}_visualized.png"
                    output_image_path = os.path.join(disease_output_path, output_image_name)

                    # 使用matplotlib保存，以确保标题和轴标签正确渲染
                    plt.figure(figsize=(16, 14))
                    plt.imshow(cv2.cvtColor(visualized_img, cv2.COLOR_BGR2RGB))  # Matplotlib expects RGB
                    plt.title(
                        f"AS-OCT Measurement: ACD {results[0]:.2f} px, LT {results[1]:.2f} px, L-ACA {results[2]:.2f}, R-ACA {results[3]:.2f}")
                    plt.axis('off')
                    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close()  # 关闭图形，避免内存占用过高

                    print(f"    可视化图像已保存到: {output_image_path}")
                    print(
                        f"      ACD: {results[0]:.2f} px, LT: {results[1]:.2f} px, L-ACA: {results[2]:.2f}, R-ACA: {results[3]:.2f}")
                else:
                    print(f"    错误：图像 '{image_file}' 的可视化失败。")

    print("\n所有图像处理完成。")
