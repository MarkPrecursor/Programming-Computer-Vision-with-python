# -*- coding: utf-8 -*-
"""
*Harris角点检测代码，不用OpenCV
*创建于2016.4.11
*作者：Mark
"""
import matplotlib.pyplot as plt
import numpy
import Harris
import cv2


def Harris_match(img1, img2):
    """算法主程序"""
    # 下面执行一点颜色空间的转换
    r, g, b = cv2.split(img1)
    img1 = cv2.merge((b, g, r))
    r, g, b = cv2.split(img2)
    img2 = cv2.merge((b, g, r))
    G_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    G_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算并描述角点
    harrisim1 = Harris.Compute_Harris_Response(G_img1)
    filtered_coords1 = numpy.array(Harris.Get_Harris_Points(harrisim1), dtype=int)
    patches1 = Harris.get_descriptors(G_img1, filtered_coords1)

    harrisim2 = Harris.Compute_Harris_Response(G_img2)
    filtered_coords2 = numpy.array(Harris.Get_Harris_Points(harrisim2), dtype=int)
    patches2 = Harris.get_descriptors(G_img2, filtered_coords2)

    # Harris.Plot_harris_points(img1, filtered_coords1)
    # Harris.Plot_harris_points(img2, filtered_coords2)
    matches = Harris.match_twosided(patches1, patches2)

    plt.figure()
    plt.gray()
    Harris.plot_matches(img1, img2, filtered_coords1, filtered_coords2, matches, show_below=False)
    plt.show()

if __name__ == '__main__':
    img1 = cv2.imread('1.JPG', cv2.IMREAD_COLOR)
    img2 = cv2.imread('2.JPG', cv2.IMREAD_COLOR)

    Harris_match(img1, img2)
