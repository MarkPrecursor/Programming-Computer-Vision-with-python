# -*- coding: utf-8 -*-
"""
*拼接图像算法实验
*创建于2016.5.3
*作者：Mark
"""
import tic
import numpy
import homography
from scipy import ndimage


def transf(H, p):
    """用于geometric_transform的单应性变换"""
    p2 = numpy.dot(H, [p[0], p[1], 1])
    return (p2[0] / p2[2], p2[1] / p2[2])

def panorama(H, fromim, toim, padding=2400, delta=2400, alpha=1):
    """使用RANSAC估计得到的单应性矩阵H，协调两幅图像，创建水平全景图
    结果为一幅和toim具有相同高度的图像，padding制定填充像素的数目，delta制定额外的平移量"""
    # 检查是否是彩色图像
    is_color = len(fromim.shape) == 3

    if H[1, 2] < 0:  # fromim在右边
        if is_color:
            # 在目标图像的右边填充0
            toim_t = numpy.hstack((toim, numpy.zeros((toim.shape[0], padding, 3))))
            formim_t = numpy.zeros((toim.shape[0], toim.shape[1]+padding, toim.shape[2]))
            for col in range(3):
                formim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
                                      transf(toim.shape[0], toim.shape[1]+padding))
        else:
            # 在目标的右边填充0
            toim_t = numpy.hstack((toim, numpy.zeros((toim.shape[0], padding))))
            formim_t = ndimage.geometric_transform(fromim, transf(toim.shape[0], toim.shape[1]+padding))
    else:  # fromim在左边
        # 为了补偿填充效果，在左边加入平移量
        H_delta = numpy.array([[1, 0, 0],
                               [0, 1, -delta],
                               [0, 0, 1]])
        H = numpy.dot(H, H_delta)

        if is_color:
            # 在目标图像的左边填充0
            toim_t = numpy.hstack((numpy.zeros((toim.shape[0], padding, 3)), toim))
            formim_t = numpy.zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
            for col in range(3):
                formim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
                                                                  transf(toim.shape[0], toim.shape[1] + padding))
        else:
            # 在目标的左边填充0
            toim_t = numpy.hstack((numpy.zeros((toim.shape[0], padding)), toim))
            formim_t = ndimage.geometric_transform(fromim, transf(toim.shape[0], toim.shape[1] + padding))

    # 协调后返回(将formim放置在toim上)
    if is_color:
        alpha = ((formim_t[:, :, 0] * fromim[:, :, 1] * fromim[:, :, 2]) > 0)
        for col in range(3):
            toim_t[:, :, col] = formim_t[:, :, col] * alpha + toim_t[:, : col] * (1 - alpha)
    else:
        alpha = (formim_t > 0)
        toim_t = formim_t * alpha + toim_t * (1 - alpha)

    return toim_t

if __name__ == "__main__":
    import cv2
    import ORB_Match
    img1 = cv2.imread('5.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('6.jpg', cv2.IMREAD_COLOR)

    # 记录程序运行时间
    e1 = cv2.getTickCount()
    # your code execution
    kp1, kp2, Matched = ORB_Match.ORB_match(img1, img2)

    def convert_points(j, matches):
        """将匹配转换成齐次坐标点的函数"""
        ndx = matches[j].nonzero()[0]
        fp = homography.make_homog(kp1[ndx, :2].T)
        ndx2 = [int(matches[j][i]) for i in ndx]
        tp = homography.make_homog(kp2[ndx2, :2].T)
        return fp, tp


    model = homography.RansacModel()

    fp, tp = convert_points(1)
    H_12 = homography.H_from_ransac(fp, tp, model)[0]  # img1到img2的单应性矩阵
    img_12 = panorama(H_12, img1, img2, 2000, 2000)

    # Detect_and_Draw()
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print "Processing time is %f s" % time