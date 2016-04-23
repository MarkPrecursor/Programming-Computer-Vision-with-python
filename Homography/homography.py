# -*- coding: utf-8 -*-
"""
*图像到图像的变换，算法练习
*创建于2016.4.22
*作者：Mark
"""
import numpy


def normalize(points):
    """在齐次坐标意义下，对点集进行归一化，使最后一行为1"""
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """将点集(dim*n的数组)转化为齐次坐标表示"""
    return numpy.vstack((points, numpy.ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    '''使用直接线性变换，计算单应性矩阵H, 使fp映射到tp. 点自动进行归一化
    '''
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化
    # 映射起始点
    m = numpy.mean(fp[:2], axis=1)
    maxstd = numpy.max(numpy.std(fp[:2], axis=1)) + 1e-9
    C1 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C1[0, 2] = -m[0] / maxstd
    C1[1, 2] = -m[1] / maxstd
    fp = numpy.dot(C1, fp)

    # 映射对应点
    m = numpy.mean(tp[:2], axis=1)
    maxstd = numpy.max(numpy.std(tp[:2], axis=1)) + 1e-9
    C2 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C2[0, 2] = -m[0] / maxstd
    C2[1, 2] = -m[1] / maxstd
    tp = numpy.dot(C2, tp)

    # 创建用于线性方法的矩阵，对于每个对应对，在矩阵中会出现两行数值
    correspondences_count = fp.shape[1]
    A = numpy.zeros((2 * correspondences_count, 9))
    for i in range(correspondences_count):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                        tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = numpy.linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = numpy.dot(numpy.linalg.inv(C2), numpy.dot(H, C1))      # 反归一化

    return H / H[2, 2]  # 归一化，然后返回


def Haffine_from_points(fp, tp):
    '''计算仿射变换的单应性矩阵H，使得tp是由fp经过仿射变换得到的'''
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化
    # 映射起始点
    m = numpy.mean(fp[:2], axis=1)
    maxstd = numpy.max(numpy.std(fp[:2], axis=1)) + 1e-9
    C1 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C1[0, 2] = -m[0] / maxstd
    C1[1, 2] = -m[1] / maxstd
    fp_cond = numpy.dot(C1, fp)

    # 映射对应点
    m = numpy.mean(tp[:2], axis=1)
    maxstd = numpy.max(numpy.std(tp[:2], axis=1)) + 1e-9
    C2 = numpy.diag([1/maxstd, 1/maxstd, 1])
    C2[0, 2] = -m[0] / maxstd
    C2[1, 2] = -m[1] / maxstd
    tp_cond = numpy.dot(C2, tp)

    # 因为归一化之后点的均值为0，所以平移量为0
    A = numpy.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = numpy.linalg.svd(A.T)
    # 创建矩阵B和C
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = numpy.concatenate((numpy.dot(C, numpy.linalg.pinv(B)), numpy.zeros((2, 1))), axis=1)
    H = numpy.vstack((tmp2, [0, 0, 1]))

    H = numpy.dot(numpy.linalg.inv(C2), numpy.dot(H, C1))  # 反归一化
    return H / H[2, 2]  # 归一化，然后返回


if __name__ == "__main__":
    '''计算直接线性变换的单应性矩阵需要四个点对，同时是因为在齐次坐标系所以每个点有三个维度'''
    fp = numpy.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T
    tp = numpy.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T
    print H_from_points(fp, tp)
    '''计算仿射变换的单应性矩阵H需要三个点对'''
    fp = numpy.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]).T
    tp = numpy.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]]).T
    print Haffine_from_points(fp, tp)
    """得到的单应性矩阵均是3*3的矩阵，对于要变换的每一个齐次坐标下的点x,映射到另外一个齐次坐标x'"""

    # 下面是采用OpenCV的仿射变换函数计算仿射变换
    import cv2
    # import matplotlib.pyplot as plt

    img = cv2.imread("empire.jpg", cv2.IMREAD_COLOR)
    rows, cols, ch = img.shape
    # 注意这里CV2中采用的不是齐次坐标
    pts1 = numpy.float32([[50, 50],  [200, 50], [50, 200]])
    pts2 = numpy.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('Input', img)
    cv2.imshow('Output', dst)
    cv2.waitKey()

    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
