# -*- coding: utf-8 -*-
"""
*Harris角点检测代码，不用OpenCV
*创建于2016.5.14
*作者：Mark
"""
import cv2
import math
import numpy
import scipy.ndimage.filters as filters


def Compute_Harris_Response(img, sigma=1.5):
    """对一副灰度图像，返回每个像素值表示Harris角点检测器响应函数值的图像"""
    # 计算导数：
    imx = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    imy = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

    # 计算Harris矩阵的分量:
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值和迹:
    Wdet = Wxx * Wyy - Wxy * 2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def Get_Harris_Points(Harris_img, min_dist=10, threshold=0.1, maxnum=2000):
    """ 从Harris图像中返回角点,
    *　min_dist为分割角点和图像边界的最小像素数目
    *　maxnum是返回的最大数目
    """
    # 寻找高于阈值的候选角点：
    corner_threshold = Harris_img.max() * threshold
    Harris_img_T = (Harris_img > corner_threshold) * 1
    # 得到候选角点的坐标:
    coords = numpy.array(Harris_img_T.nonzero()).T
    # 得到Harris响应值：
    Candidate_Value = [Harris_img[c[0], c[1]] for c in coords]
    # 对候选点按照响应值排序
    index = numpy.argsort(Candidate_Value)

    if index.size > maxnum:
        index = index[:maxnum]

    # 将可行点的位置保存到数组当中
    Allowed_locations = numpy.zeros(Harris_img.shape, dtype=numpy.uint8)
    Allowed_locations[min_dist:-min_dist,
                      min_dist:-min_dist] = 1  # 这一句是圈出一圈限定范围

    # 按照最小距离原则选择最佳的Harris点
    filtered_coords = []
    for i in index:
        if Allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            Allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
                              (coords[i, 0] - min_dist):(coords[i, 0] + min_dist)] = 0
    return filtered_coords


def Get_MainDirection(local):
    """对于当前位置计算主方向，输入的是局部图像"""
    shape = local.shape
    center = shape[0] / 2
    center_Value = float(local[center, center])
    d_vector = numpy.zeros(2, dtype=float)  # 记录向量总加和
    if shape[0] == shape[1]:
        for i in range(shape[0]):
            for j in range(shape[1]):
                Weight = (float(local[i, j]) - center_Value) / (0.01 
                    + math.pow((i - center + 1)**2 + (j - center + 3)**2, 0.25))
                vector = numpy.matrix((i - center, j - center), dtype=float)
                norm_2 = math.sqrt(vector * vector.T)
                if norm_2 != 0:
                    vector = Weight * vector / norm_2
                d_vector += numpy.array((vector[0, 0], vector[0, 1]))
        if d_vector[0] != 0:
            main_direction = math.degrees(math.atan(d_vector[1] / d_vector[0]))
            if d_vector[0] < 0:
                main_direction += 180
        else:
            main_direction = 90

        return main_direction
    else:
        print "Wrong shape"
        return -1.


def Calculate_Descripter(local, main_direction):
    """根据计算得到的主方向旋转图像并读取三进制特征"""
    shape = local.shape
    center = shape[0] / 2
    center_Value = local[center, center]
    M = cv2.getRotationMatrix2D((center, center), main_direction, 1)
    dst = cv2.warpAffine(local, M, shape)
    dst = dst[center - 5: center + 5 + 1, center - 5: center + 5 + 1]
    # T, dst = cv2.threshold(dst, center_Value, 1, cv2.THRESH_BINARY)
    desc = numpy.reshape(dst, 121)

    return desc


def Get_LBP_descriptors(img, filtered_coords, wid=5):
    """  建立有主方向的局部二进制描述子    """
    num = filtered_coords.__len__()  # 对应的点的个数,每个元素是一个坐标
    # 特征描述子，每个特征对应121维
    Desc = numpy.zeros((num, (wid * 2 + 1)**2), dtype=float)
    Main_directions = numpy.zeros(num, dtype=float)
    for i in range(num):
        (m, n) = (filtered_coords[i][0], filtered_coords[i][1])
        local = img[m - wid: m + wid + 1, n - wid: n + wid + 1]
        m_direction = Get_MainDirection(local)
        Main_directions[i] = m_direction

        r = int(round(5.5 * math.sqrt(2)) + 3)
        local = img[m - r: m + r + 1, n - r: n + r + 1]
        Desc[i] = Calculate_Descripter(local, m_direction)

    return Desc, Main_directions


def ncc(patch1, patch2):
    """返回两幅图片之间归一化的互相关"""
    d1 = (patch1 - numpy.mean(patch1)) / numpy.std(patch1)
    d2 = (patch2 - numpy.mean(patch2)) / numpy.std(patch2)
    return numpy.sum(d1 * d2) / (len(patch1) - 1)


def plot_features(img, locations, m_dir):
    '''将检测到的特征点绘制到图像上'''
    num = locations.__len__()
    for i in range(num):
        p = (locations[i][1], locations[i][0])
        cv2.circle(img, p, 5, (0, 255, 0), 2)
        theta = math.tan(m_dir[i] / 180 * 3.1415926)
        vector = numpy.matrix((1, theta))
        vector = (10 * vector / math.sqrt((vector * vector.T))).astype(int)
        Point = (p[0] + vector[0, 0], p[1] + vector[0, 1])
        cv2.line(img, p, Point, (255, 0, 128), 2)


def Harris_Detetct(img):
    """Harris特征检测，返回数组标记特征位置"""
    G_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    HarrisImg = Compute_Harris_Response(G_img)
    Filtered_coords = Get_Harris_Points(HarrisImg)
    m_Descs, m_directions = Get_LBP_descriptors(G_img, Filtered_coords)
    return Filtered_coords, m_Descs, m_directions


def match(Decs1, Decs2):
    dist_ratio = 0.68
    shape1 = Decs1.shape[0]
    shape2 = Decs2.shape[0]

    desc1 = numpy.matrix(Decs1, dtype=float)
    desc2 = numpy.matrix(Decs2, dtype=float)

    # 先进行归一化
    for i in range(shape1):
        desc1[i, :] /= numpy.linalg.norm(desc1[i, :])

    for i in range(shape2):
        desc2[i, :] /= numpy.linalg.norm(desc2[i, :])

    # 下面来计算余弦相似度
    matchscores = numpy.zeros((shape1, 1), 'int')
    desc2_T = desc2.T
    for i in range(shape1):
        dotprods = desc1[i, :] * desc2_T
        index = numpy.argsort(numpy.arccos(dotprods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if (numpy.arccos(dotprods[0, index[0, 0]]) < dist_ratio * numpy.arccos(dotprods[0, index[0, 1]])):
            matchscores[i] = int(index[0, 0])
    # 这里返回的是下标，因此绘制的时候还要根据下标去查找
    return matchscores


def appendimages(im1, im2):
    """返回将两幅图并排拼成的一幅新图像"""
    # 选取具有最少行数的图像,然后填充足够的空行
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    # 这一步防止两幅图像大小不同
    if rows1 < rows2:
        im1 = numpy.concatenate(
            (im1, numpy.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = numpy.concatenate(
            (im2, numpy.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    return numpy.concatenate((im1, im2), axis=1)


def match_twosided(desc1, desc2):
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # 去除不对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12


def plot_matches(img1, img2, points1, points2, matchscores):
    """显示一幅带有连接匹配连线的图像"""
    '''
    参数：
        *im1,　im2: 数组图像
        *locs1, locs2: 特征位置
        *matchscores: match(*)的输出
        *show_below: 如果图像应该显示在匹配的下方
    '''
    width = img1.shape[1]
    img3 = appendimages(img1, img2)
    color = [(255, 0, 0), (255, 255, 0), (0, 0, 255),
             (255, 0, 255), (0, 255, 0)]

    for i in range(matchscores.size):
        index = int(matchscores[i])
        if index > 0:
            locs1 = numpy.array(
                (int(points1[i][0]), int(points1[i][1])), dtype=int)
            locs2 = numpy.array(
                (int(points2[index][0]), int(points2[index][1])), dtype=int)

            cv2.line(img3, (locs1[0], locs1[1]),
                     (locs2[0] + width, locs2[1]), color[i % 5], 1)
            cv2.circle(img3, (locs1[0], locs1[1]), 2, (255, 0, 0), 2)
            cv2.circle(img3, (locs2[0] + width, locs2[1]), 2, (0, 255, 0), 2)

    return img3


def Harris_match(img1, img2):
    Key_points1, Descs1, m_directions1 = Harris_Detetct(img1)
    Key_points2, Descs2, m_directions2 = Harris_Detetct(img2)
    # plot_features(img1, Key_points1, m_directions1)

    matched = match_twosided(Descs1, Descs2)

    img3 = plot_matches(img1, img2, Key_points1, Key_points2, matched[:1000])

    cv2.imwrite('result.jpg', img3)


if __name__ == '__main__':
    img1 = cv2.imread("1.JPG", cv2.IMREAD_COLOR)
    img2 = cv2.imread("2.JPG", cv2.IMREAD_COLOR)
    Harris_match(img1, img2)
    # Key_points, Descs, m_directions = Harris_Detetct(img1)
    # plot_features(img, Key_points, m_directions)
    # cv2.imshow('Points', img)
    # cv2.waitKey()
    # cv2.imwrite('m_points1.jpg', img1)

    # Plot_harris_points(img, Filtered_coords)
