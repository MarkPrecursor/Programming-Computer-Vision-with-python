# -*- coding: utf-8 -*-
"""
*ORB特征的检测以及特征匹配
*创建于2016.4.12
*作者：Mark
"""
import cv2
import numpy


def Detect_and_Draw():
    img = cv2.imread('1.JPG', cv2.IMREAD_COLOR)
    G_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(G_img, None)
    cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))
    cv2.imshow('result', img)
    cv2.waitKey(0)


def match(desc1, desc2):
    shape = desc1.shape
    dist_ratio = 0.6
    desc1 = numpy.matrix(desc1, dtype=float)
    desc2 = numpy.matrix(desc2, dtype=float)

    # 先进行归一化
    for i in range(shape[0]):
        desc1[i, :] /= numpy.linalg.norm(desc1[i, :])
        desc2[i, :] /= numpy.linalg.norm(desc2[i, :])

    # 下面来计算余弦相似度
    matchscores = numpy.zeros((shape[0], 1), 'int')
    desc2_T = desc2.T
    for i in range(shape[0]):
        dotprods = desc1[i, :] * desc2_T
        index = numpy.argsort(numpy.arccos(dotprods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if (numpy.arccos(dotprods[0, index[0, 0]]) < dist_ratio * numpy.arccos(dotprods[0, index[0, 1]])):
                    matchscores[i] = int(index[0, 0])
    return matchscores


def appendimages(im1, im2):
    """返回将两幅图并排拼成的一幅新图像"""
    # 选取具有最少行数的图像,然后填充足够的空行
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    # 这一步防止两幅图像大小不同
    if rows1 < rows2:
        im1 = numpy.concatenate((im1, numpy.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = numpy.concatenate((im2, numpy.zeros((rows1-rows2, im2.shape[1]))), axis=0)
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
    color = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0)]

    for i in range(matchscores.size):
        locs1 = numpy.array(points1[i].pt, dtype=int)
        locs2 = numpy.array(points2[i].pt, dtype=int)
        cv2.line(img3, (locs1[0], locs1[1]), (locs2[0]+width, locs2[1]), color[i % 5], 1)
    return img3


def ORB_match(img1, img2):
    G_img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
    G_img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(G_img1, None)
    kp2, des2 = orb.detectAndCompute(G_img2, None)

    cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
    cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0))

    matched = match_twosided(des1, des2)

    img3 = plot_matches(img1, img2, kp1, kp2, matched[:100])

    cv2.imwrite('result1.jpg', img3)
    return kp1, kp2, matched
    # cv2.imshow('result', img3)
    # cv2.waitKey(0)


if __name__ == '__main__':
    img1 = cv2.imread('5.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('6.jpg', cv2.IMREAD_COLOR)

    #记录程序运行时间
    e1 = cv2.getTickCount()
    # your code execution
    ORB_match(img1, img2)
    # Detect_and_Draw()
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print "Processing time is %f s"%time
