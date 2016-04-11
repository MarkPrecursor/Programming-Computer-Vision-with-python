# -*- coding: utf-8 -*-
"""
*Harris角点检测代码，不用OpenCV
*创建于2016.3.27
*作者：Mark
"""
import numpy
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

def Compute_Harris_Response(img, sigma=3):
    """对一副灰度图像，返回每个像素值表示Harris角点检测器响应函数值的图像"""
    # 计算导数：
    imx = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    imy = numpy.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

    # 计算Harris矩阵的分量:
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # 计算特征值和迹:
    Wdet = Wxx * Wyy - Wxy * 2
    Wtr = Wxx + Wyy

    return Wdet/Wtr

def Get_Harris_Points(Harris_img, min_dist=10, threshold=0.1, maxnum = 100):
    """ 从Harris图像中返回角点,
    *　min_dist为分割角点和图像边界的最小像素数目
    *　maxnum是返回的最大数目
    """
    # 寻找高于阈值的候选角点：
    corner_threshold = Harris_img.max() * threshold
    Harris_img_T = (Harris_img > corner_threshold) * 1
    #得到候选角点的坐标:
    coords = numpy.array(Harris_img_T.nonzero()).T
    #得到Harris响应值：
    Candidate_Value = [Harris_img[c[0], c[1]] for c in coords]
    # 对候选点按照响应值排序
    index = numpy.argsort(Candidate_Value)

    if index.size > maxnum:
        index = index[:maxnum]

    # 将可行点的位置保存到数组当中
    Allowed_locations = numpy.zeros(Harris_img.shape, dtype=numpy.uint8)
    Allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1  # 这一句是圈出一圈限定范围

    # 按照最小距离原则选择最佳的Harris点
    filtered_coords = []
    for i in index:
        if Allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            Allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist),
            (coords[i, 0]-min_dist):(coords[i, 0]+min_dist)] = 0
    return filtered_coords

def Plot_harris_points(img, filtered_coords):
    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '.')
    plt.axis('off')
    plt.show()

def get_descriptors(image, filtered_coords, wid=5):
  """
     *对于每个返回的点，返回点周围2*wid+1个像素的值
     *该函数创建检测出的Harris角点的特征描述子
     *图像块长度需要是奇数大小
  """
  num = filtered_coords.shape[0]  # 对应的点的个数,每个元素是一个坐标
  desc = numpy.zeros((num, (2*wid+1)**2), dtype=numpy.uint8)  # 每一个元素是一个像素值
  for i in range(num):
      coord = filtered_coords[i, :]
      desc[i, :] = image[coord[0]-wid:coord[0]+wid+1,  # 这里将当前点的邻域抽成了一个向量
                         coord[1]-wid:coord[1]+wid+1].flatten()
  return desc

def ncc(patch1, patch2):
  """返回两幅图片之间归一化的互相关"""
  d1 = (patch1 - numpy.mean(patch1)) / numpy.std(patch1)
  d2 = (patch2 - numpy.mean(patch2)) / numpy.std(patch2)
  return numpy.sum(d1 * d2) / (len(patch1) - 1)

def match(desc1, desc2, threshold=0.5):
    """对于第一幅图像中的每个角点描述子,使用归一化互相关，选取它在第二副图像中的匹配角点"""
    d = -numpy.ones((len(desc1), len(desc2)))  # 点对的距离
    for i in range(len(desc1)):
        for j in range(len(desc2)):
          ncc_value = ncc(desc1[i], desc2[j])
          if ncc_value > threshold:
              d[i, j] = ncc_value
    ndx = numpy.argsort(-d)
    return ndx[:, 0]

def match_twosided(desc1, desc2, threshold=0.5):
    """为获得更稳定的匹配，两幅图相互匹配，然后过滤掉两种方法中都不是最好的匹配"""
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = numpy.where(matches_12 >= 0)[0]
    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12

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

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """显示一幅带有连接匹配连线的图像"""
    '''
    参数：
        *im1,　im2: 数组图像
        *locs1, locs2: 特征位置
        *matchscores: match(*)的输出
        *show_below: 如果图像应该显示在匹配的下方
    '''
    import pylab
    im3 = appendimages(im1, im2)
    if show_below:
        im3 = numpy.vstack((im3, im3))
    pylab.imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            pylab.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
    pylab.axis('off')


if __name__ == '__main__':
    import cv2
    img = cv2.imread("1.JPG", cv2.IMREAD_COLOR)
    r, g, b = cv2.split(img)
    img = cv2.merge((b, g, r))
    G_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    HarrisImg = Compute_Harris_Response(G_img)
    # cv2.imshow("result", HarrisImg)
    # cv2.waitKey(0)
    Filtered_coords = Get_Harris_Points(HarrisImg)
    Plot_harris_points(img, Filtered_coords)
