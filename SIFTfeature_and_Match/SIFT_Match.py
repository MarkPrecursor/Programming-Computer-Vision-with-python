# -*- coding: utf-8 -*-
"""
*SIFT特征的检测以及特征匹配
*创建于2016.5.9
*作者：Mark
"""
import os
import cv2
import numpy
import scipy.io as sio
from PIL import Image

def process_image(imagename, resultname, params='--edge-thresh 10 --peak-thresh 5'):
  """处理一幅图像，提取SIFT特征"""
  im = Image.open(imagename).convert('L')
  im.save('tmp.pgm')
  imagename = 'tmp.pgm'

  cmd = str('/home/mark/Engineer/Python_Project/CV/PCVP/SIFTfeature_and_Match/sift ' + imagename + " --output=" + resultname + ' ' + params)
  os.system(cmd)
  print 'processed', imagename, 'to', resultname
  
  # Re-write as .mat file, which loads faster.
  f = numpy.loadtxt(resultname)
  sio.savemat(resultname + '.mat', {'f':f}, oned_as='row')

def read_features_from_file(filename):
  '''读取特征值，以矩阵的方式返回'''
  f = sio.loadmat(filename + '.mat')['f']
  # f = numpy.loadtxt(filename)
  return f[:, :4], f[:, 4:]  # 分别是特征位置，特征描述子

def write_features_to_file(filename, locations, desc):
  '''将特征值和描述子保存到文件中'''
  numpy.savetxt(filename, numpy.hstack((locations, desc)))

def plot_features(img, locations, circle=True):
  '''将检测到的特征点绘制到图像上'''
  if circle:
    for p in locations:
      p = (int(p[0]), int(p[1]), int(p[2]), int(p[3]))
      cv2.circle(img, p[:2], p[2], (0, 255, 0), 2)
  else:
    for p in locations:
      cv2.circle(img, p[:2], 1, (0, 255, 0))
  

def Detect_and_Draw():
  img_name = "1.JPG"
  img = cv2.imread(img_name, cv2.IMREAD_COLOR)
  G_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

  process_image(img_name, 'test.sift')
  locas, descs = read_features_from_file('test.sift')
  plot_features(img, locas)

  cv2.imshow('result', img)
  cv2.waitKey(0)


def match(desc1, desc2):
  shape1 = desc1.shape
  shape2 = desc2.shape
  dist_ratio = 0.6
  desc1 = numpy.matrix(desc1, dtype=float)
  desc2 = numpy.matrix(desc2, dtype=float)

  # 先进行归一化
  for i in range(shape1[0]):
      desc1[i, :] /= numpy.linalg.norm(desc1[i, :])

  for i in range(shape2[0]):
      desc2[i, :] /= numpy.linalg.norm(desc2[i, :])

  # 下面来计算余弦相似度
  matchscores = numpy.zeros((shape1[0], 1), 'int')
  desc2_T = desc2.T
  for i in range(shape1[0]):
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
    index = int(matchscores[i])
    if index > 0:
      locs1 = numpy.array((int(points1[i][0]), int(points1[i][1])), dtype=int)
      locs2 = numpy.array((int(points2[index][0]), int(points2[index][1])), dtype=int)

      cv2.line(img3, (locs1[0], locs1[1]), (locs2[0] + width, locs2[1]), color[i % 5], 1)
      cv2.circle(img3, (locs1[0], locs1[1]), 2, (0, 255, 0), 2)
      cv2.circle(img3, (locs2[0] + width, locs2[1]), 2, (0, 255, 0), 2)

  return img3


def SIFT_match(img_name1, img_name2):
  img1 = cv2.imread(img_name1, cv2.IMREAD_COLOR)
  img2 = cv2.imread(img_name2, cv2.IMREAD_COLOR)

  process_image(img_name1, 'match1.sift')
  process_image(img_name2, 'match2.sift')

  locas1, descs1 = read_features_from_file('match1.sift')
  locas2, descs2 = read_features_from_file('match2.sift')

  matched = match_twosided(descs1, descs2)

  img3 = plot_matches(img1, img2, locas1, locas2, matched[:1000])

  cv2.imwrite('result2.jpg', img3)
# cv2.imshow('result', img3)
# cv2.waitKey(0)


if __name__ == '__main__':
  img_name1 = '5.jpg'
  img_name2 = '6.jpg'
  
  #记录程序运行时间
  e1 = cv2.getTickCount()
  # your code execution
  SIFT_match(img_name1, img_name2)
  # Detect_and_Draw()
  e2 = cv2.getTickCount()
  time = (e2 - e1)/cv2.getTickFrequency()
  print "Processing time is %f s"%time
