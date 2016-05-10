# -*- coding: utf-8 -*-
"""
* 读取摄像头，采集图片并进行棋盘标定
* 创建于：2016.5.9
* 作者：Mark
"""
import homography as homo
import Clibration as Cli
from PIL import Image
from pylab import *
import numpy
import cv2

def cube_points(c, wid):
  """用于创建立方体的一个点列表"""
  p = []
  # bottom
  p.append([c[0] - wid, c[1] - wid, c[2] - wid])
  p.append([c[0] - wid, c[1] + wid, c[2] - wid])
  p.append([c[0] + wid, c[1] + wid, c[2] - wid])
  p.append([c[0] + wid, c[1] - wid, c[2] - wid])
  p.append([c[0] - wid, c[1] - wid, c[2] - wid])

  # top
  p.append([c[0] - wid, c[1] - wid, c[2] + wid])
  p.append([c[0] - wid, c[1] + wid, c[2] + wid])
  p.append([c[0] + wid, c[1] + wid, c[2] + wid])
  p.append([c[0] + wid, c[1] - wid, c[2] + wid])
  p.append([c[0] - wid, c[1] - wid, c[2] + wid])

  # sides
  p.append([c[0] - wid, c[1] - wid, c[2] + wid])
  p.append([c[0] - wid, c[1] + wid, c[2] + wid])
  p.append([c[0] - wid, c[1] + wid, c[2] - wid])
  p.append([c[0] + wid, c[1] + wid, c[2] - wid])
  p.append([c[0] + wid, c[1] + wid, c[2] + wid])
  p.append([c[0] + wid, c[1] - wid, c[2] + wid])
  p.append([c[0] + wid, c[1] - wid, c[2] - wid])

  return numpy.array(p).T


if __name__ == '__main__':
  # Cli.Collect_Pictures(12)
  # Cli.Clibration()
  # Cli.Result_test()

  # Load previously saved data
  with numpy.load('cam_data.npz') as X:
    K, dist, R, t = [X[i] for i in ('K', 'dist', 'R', 't')]

  img1 = cv2.imread('book_frontal.JPG', cv2.IMREAD_COLOR)
  img2 = cv2.imread('book_perspective.bmp', cv2.IMREAD_COLOR)

  G_img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
  G_img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)

  m_ORB = cv2.ORB_create()

  kp1, des1 = m_ORB.detectAndCompute(G_img1, None)
  kp2, des2 = m_ORB.detectAndCompute(G_img2, None)
  
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)

  matches = sorted(matches, key=lambda x: x.distance)  # Sort them in the order of their distance.

  obj1, obj2 = Cli.Manage_data(matches, kp1, kp2)

  model = homo.RansacModel()

  H = homo.H_from_ransac(obj1, obj2, model)

  # # Draw first 60 matches.
  # img3 = img2
  # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:60], img3)

  # cv2.imshow('result', img3)
  # cv2.imwrite('result.jpg', img3)
  # cv2.waitKey()

  # 位于边长为0.2， z = 0平面上底部的正方形
  Box = cube_points([0, 0, 0.1], 0.1)

  cam1 = Cli.Camera(numpy.hstack((K, numpy.dot(K, numpy.array([[0], [0], [-1]])))))
  box_cam1 = cam1.project(homo.make_homog(Box[:, :5]))

  # 使用H将点变换到第二幅图像中
  box_trans = homo.normalize(numpy.dot(H[0], box_cam1))

  # 从cam1和H中计算第二个照相机矩阵
  cam2 = Cli.Camera(numpy.dot(H[0], cam1.P))
  A = numpy.dot(numpy.linalg.inv(K), cam2.P[:, :3])
  A = numpy.array([A[:, 0], A[:, 1], numpy.cross(A[:, 0], A[:, 1])]).T
  cam2.P[:, :3] = numpy.dot(K, A)

  # 使用第二个照相机矩阵投影
  box_cam2 = cam2.project(homo.make_homog(Box))

  # 测试：将点投影在Z=0上，应该能够得到相同点
  point = numpy.array([[1, 1, 0, 1]]).T
  print homo.normalize(numpy.dot(numpy.dot(H[0], cam1.P), point)).T
  print homo.normalize(numpy.dot(H[0], cam1.project(point))).T
  print cam2.project(point).T


  # visualize

  # 2d projection of bottom square in template
  figure()
  imshow(img1)
  plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)

  # 2d projection of bottom square in image, transferred with H
  figure()
  imshow(img2)
  plot(box_trans[0, :], box_trans[1, :], linewidth=3)

  # 3d cube
  figure()
  imshow(img2)
  plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)

  import pickle
  with open('out_ch4_camera.pickle', 'wb') as f:
    pickle.dump(K, f)
    pickle.dump(numpy.dot(numpy.linalg.inv(K), cam2.P), f)

  show()
