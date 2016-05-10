# -*- coding: utf-8 -*-
"""
* 摄像机标定，及投影建模所需要的操作
* 创建于：2016.5.9
* 作者: Mark
"""
import cv2
import glob
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

def rotation_matrix(a):
  """创建一个用于围绕向量a轴旋转的三维旋转矩阵"""
  R = np.eye(4)
  R[:3, :3] = lin.expm([[0, -a[2], a[1]],
                        [a[2], 0, -a[0]],
                        [-a[1], a[0], 0]])
  return R


def m_Calibration(img):
  '''电脑摄像头粗略的标定结果'''
  shape = img.shape
  f_x = 595
  f_y = 582
  K = np.diag([f_x, f_y, 1])
  K[0, 2] = 0.5 * shape[1]
  K[1, 2] = 0.5 * shape[0]
  # np.savetxt('K.txt', K)
  return K


class Camera(object):
  """表示针孔照相机的类"""
  def __init__(self, P):
    '''初始化P = K[R|t]照相机模型'''
    self.P = P
    self.K = None  # Calibration matrix.
    self.R = None  # Rotation matrix.
    self.t = None  # Translation vector.
    self.c = None  # Camera center.

  def project(self, X):
    """x(4*n的数组)的投影点，并进行坐标归一化"""
    x = np.dot(self.P, X)
    for i in range(3):
      x[i] /= x[2]
    return x

  def factor(self):
    '''将照相机矩阵P分解为K, R, t, 其中P = K[R|t].'''
    K, R = lin.rq(self.P[:, :3])     # 分解前3x3的部分

    # 保证K正定
    T = np.diag(np.sign(np.diag(K)))  # sign是符号函数
    if np.linalg.det(T) < 0:
      T[1, 1] *= -1

    self.K = np.dot(K, T)
    self.R = np.dot(T, R)  # T的逆矩阵是它自身
    self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

    return self.K, self.R, self.t

  def center(self):
    '''计算并返回照相机的中心'''
    if self.c is not None:
      return self.c
    else:
      # 通过因子分解计算c
      self.factor()
      self.c = -np.dot(self.R.T, self.t)
      return self.c


def example():
  '''书上关于单应性变换的实例'''
  # 载入点
  points = np.loadtxt('house.p3d').T
  points = np.vstack((points, np.ones(points.shape[1])))
  
  # 设置照相机参数
  P = np.hstack((np.eye(3), np.array([[0], [0], [-10]])))
  cam = Camera(P)
  x = cam.project(points)
  
  plt.figure()
  plt.plot(x[0], x[1], 'k.')
  # plt.savefig('result.png')
  plt.show()


def Course_version():
  """下面这里用书上最简单的方法来进行摄像机内参数标定，标定精度比较粗糙，仅作实验用途"""
  img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
  gray = np.float32(img)
  dst = cv2.cornerHarris(gray, 4, 3, 0.06)
  img[dst > 0.42 * dst.max()] = 255
  img[img[:] < 255] = 0
  image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  print contours
  cv2.imshow('result', img)
  cv2.waitKey(0)


def Clibration():
  '''这里采用opencv函数进行棋盘标定以及标定检验，结果存储在“cam_data.npz” '''
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((6*7, 3), np.float32)
  objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

  # Arrays to store object points and image points from all the images.
  objpoints = []  # 3d point in real world space
  imgpoints = []  # 2d points in image plane.

  images = glob.glob('*.jpg')  # 获取文件目录下所有.jpg文件的目录

  for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
      objpoints.append(objp)  # 在这里同于摄像机标定的图像是累加的
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)
      # Draw and display the corners
      img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
      cv2.imshow('img', img)
      cv2.waitKey(600)

  cv2.destroyAllWindows()
  # 下面的返回值分别对应: 标定是否成功， 摄像机内参数K，摄像机畸变参数dist， 摄像机的旋转R和位移t
  ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  # K = m_Calibration()

  np.savez('cam_data.npz', K=K, dist=dist, R=rvecs, t=tvecs)  # 存储

  img = cv2.imread('left12.jpg')
  h, w = img.shape[:2]
  # 输出的参数是畸变矫正之后的摄像机内部参数，roi应该是畸变矫正需要切掉周边的一圈，所以输出这样一个ROI
  newcameraK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

  # 下面采用两种方法进行摄像机的畸变矫正
  dst = cv2.undistort(img, K, dist, None, newcameraK)
  # crop the image
  x, y, w, h = roi
  dst = dst[y: y + h, x: x + w]
  cv2.imwrite('calibresult1.png', dst)

  # 首先我们要找到从畸变图像到非畸变图像的映射方程。再使用重映射方程。
  # undistort
  mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newcameraK, (w, h), 5)
  dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
  # crop the image
  x, y, w, h = roi
  dst = dst[y: y + h, x: x + w]
  cv2.imwrite('calibresult2.png', dst)

  # 利用反向投影误差对我们找到的参数的准确性进行估计。得到的结果越接近0越好。
  mean_error = 0
  for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
  print "total error: ", mean_error/len(objpoints)


def draw(img, corners, imgpts):
  corner = tuple(corners[0].ravel())
  img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
  img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
  img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
  return img


def Result_test():
  '''用于检验摄像机标定的结果'''
  # Load previously saved data
  with np.load('cam_data.npz') as X:
    K, dist, R, t = [X[i] for i in ('K', 'dist', 'R', 't')]

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objp = np.zeros((6 * 7, 1, 3), np.float32)
  objp[:, :, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 1, 2)
  axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)  # 用于投影的坐标轴

  for fname in glob.glob('left*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret == True:
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      
      # Find the rotation and translation vectors.这里的Objp和corner2应该是匹配完成了的点对
      retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, K, dist)
       
      # project 3D points to image plane
      imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, dist)
      
      img = draw(img, corners2, imgpts)
      cv2.imshow('img', img)
      k = cv2.waitKey(0) & 0xff
      if k == 's':
        cv2.imwrite(fname[:6]+'.png', img)
  
  cv2.destroyAllWindows()


def Manage_data(matches, kp1, kp2):
  '''使用RANSAC.py计算单应性矩阵的时候用来预处理数据'''
  objp1 = np.zeros((60, 2), np.float32)
  objp2 = np.zeros((60, 2), np.float32)

  for i in range(60):
    p = matches[i]
    i1 = p.queryIdx
    i2 = p.trainIdx
    objp1[i, :] = kp1[i1].pt
    objp2[i, :] = kp2[i2].pt

  one = np.ones((1, objp1.shape[0]), dtype=float)
  objp1 = np.vstack((objp1.T, one))
  objp2 = np.vstack((objp2.T, one))

  return objp1, objp2


def Collect_Pictures(num):
  '''采用opencv进行棋盘标定时用来采集图像'''
  cap = cv2.VideoCapture(0)

  while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
      for i in range(num):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        name = str(i)+".jpg"
        cv2.imwrite(name, frame)
        cv2.waitKey(1500)
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  Clibration()
