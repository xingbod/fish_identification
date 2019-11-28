import pathlib
import glob

import os
import random
import numpy as np
from scipy.special import binom


import cv2  # For OpenCV modules (For Image I/O and Contour Finding)
import scipy.fftpack  # For FFT2

def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name
class pigementUtil():
    def __init__(self,SAVE_PATH,alpha=0.2):
        self.alpha = alpha
        self.SAVE_PATH = SAVE_PATH

    def imclearborder(self,imgBW, radius):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]

        contourList = []  # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    #### bwareaopen definition
    def bwareaopen(self,imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

        return imgBWcopy

    def binaryiterative(self,Ihmf2):
        i = 70
        Ithresh = Ihmf2 < i
        while i > 10:
            # print(i)
            Ithresh = Ihmf2 < i
            rate1 = np.count_nonzero(Ithresh) / (
                        Ithresh.shape[0] * Ithresh.shape[1])  # should be better smaller than 0.2
            if rate1 <= self.alpha:
                print('stop at:', i)
                break
            i -= 5

        return Ithresh

    def transform2pigment(self,img_path):
        # Read in image
        # img = cv2.imread(img_path,0)
        img = cv2.imread(img_path)
        # 读取图像
        r, g, b = cv2.split(img)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg

        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        # img = cv2.merge([b, g, r])
        img = b
        # Number of rows and columns
        rows = img.shape[0]
        cols = img.shape[1]

        # Remove some columns from the beginning and end
        img = img[:, 59:cols - 20]

        # Number of rows and columns
        rows = img.shape[0]
        cols = img.shape[1]

        # Convert image to 0 to 1, then do log(1 + I)
        imgLog = np.log1p(np.array(img, dtype="float") / 255)

        # Create Gaussian mask of sigma = 10
        M = 2 * rows + 1
        N = 2 * cols + 1
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
        centerX = np.ceil(N / 2)
        centerY = np.ceil(M / 2)
        gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

        # Low pass and high pass filters
        Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
        Hhigh = 1 - Hlow

        # Move origin of filters so that it's at the top left corner to
        # match with the input image
        HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
        HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

        # Filter the image and crop
        If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
        Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
        Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

        # Set scaling factors and add
        gamma1 = 0.3
        gamma2 = 1.5
        Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

        # Anti-log then rescale to [0,1]
        Ihmf = np.expm1(Iout)
        Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        Ihmf2 = np.array(255 * Ihmf, dtype="uint8")

        # Threshold the image - Anything below intensity 65 gets set to white
        Ithresh = self.binaryiterative(Ihmf2)
        # Ithresh = Ihmf2 < 45

        # Ithresh = cv2.adaptiveThreshold(Ihmf2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,0.5)
        Ithresh = 255 * Ithresh.astype("uint8")

        # Clear off the border.  Choose a border radius of 5 pixels
        Iclear = self.imclearborder(Ithresh, 5)

        # Eliminate regions that have areas below 120 pixels
        Iopen = self.bwareaopen(Iclear, 120)
        return img, Ihmf2, Ithresh, Iopen

    def process2pigment(self,file):
      
        img, Ihmf2, Ithresh, Iopen = self.transform2pigment(file)
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        print('processing:' + file)
        save_path = self.SAVE_PATH + '/' + parts[-2]
        check_folder(save_path)
        print('saving:' + save_path)

        #cv2.imwrite(save_path + '/Homomorphic_' + parts[-1], Ihmf2)
        #cv2.imwrite(save_path + '/Threshold_' + parts[-1], Ithresh)
        cv2.imwrite(save_path + '/Opend_' + parts[-1], Iopen)
        return Iopen

#### imclearborder definition

