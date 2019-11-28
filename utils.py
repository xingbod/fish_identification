from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import glob
import tensorflow as tf
import os
import random
import numpy as np
from scipy.special import binom

from tensorflow import  keras
from tensorflow.keras import datasets, layers, optimizers, models
from tensorflow.keras import regularizers
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
        
import cv2 # For OpenCV modules (For Image I/O and Contour Finding)
import scipy.fftpack # For FFT2

def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def getfilelist(dirs):
  Filelist = []
  for home, dirs, files in os.walk(dirs):
    for filename in files:
# 文件名列表，包含完整路径
      Filelist.append(os.path.join(home, filename))
# # 文件名列表，只包含文件名
# Filelist.append( filename)
  return Filelist

		
class LoadFishDataUtil():
    def __init__(self, directory_str,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES=None,SPLIT_WEIGHTS=(0.7, 0.15, 0.15)):
      self.directory_str=directory_str
      self.SPLIT_WEIGHTS=SPLIT_WEIGHTS
      self.BATCH_SIZE=BATCH_SIZE
      self.data_dir = pathlib.Path(directory_str)
      self.image_count = len(list(self.data_dir.glob('*/*.png'))) #Opend
      if CLASS_NAMES is None:
        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"])
      else:
        self.CLASS_NAMES = CLASS_NAMES

      self.class_num=len(self.CLASS_NAMES)
 
      self.IMG_HEIGHT = IMG_HEIGHT
      self.IMG_WIDTH = IMG_WIDTH
      self.STEPS_PER_EPOCH = np.ceil(self.image_count/BATCH_SIZE)


    def get_label(self,file_path):
    # convert the path to a list of path components
      parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
      #print(parts[-2] == self.CLASS_NAMES)
      wh = tf.where(tf.equal(self.CLASS_NAMES,parts[-2]))
      return parts[-2] == self.CLASS_NAMES
    
    def get_label_withname(self,file_path):
    # convert the path to a list of path components
      parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
      
      wh = tf.where(tf.equal(self.CLASS_NAMES,parts[-2]))
      return wh
    
    def decode_img(self,img):
    # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img, tf.float32)
    #img = (img/127.5) - 1
    # resize the image to the desired size.
      return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self,file_path):
      label = self.get_label(file_path)
    # load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      img = self.decode_img(img)
      return img, label
    
    def process_path_withname(self,file_path):
      label = self.get_label_withname(file_path)
    # load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      img = self.decode_img(img)
      return img, label
  

    def prepare_for_training(self,ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
      if cache:
        if isinstance(cache, str):
          ds = ds.cache(cache)
        else:
          ds = ds.cache()

      ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
      ds = ds.repeat()

      ds = ds.batch(self.BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
      ds = ds.prefetch(buffer_size=self.AUTOTUNE)

      return ds
  
    def prepare_for_testing(self,ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
      if cache:
        if isinstance(cache, str):
          ds = ds.cache(cache)
        else:
          ds = ds.cache()

      #ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
      #ds = ds.repeat()

      ds = ds.batch(self.BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
      ds = ds.prefetch(buffer_size=self.AUTOTUNE)

      return ds

    def loadFishData(self):
      list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
      self.AUTOTUNE = tf.data.experimental.AUTOTUNE
      self.labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
   
      train_size = int(self.SPLIT_WEIGHTS[0] * self.image_count)
      val_size = int(self.SPLIT_WEIGHTS[1] * self.image_count)
      test_size = int(self.SPLIT_WEIGHTS[2] * self.image_count)
      train_ds = self.prepare_for_training(self.labeled_ds)

      full_dataset = train_ds.shuffle(buffer_size=1000,reshuffle_each_iteration = False )
      train_dataset = full_dataset.take(train_size)
      remianing_train_set = full_dataset.skip(train_size)
      val_dataset = remianing_train_set.take(val_size)
      remain_val_set = remianing_train_set.skip(val_size)
      test_dataset = remain_val_set.take(test_size)
      return train_dataset,val_dataset,test_dataset,self.STEPS_PER_EPOCH,self.CLASS_NAMES,self.class_num
		
		
    def loadTestFishData(self):
      list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
      print(f"we have total {self.image_count} images in this folder")
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
      self.AUTOTUNE = tf.data.experimental.AUTOTUNE
      self.labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
      dataset = self.labeled_ds.batch(self.BATCH_SIZE)#按照顺序取出4行数据，最后一次输出可能小于batch
      #dataset = dataset.repeat()#数据集重复了指定次数
      # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
      #为了配合输出次数，一般默认repeat()空
      #test_ds = self.prepare_for_testing(self.labeled_ds)
      #test_ds = self.labeled_ds
        
     
      return dataset,self.class_num
    
    def loadFishDataWithname(self):
      list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
      self.AUTOTUNE = tf.data.experimental.AUTOTUNE
      self.labeled_ds = list_ds.map(self.process_path_withname, num_parallel_calls=self.AUTOTUNE)
   
      train_size = int(self.SPLIT_WEIGHTS[0] * self.image_count)
      val_size = int(self.SPLIT_WEIGHTS[1] * self.image_count)
      test_size = int(self.SPLIT_WEIGHTS[2] * self.image_count)
      train_ds = self.prepare_for_training(self.labeled_ds)

      full_dataset = train_ds.shuffle(buffer_size=1000,reshuffle_each_iteration = False )
      train_dataset = full_dataset.take(train_size)
      remianing_train_set = full_dataset.skip(train_size)
      val_dataset = remianing_train_set.take(val_size)
      remain_val_set = remianing_train_set.skip(val_size)
      test_dataset = remain_val_set.take(test_size)
      return train_dataset,val_dataset,test_dataset,self.STEPS_PER_EPOCH,self.CLASS_NAMES,self.class_num
		
		
    def loadTestFishDataWithname(self):
      list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
      print(f"we have total {self.image_count} images in this folder")
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
      self.AUTOTUNE = tf.data.experimental.AUTOTUNE
      self.labeled_ds = list_ds.map(self.process_path_withname, num_parallel_calls=self.AUTOTUNE)
      dataset = self.labeled_ds.batch(self.BATCH_SIZE)#按照顺序取出4行数据，最后一次输出可能小于batch
      #dataset = dataset.repeat()#数据集重复了指定次数
      # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
      #为了配合输出次数，一般默认repeat()空
      #test_ds = self.prepare_for_testing(self.labeled_ds)
      #test_ds = self.labeled_ds
        
     
      return dataset,self.class_num 
		
		

#### imclearborder definition

def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    contourList = [] # ID list of contours that touch the border

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
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy
def binaryiterative(Ihmf2):
    i = 70
    Ithresh = Ihmf2 < i
    while i > 10:
        #print(i)
        Ithresh = Ihmf2 < i
        rate1=np.count_nonzero(Ithresh)/(Ithresh.shape[0]*Ithresh.shape[1]) # should be better smaller than 0.2
        if rate1 <= .2:
            print('stop at:', i)
            break
        i -= 5
        
    return Ithresh
def transform2pigment(img_path):
     # Read in image
    #img = cv2.imread(img_path,0)
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
    #img = cv2.merge([b, g, r])
    img=b
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
    Ithresh = binaryiterative(Ihmf2)
    #Ithresh = Ihmf2 < 45
    
    #Ithresh = cv2.adaptiveThreshold(Ihmf2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,0.5)
    Ithresh = 255 * Ithresh.astype("uint8")

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 5)

    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 120)
    return img,Ihmf2,Ithresh,Iopen


def process2pigment(file):
    SAVE_PATH ='/media/xingbo/Storage/fish_identification/data/SESSION_AQUARIUM/SESSION2_PIGMENT'
    print(SAVE_PATH)
    img,Ihmf2,Ithresh,Iopen =transform2pigment(file)
    path = os.path.normpath(file)
    parts=path.split(os.sep)
    print('processing:'+file)
    save_path=SAVE_PATH+'/'+parts[-2]
    check_folder(save_path)
    print('saving:'+save_path)

    cv2.imwrite(save_path+'/Homomorphic_'+parts[-1],Ihmf2)
    cv2.imwrite(save_path+'/Threshold_'+parts[-1],Ithresh)
    cv2.imwrite(save_path+'/Opend_'+parts[-1],Iopen)
    return Iopen

#### Main program