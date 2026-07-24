import argparse
import numpy as np
import cv2 as cv
import math as m
import keras
import gdown
import os

# making directory for and downloading weights

try :
  os.mkdir("model")
  gdown.download(
    url="https://drive.google.com/uc?id=1itzGsFcV0tCxW3tjlj8Ns0aTtou9gyob" , # best t1
    output="model/best-t1.keras"
  )

  gdown.download(
      url="https://drive.google.com/uc?id=1hhhcqCgQG0Q_fMvZr6idaBnldRMlqp_G" , # best t2
      output="model/best-t2.keras"
  )
except :
  pass

#--------
def padding(img , padding_dimension , dim="all") :
  # this fucntion add square padding
  if not isinstance(img , (str , np.ndarray)) : 
    raise TypeError('invalid object , img should be numpy array or image address')
  if not isinstance(padding_dimension , int) :
    raise TypeError('padding dimension should be an integer indicating number of rows and columns')
  #-------
  if isinstance(img , str) :
    image = cv.imread(img)
  else :
    image = img
  if dim == "all" :
    #---- add padding to right and down border
    x_dim = image.shape[0]
    y_dim = image.shape[1]
    #-----
    x_adding = np.zeros((padding_dimension , y_dim , 3)).astype(np.uint8)
    y_adding = np.zeros((x_dim + padding_dimension, padding_dimension , 3)).astype(np.uint8)
    #----
    image = np.concatenate((image , x_adding) , axis=0)
    image = np.concatenate((image , y_adding) , axis=1)
    #------- add padding to left and up border
    image = np.flip(image , axis=0)
    image = np.flip(image , axis=1)
    #---
    x_dim = image.shape[0]
    y_dim = image.shape[1]
    #-----
    x_adding = np.zeros((padding_dimension , y_dim , 3)).astype(np.uint8)
    y_adding = np.zeros((x_dim + padding_dimension, padding_dimension , 3)).astype(np.uint8)
    #----
    image = np.concatenate((image , x_adding) , axis=0)
    image = np.concatenate((image , y_adding) , axis=1)
    #---- reverting image
    image = np.flip(image , axis=1)
    image = np.flip(image , axis=0)
    #-----------
  elif dim == "down" : 
    x_dim    = image.shape[0]
    y_dim    = image.shape[1]
    #-------
    x_adding = np.zeros((padding_dimension , y_dim , 3)).astype(np.uint8)
    image    = np.concatenate((image , x_adding) , axis=0) 
    #-------
  elif dim == "right" :
    x_dim    = image.shape[0]
    y_dim    = image.shape[1]
    #------
    y_adding = np.zeros((x_dim , padding_dimension , 3)).astype(np.uint8)
    image    = np.concatenate((image , y_adding) , axis=1) 
    #------
  elif dim == "up" :
    x_dim    = image.shape[0]
    y_dim    = image.shape[1]
    #-------
    x_adding = np.zeros((padding_dimension , y_dim , 3)).astype(np.uint8)
    image    = np.flipud(image) 
    image    = np.concatenate((image , x_adding) , axis=0)
    image    = np.flipud(image)  
    #-------
  elif dim == "left" :
    x_dim    = image.shape[0]
    y_dim    = image.shape[1]
    #------
    y_adding = np.zeros((x_dim , padding_dimension , 3)).astype(np.uint8)
    image    = np.fliplr(image)
    image    = np.concatenate((image , y_adding) , axis=1)
    image    = np.fliplr(image)  
    #------
  return np.array(image)
#---------------------------
def image_slider(image , slide_size) :
  x_dim = image.shape[0]
  y_dim = image.shape[1]
  #-----
  row_start = 0
  col_start   = 0
  cnt = 0
  while row_start < x_dim :
   row_end = slide_size + row_start
   col_end = slide_size + col_start
   slide = image[row_start:row_end , col_start:col_end]
   col_start = col_start + slide_size
   if col_start == y_dim : 
     row_start = row_start + slide_size
     col_start = 0
   yield slide
  
#---------------------------
def heatMap(img , map, out) :
    hmap         = cv.applyColorMap(map , cv.COLORMAP_JET)
    super_impose = cv.addWeighted(img , 0.6 , hmap , 0.4 , 0) 
    cv.imwrite(out , super_impose)
#---------------------------
def cli() :
    parser = argparse.ArgumentParser(
        prog  ="deepMammolyzer", 
        usage = "finding suspicious microcalcifications on mammpgrams" , 
    )
    parser.add_argument(
        "input_image_path" , 
        type = str , 
    )
    parser.add_argument(
        "output_image_path" , 
        type = str ,
    )
    parser.add_argument(
        "-T" ,
        type = str,
        required=True
    )
    #-----------
    args   = parser.parse_args()
    ipath  = args.input_image_path
    opath  = args.output_image_path
    typ    = args.T
    #-----------
    if typ == "full" : 
      resnet50_cbam = keras.models.load_model("model/best-t1.keras")
    elif typ == "sus" :
      resnet50_cbam     = keras.models.load_model("model/best-t1.keras")
      resnet50_cbam_sus = keras.models.load_model("model/best-t2.keras") # !!!! fixing 
    print("resnet50 loaded")
    #-----------
    image = cv.imread(ipath)
    x_dim = image.shape[0]
    y_dim = image.shape[1]
    if x_dim % 224 != 0 :
      padd_dim = m.ceil(x_dim/224)*224 - x_dim
      image    = padding(image , padd_dim , dim="down")
    if y_dim % 224 != 0 :
      padd_dim = m.ceil(y_dim/224)*224 - y_dim
      image    = padding(image , padd_dim , dim="right")
    #----------
    x_dim_padd  = int(image.shape[0])
    y_dim_padd  = int(image.shape[1])
    x_dim_slide = int(image.shape[0]/224)
    y_dim_slide = int(image.shape[1]/224)
    #----------
    cnt = 0
    slider = image_slider(image , 224)
    pred_proba = []
    while True :
      try :
        slide = next(slider)
        if np.sum(slide) < 3000000 :
          pred_proba.append(1)
        else :
          if typ == "full" :
            slide = slide / 255
            slide = np.expand_dims(slide , axis=0)
            pred = resnet50_cbam.predict(slide)
            pred_proba.append(pred[0][0])
          elif typ == "sus" :
            slide = slide / 255
            slide = np.expand_dims(slide , axis=0)
            pred = resnet50_cbam.predict(slide)
            if pred > 0.226 :
              pred_proba.append(1)
            else :
              pred_2 = resnet50_cbam_sus(slide)
              pred_proba.append(1 - pred_2[0][0])
      except :
        heatmap_vector  = np.array(pred_proba)
        heatmap_arr     = np.reshape(heatmap_vector , (x_dim_slide , y_dim_slide))
        heatmap_arr     = 1 - heatmap_arr 
        heatmap_arr     = cv.normalize(heatmap_arr , None , 0 , 255 , cv.NORM_MINMAX, cv.CV_8U) 
        heatmap_arr     = np.stack((heatmap_arr , heatmap_arr , heatmap_arr), axis=2)
        heatmap_resize  = cv.resize(heatmap_arr , (y_dim_padd , x_dim_padd) , interpolation=cv.INTER_CUBIC)
        #---
        heatMap(image , heatmap_resize , opath)
        #---
        break
#---------
cli()             