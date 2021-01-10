###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2



#My pre processing (use for both training and testing!)
def my_PreProc(data, train_test = "train"):
    assert(len(data.shape)==4)
    #assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = data
    if data.shape[1]==3:
        train_imgs = rgb2gray(data)
    #my preprocessing:
    # train_imgs = ImgEnhance(train_imgs, train_test)
    train_imgs = dataset_normalized(train_imgs, train_test)
    train_imgs = clahe_equalized(train_imgs, train_test)
    train_imgs = adjust_gamma(train_imgs, 1.2, train_test)
    train_imgs = train_imgs/255.  #reduce to 0-1 range

    #print(train_imgs.shape)
    #b = np.zeros((train_imgs.shape[0],1,train_imgs.shape[2],train_imgs.shape[3]),dtype=train_imgs.dtype)
    #g = np.zeros((train_imgs.shape[0],1,train_imgs.shape[2],train_imgs.shape[3]),dtype=train_imgs.dtype)
    #r = np.zeros((train_imgs.shape[0],1,train_imgs.shape[2],train_imgs.shape[3]),dtype=train_imgs.dtype)
    #b[:,0,:,:] = train_imgs[:,0,:,:]
    #g[:,0,:,:] = train_imgs[:,1,:,:]
    #r[:,0,:,:] = train_imgs[:,2,:,:]
    #print(b.shape)
    #print(g.shape)
    #print(r.shape)
    #train_imgs_b = dataset_normalized(b, train_test)
    #train_imgs_b = clahe_equalized(train_imgs_b, train_test)
    #train_imgs_b = adjust_gamma(train_imgs_b, 1.2, train_test)
    #train_imgs_b = train_imgs_b/255.  #reduce to 0-1 range
    #
    #train_imgs_g = dataset_normalized(g, train_test)
    #train_imgs_g = clahe_equalized(train_imgs_g, train_test)
    #train_imgs_g = adjust_gamma(train_imgs_g, 1.2, train_test)
    #train_imgs_g = train_imgs_g/255.  #reduce to 0-1 range
    #
    #train_imgs_r = dataset_normalized(r, train_test)
    #train_imgs_r = clahe_equalized(train_imgs_r, train_test)
    #train_imgs_r = adjust_gamma(train_imgs_r, 1.2, train_test)
    #train_imgs_r = train_imgs_r/255.  #reduce to 0-1 range
    #print(train_imgs_b.shape)
    #print(train_imgs_g.shape)
    #print(train_imgs_r.shape)
    #train_imgs = np.zeros((train_imgs.shape[0],3,train_imgs.shape[2],train_imgs.shape[3]),dtype=train_imgs.dtype)
    #train_imgs[:,0:0,:,:] = train_imgs_b
    #train_imgs[:,1:1,:,:] = train_imgs_g
    #train_imgs[:,2:2,:,:] = train_imgs_r
    ##train_imgs = np.dstack([train_imgs_b,train_imgs_g,train_imgs_r])  
    #print(train_imgs.shape)
    
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    #bn_imgs = rgb[:,1,:,:]
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#==== histogram equalization
def histo_equalized(imgs, train_test = "train"):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    histo_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        histo_imgs[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/histo_imgs_" + str(i) + ".jpg",np.array(histo_imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/histo_imgs_" + str(i) + ".jpg",np.array(histo_imgs[i,0], dtype = np.uint8))
    return histo_imgs


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs, train_test = "train"):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        clahe_imgs[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/clahe_imgs_" + str(i) + ".jpg",np.array(clahe_imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/clahe_imgs_" + str(i) + ".jpg",np.array(clahe_imgs[i,0], dtype = np.uint8))
        #print("clahe_imgs[" + str(i) + ",0].shape")
        #print(clahe_imgs[i,0].shape)
    return clahe_imgs


# ===== normalize over the dataset
def dataset_normalized(imgs, train_test = "train"):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/gray_" + str(i) + ".jpg",np.array(imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/gray_" + str(i) + ".jpg",np.array(imgs[i,0], dtype = np.uint8))
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/normalized_" + str(i) + ".jpg",np.array(imgs_normalized[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/normalized_" + str(i) + ".jpg",np.array(imgs_normalized[i,0], dtype = np.uint8))
        #print("imgs_normalized[" + str(i) + "].shape")
        #print(imgs_normalized[i].shape)
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0, train_test = "train"):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    gamma_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        gamma_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/gamma_img_" + str(i) + ".jpg",np.array(gamma_imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/gamma_img_" + str(i) + ".jpg",np.array(gamma_imgs[i,0], dtype = np.uint8))
        #print("gamma_imgs[" + str(i) + ",0].shape")
        #print(gamma_imgs[i,0].shape)
    return gamma_imgs




def FTSalience(imgs, train_test = "train"):
    # print("imgs.shape")
    # print(imgs.shape)
    FT_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        FT_img = np.array(imgs[i,0], dtype = np.uint8)
        # print("FT_img.shape")
        # print(FT_img.shape)
        img_mean = np.mean(FT_img)
        FT_img = cv2.GaussianBlur(FT_img, (5,5), 1.5, 1.5)
        FT_imgs[i,0] = abs(FT_img-0.2*img_mean)
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/FT_img_" + str(i) + ".jpg",np.array(FT_imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/FT_img_" + str(i) + ".jpg",np.array(FT_imgs[i,0], dtype = np.uint8))
    return FT_imgs

def ImgEnhance(imgs, train_test = "train"):
    # print("imgs.shape")
    # print(imgs.shape)
    Enhance_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        Src = np.array(imgs[i,0], dtype = np.uint8)
        # print("Src.shape")
        # print(Src.shape)
        srcGauss = cv2.GaussianBlur(Src, (5,5), 1.5, 1.5)
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/srcGauss_" + str(i) + ".jpg",np.array(srcGauss, dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/srcGauss_" + str(i) + ".jpg",np.array(srcGauss, dtype = np.uint8))
        cv2.Laplacian(srcGauss,-1, srcGauss, 3, 1.0, 1.0)
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/GaussLaplacian_" + str(i) + ".jpg",np.array(srcGauss, dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/GaussLaplacian_" + str(i) + ".jpg",np.array(srcGauss, dtype = np.uint8))
        #Enhance_imgs[i,0] = Src-srcGauss
        Enhance_imgs[i,0] = cv2.subtract(Src, srcGauss)
        if train_test=="train":
            cv2.imwrite("prePrpImgs/train/Enhance_imgs_" + str(i) + ".jpg",np.array(Enhance_imgs[i,0], dtype = np.uint8))
        elif train_test=="test":
            cv2.imwrite("prePrpImgs/test/Enhance_imgs_" + str(i) + ".jpg",np.array(Enhance_imgs[i,0], dtype = np.uint8))
    return Enhance_imgs



