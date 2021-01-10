 #==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image, ImageOps



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test1 = "./DRIVE/test/1st_manual/"
groundTruth_imgs_test2 = "./DRIVE/test/2nd_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./datasets_drive/"
AngleStep = 15
AngleNum = int(360 / AngleStep)
MultipleNum = AngleNum*2
UpsetOrder = False

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    imgs = None
    groundTruth = None
    border_masks = None
    if train_test=="train":
        imgs = np.empty((Nimgs*MultipleNum,height,width,channels))
        groundTruth = np.empty((Nimgs*MultipleNum,1,height,width))
        border_masks = np.empty((Nimgs*MultipleNum,1,height,width))
    else:
        imgs = np.empty((Nimgs,height,width,channels))
        groundTruth = np.empty((Nimgs,1,height,width))
        border_masks = np.empty((Nimgs,1,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            #img = img.resize((width, height),Image.ANTIALIAS)
            if train_test=="train" and UpsetOrder:
                imgs[i] = np.asarray(img)
                imgs[i + 20] = np.asarray(img.rotate(90))
                imgs[i + 40] = np.asarray(img.rotate(180))
                imgs[i + 60] = np.asarray(img.rotate(-90))
                mirro = ImageOps.mirror(img)
                imgs[i + 80] = np.asarray(mirro)
                imgs[i + 100] = np.asarray(mirro.rotate(90))
                imgs[i + 120] = np.asarray(mirro.rotate(180))
                imgs[i + 140] = np.asarray(mirro.rotate(-90))           
            elif train_test=="train" and not UpsetOrder:
                for j in range(AngleNum):
                    imgs[i*MultipleNum + j] = np.asarray(img.rotate(AngleStep*j))
                mirro = ImageOps.mirror(img)
                for j in range(AngleNum):
                    imgs[i*MultipleNum + j + AngleNum] = np.asarray(mirro.rotate(AngleStep*j))
            else:
                imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = ""
            if train_test=="train":
                groundTruth_name = files[i][0:2] + "_manual1.gif"
            elif train_test=="test1":
                groundTruth_name = files[i][0:2] + "_manual1.gif"
            elif train_test=="test2":
                groundTruth_name = files[i][0:2] + "_manual2.gif"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            #g_truth = g_truth.resize((width, height),Image.ANTIALIAS)            
            if train_test=="train" and UpsetOrder:
                groundTruth[i] = np.asarray(g_truth)
                groundTruth[i + 20] = np.asarray(g_truth.rotate(90))
                groundTruth[i + 40] = np.asarray(g_truth.rotate(180))
                groundTruth[i + 60] = np.asarray(g_truth.rotate(-90))
                mirro = ImageOps.mirror(g_truth)
                groundTruth[i + 80] = np.asarray(mirro)
                groundTruth[i + 100] = np.asarray(mirro.rotate(90))
                groundTruth[i + 120] = np.asarray(mirro.rotate(180))
                groundTruth[i + 140] = np.asarray(mirro.rotate(-90))            
            elif train_test=="train" and not UpsetOrder:
                for j in range(AngleNum):
                    groundTruth[i*MultipleNum + j] = np.asarray(g_truth.rotate(AngleStep*j))
                mirro = ImageOps.mirror(g_truth)
                for j in range(AngleNum):
                    groundTruth[i*MultipleNum + j + AngleNum] = np.asarray(mirro.rotate(AngleStep*j))
            else:
                groundTruth[i,0] = np.asarray(g_truth)
                print("groundTruth[", i, ",0] max: " +str(np.max(groundTruth[i,0])))
            if train_test=="test2":
                groundTruth[i,0] = groundTruth[i,0] * 255
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test1":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            elif train_test=="test2":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            #b_mask = b_mask.resize((width, height),Image.ANTIALIAS)
            if train_test=="train" and UpsetOrder:
                border_masks[i] = np.asarray(b_mask)
                border_masks[i + 20] = np.asarray(b_mask.rotate(90))
                border_masks[i + 40] = np.asarray(b_mask.rotate(180))
                border_masks[i + 60] = np.asarray(b_mask.rotate(-90))
                mirro = ImageOps.mirror(b_mask)
                border_masks[i + 80] = np.asarray(mirro)
                border_masks[i + 100] = np.asarray(mirro.rotate(90))
                border_masks[i + 120] = np.asarray(mirro.rotate(180))
                border_masks[i + 140] = np.asarray(mirro.rotate(-90))          
            elif train_test=="train" and not UpsetOrder:
                for j in range(AngleNum):
                    border_masks[i*MultipleNum + j] = np.asarray(b_mask.rotate(AngleStep*j))
                mirro = ImageOps.mirror(b_mask)
                for j in range(AngleNum):
                    border_masks[i*MultipleNum + j + AngleNum] = np.asarray(mirro.rotate(AngleStep*j))
            else:
                border_masks[i,0] = np.asarray(b_mask)

    print("groundTruth max: " +str(np.max(groundTruth)))
    print("groundTruth min: " +str(np.min(groundTruth)))
    print("border_masks max: " +str(np.max(border_masks)))
    print("border_masks min: " +str(np.min(border_masks)))
    assert(np.max(groundTruth)==255)
    assert(np.max(border_masks)==255)
    assert(np.min(groundTruth)==0)
    assert(np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    print("imgs.shape", imgs.shape)
    print("groundTruth.shape", groundTruth.shape)
    print("border_masks.shape", border_masks.shape)
    if train_test=="train":
        assert(imgs.shape == (Nimgs*MultipleNum,channels,height,width))
        assert(groundTruth.shape == (Nimgs*MultipleNum,1,height,width))
        assert(border_masks.shape == (Nimgs*MultipleNum,1,height,width))
        groundTruth = np.reshape(groundTruth,(Nimgs*MultipleNum,1,height,width))
        border_masks = np.reshape(border_masks,(Nimgs*MultipleNum,1,height,width))
    else:
        assert(imgs.shape == (Nimgs,channels,height,width))
        assert(groundTruth.shape == (Nimgs,1,height,width))
        assert(border_masks.shape == (Nimgs,1,height,width))
        groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
        border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
del imgs_train
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
del groundTruth_train
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")
del border_masks_train

# #getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test1,borderMasks_imgs_test,"test1")
print("saving test1 datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
del imgs_test
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test1.hdf5")
del groundTruth_test
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
del border_masks_test

imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test2,borderMasks_imgs_test,"test2")
print("saving test2 datasets")
del imgs_test
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test2.hdf5")
del groundTruth_test
del border_masks_test
