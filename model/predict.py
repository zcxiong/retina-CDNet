###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import configparser
import time
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
from get_imagedata import *


#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')
with_mash = config.getboolean('data attributes', 'with_mash')
permute1D = config.getboolean('data attributes', 'permute1D')
#with_GT = config.getboolean('testing settings', 'with_GT')

#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#the border masks provided by the DRIVE

if with_mash:
    test_border_masks_path = config.get('data paths', 'test_border_masks')
    test_border_masks_path = path_data + test_border_masks_path
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
batchsize = int(config.get('data attributes', 'batch_size'))
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if with_mash:
    print("Using mask images.")
    patches_imgs_test, patches_masks_test, test_border_masks = get_data_testing(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        DRIVE_test_border = test_border_masks_path,
        batch_h = patch_height,
        batch_w = patch_width
    )
else:
    print("Without using mask images.")
    patches_imgs_test, patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        DRIVE_test_border = "",
        batch_h = patch_height,
        batch_w = patch_width
    )

print("patches_imgs_test shape :", patches_imgs_test.shape)
print("patches_masks_test shape :", patches_masks_test.shape)


#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
#Load the saved model
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=batchsize, verbose=2)
# times = 0
# for i in range(10):
#     start = time.clock()
#     predictions = model.predict(patches_imgs_test, batch_size=batchsize, verbose=2)
#     end = time.clock()
#     duration = end - start
#     print("\nrun time: ", duration)
#     times += duration
# print("\nrun times: ", times / 20 / 10)
# print("predicted images size :", predictions.shape)

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions,"original1",patches_imgs_test.shape[2],patches_imgs_test.shape[3], permute1D)
print("pred_patches shape:", pred_patches.shape)

visualize(group_images(patches_imgs_test,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(patches_masks_test,N_visual),path_experiment+"all_groundTruths")#.show()
visualize(group_images(pred_patches,N_visual),path_experiment+"all_predictions")

N_predicted = patches_imgs_test.shape[0]
group = N_visual
assert (N_predicted%group==0)

print("pred_patches max = ", np.max(pred_patches))
print("pred_patches min = ", np.min(pred_patches))
print("patches_masks_test max = ", np.max(patches_masks_test))
print("patches_masks_test min = ", np.min(patches_masks_test))
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(patches_imgs_test[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(patches_masks_test[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_patches[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))


print("pred_patches.shape = ", pred_patches.shape)
print("patches_masks_test.shape = ", patches_masks_test.shape)
if with_mash:
    print("test_border_masks.shape = ", test_border_masks.shape)
    y_scores, y_true = pred_only_FOV_with_mask(pred_patches,patches_masks_test, test_border_masks)
else:
    y_scores, y_true = pred_only_FOV(pred_patches,patches_masks_test)
print("Calculating results only inside the FOV:")
print("y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_patches.shape[0]*pred_patches.shape[2]*pred_patches.shape[3]) +" (584*565==329960)")
print("y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(patches_masks_test.shape[2]*patches_masks_test.shape[3]*patches_masks_test.shape[0])+" (584*565==329960)")

#Area under the ROC curve
print("y_true=", y_true)
print("y_scores=", y_scores)
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
print("fpr,tpr:", fpr.shape,tpr.shape)
np.savetxt(path_experiment+'fpr.txt',fpr,fmt='%0.8f')
np.savetxt(path_experiment+'tpr.txt',tpr,fmt='%0.8f')

AUC_ROC = roc_auc_score(y_true, y_scores)
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
np.savetxt(path_experiment+'recall.txt',recall,fmt='%0.8f')
np.savetxt(path_experiment+'precision.txt',precision,fmt='%0.8f')
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " +str(F1_score))


#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
				# +"\nRUNTIME: " +str(end - start) + 's'
                )
file_perf.close()
