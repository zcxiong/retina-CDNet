###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################

from get_NN_Model import *
from keras import backend as BK

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from get_imagedata import get_data_training


BK.set_image_data_format("channels_first")
BK.set_image_dim_ordering("th")

acc_loss = 0
acc_loss_list = []

#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
if not os.path.exists('./'+name_experiment):
    os.makedirs('./'+name_experiment)

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('data attributes', 'batch_size'))

DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth')#masks
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
permute1D = config.getboolean('data attributes', 'permute1D')

#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
    batch_h = patch_height,
    batch_w = patch_width
)

patches_imgs_test, patches_masks_test = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'test_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),
    batch_h = patch_height,
    batch_w = patch_width
)

N_sample = min(patches_imgs_train.shape[0],36)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],6),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],6),'./'+name_experiment+'/'+"sample_input_masks")#.show()

print("patches_imgs_train.shape = ", patches_imgs_train.shape)
print("patches_masks_train.shape = ", patches_masks_train.shape)
patches_masks_train = masks_Unet(patches_masks_train, permute1D)
patches_masks_test = masks_Unet(patches_masks_test, permute1D)
print("patches_masks_train.shape = ", patches_masks_train.shape)



n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_model_by_name(n_ch, patch_height, patch_width, name_experiment)
print("Check: final output of the network:")
print(model.output_shape)
#plot_model(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png', show_shapes=True, show_layer_names = True)   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

#============  Training ==================================
#============  CallBacks ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
class epochRecord(Callback):
    #def on_batch_begin(self, batch, logs={}):
    #    print(batch)
        
    def on_epoch_end(self, epoch, logs={}):
        #global_list.myEpochRecord = global_list.myEpochRecord + 1
        #model.save_weights('./'+name_experiment+'/'+name_experiment + str(epoch) +'_weights_loss' + str(logs.get('loss')) + '_vloss' + str(logs.get('val_loss')) + '.h5', overwrite=True)
        #print logs.get('lr')
        model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        file_perf = open('./'+name_experiment+'/time.txt', 'a')
        file_perf.write('epoch:' + str(epoch) + ',')
        file_perf.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        file_perf.write(':loss='+ str(logs.get('loss')) + ',val_loss=' + str(logs.get('val_loss')) + ',acc=' + str(logs.get('acc')) + ',val_acc=' + str(logs.get('val_acc')))
        file_perf.write('\n')
        file_perf.close()
        
        acc = logs.get('acc')
        file_acc = open('./'+name_experiment+'/log_acc.txt', 'a')
        file_acc.write(str(acc) + ',')
        file_acc.close()
        val_acc = logs.get('val_acc')
        file_val_acc = open('./'+name_experiment+'/log_val_acc.txt', 'a')
        file_val_acc.write(str(val_acc) + ',')
        file_val_acc.close()

        loss = logs.get('loss')
        file_loss = open('./'+name_experiment+'/log_loss.txt', 'a')
        file_loss.write(str(loss) + ',')
        file_loss.close()
        val_loss = logs.get('val_loss')
        file_val_loss = open('./'+name_experiment+'/log_val_loss.txt', 'a')
        file_val_loss.write(str(val_loss) + ',')
        file_val_loss.close()
        acc_val_acc = acc * 0.3 + val_acc * 0.7
        loss_val_loss = loss * 0.3 + val_loss * 0.7
        global acc_loss
        global acc_loss_list
        acc_loss_list.append(acc_val_acc / loss_val_loss)
        print('epoch ' + str(epoch) + ': acc*0.3+val_acc*0.7 = ' + str(acc_val_acc) + ', loss*0.3+val_loss*0.7 = ' + str(loss_val_loss) + ', acc/loss = ' + str(acc_val_acc / loss_val_loss))
        if (acc_val_acc / loss_val_loss) > acc_loss:
            print('acc/loss improved from ', str(acc_loss), ' to ', str(acc_val_acc / loss_val_loss))
            model.save_weights('./'+name_experiment+'/' + name_experiment + '_maxAcc_minLoss_weights.h5', overwrite=True)
            acc_loss = acc_val_acc / loss_val_loss
        else:
            print('acc/loss didnot improve from ', str(acc_loss))
        if (acc_val_acc / loss_val_loss) > 10:
            model.save_weights('./'+name_experiment+'/'+name_experiment + str(epoch) +'_weights_loss' + str(logs.get('loss')) + '_vloss' + str(logs.get('val_loss')) + '.h5', overwrite=True)
        file_loss = open('./'+name_experiment+'/acc_loss.txt', 'a')
        file_loss.write('epoch' + str(epoch) + ':acc = ' + str(acc) + '-')
        file_loss.write('val_acc = ' + str(val_acc) + '-')
        file_loss.write('loss = ' + str(loss) + '-')
        file_loss.write('val_loss = ' + str(val_loss) + '-')
        file_loss.write('acc * 0.3 + val_acc * 0.7 = ' + str(acc_val_acc) + '-')
        file_loss.write('loss * 0.3 + val_loss * 0.7 = ' + str(loss_val_loss) + '-')
        file_loss.write('acc / loss = ' + str(acc_val_acc / loss_val_loss) + '\n')
        file_loss.close()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=16, min_lr=5e-4)

er = epochRecord()

es = EarlyStopping(monitor='val_loss', patience=64)

print("patches_imgs_train.shape=" + str(patches_imgs_train.shape))
print("patches_masks_train.shape=" + str(patches_masks_train.shape))
history = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.0, validation_data = (patches_imgs_test, patches_masks_test), callbacks=[checkpointer, er, reduce_lr, es])

file_perf = open('./'+name_experiment+'/log.txt', 'w')
file_perf.write(str(history.history))
file_perf.close()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure(1)
plt.plot(acc,"*-",label="acc")
plt.plot(val_acc,".-",label="val_acc")
plt.title('acc of training')
plt.legend(loc="lower right")
plt.savefig('./'+name_experiment+'/'+'acc.png')

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(2)
plt.plot(loss,"*-",label="loss")
plt.plot(val_loss,".-",label="val_loss")
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.title('loss of training')
plt.savefig('./'+name_experiment+'/'+'loss.png')

plt.figure(3)
plt.plot(acc_loss_list,".-",label="acc_loss")
plt.legend(loc="lower right")
plt.title('acc_loss of training')
plt.savefig('./'+name_experiment+'/'+'acc_loss.png')

#========== Save and test the last model ===================
#model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
