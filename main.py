import os
import numpy as np
import tensorflow as tf
import time

from tqdm import tqdm

import ldr2hdr
import skynet
import grad_cam as gc

import utils
from random_tone_map import random_tone_map
import matplotlib.pyplot as plt

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    filters = np.transpose(filters, [0, 3, 1, 2])
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], interpolation='nearest')
    plt.show()

def gradcam_show(filters, mpath, index, nx=8):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    # filters = np.transpose(filters, [0, 3, 1, 2])
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i], interpolation='nearest')
    # plt.show()


    plt.savefig("{}/{}.png".format(mpath, index))


AUTO = tf.data.AUTOTUNE

# Hyper parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SKYNET_EPOCHS = 5000
LDR2HDR_EPOCHS = 1000
EARLY_STOPPPING = 0.0135
FC_LAYER_DIM = 64
IMSHAPE = (32,128,3)
RENDERSHAPE = (64,64,3)

TRAIN_SKYNET = False
TRAIN_LDR2PSEUDOHDR = True

# Tone Mapping Operators
TMO = ['exposure', 'reinhard', 'mantiuk', 'drago']

CURRENT_WORKINGDIR = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "/home/cvnar2/shinywings/research/sky_ldr2hdr/DataGeneration/dataset/tfrecord")

SKYNET_PRETRAINED_DIR = None # None
LDR2HDR_PRETRAINED_DIR = None # None

ENCODING_STYLE = "utf-8"
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

"""RGB to Lab"""
# def toLAB(Y):
#     t = Y # reference white : 1  / Don't panic with the Wikipedia Y_n = 100
    
#     for i in range(t.shape[0]):
#         for j in range(t.shape[1]):
#             t[i,j] = np.power(t[i,j], 1/3) if t[i,j] > 0.008856452 else t[i,j]/0.12841855 + 4/29
    
#     return 116*t-16
# 1024 2048 -> 512 1024  
# Create grid and multivariate normal

# hdr = 0.2627*img[:,:,2] + 0.6780*img[:,:,1] + 0.0593*img[:,:,0]
# hdr = toLAB(hdr)

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'render': tf.io.FixedLenFeature([], tf.string),
        'azimuth' : tf.io.FixedLenFeature([], tf.float32),
        'elevation' : tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    hdr = tf.io.decode_raw(example['image'], np.float32)
    hdr = tf.reshape(hdr, IMSHAPE)

    render_img = tf.io.decode_raw(example['render'], np.float32)
    render_img = tf.reshape(render_img, RENDERSHAPE)

    azimuth= example['azimuth']
    elevation= example['elevation']
   
    return hdr, render_img, [azimuth, elevation]

def configureDataset(dirpath, train= "train"):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecord"), shuffle=False)
    tfrecords_list.extend(a)

    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)

    if train:
        ds = ds.shuffle(buffer_size = 10000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    else:
        ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        
    return ds

def hdr_logCompression(x, validDR = 5000.):

    # disentangled way
    x = tf.math.multiply(validDR, x)
    numerator = tf.math.log(1.+ x)
    denominator = tf.math.log(1.+validDR)
    output = tf.math.divide(numerator, denominator) - 1.

    return output

def hdr_logDecompression(x, validDR = 5000.):

    x = x + 1.
    denominator = tf.math.log(1.+validDR)
    x = tf.math.multiply(x, denominator)
    x = tf.math.exp(x)
    output = tf.math.divide(x, validDR)
    
    return output

def _tone_mapping(hdrs):
    ldrs = []
    for hdr in hdrs:

        choice = np.random.randint(0, len(TMO))

        if TMO[choice] == "exposure":
            pe = utils.PercentileExposure()
            ldr = pe(hdr)
        else:
            try:
                ldr = random_tone_map(hdr.numpy(), TMO[choice])
            except:
                ldr = np.zeros_like(hdr.numpy())

        # Normalize to [-1..1]
        ldr = ldr.astype(np.float32)
        norm_ldr = tf.subtract(tf.divide(ldr,127.5),1.)
        ldrs.append(norm_ldr)

    ldrs = tf.convert_to_tensor(ldrs, dtype=tf.float32)
    return ldrs

def createDirectories(path, name="name", dir="dir"):
    
    path = utils.createNewDir(path, dir)
    root_logdir = utils.createNewDir(path, name)
    logdir = utils.createNewDir(root_logdir)

    if dir=="tensorboard":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=False)
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        test_summary_writer = tf.summary.create_file_writer(test_logdir)
        return train_summary_writer, test_summary_writer, logdir

    if dir=="outputImg":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=True)
        return train_logdir, test_logdir

if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    """Path for tf.summary.FileWriter and to store model checkpoints"""
    root_dir=os.getcwd()
    
    train_summary_writer_ldr2hdr, test_summary_writer_ldr2hdr, logdir_ldr2hdr = createDirectories(root_dir, name="ldr2hdr", dir="tensorboard")

    print('tensorboard --logdir={}'.format(logdir_ldr2hdr))

    """"Create Output Image Directory"""
    if(TRAIN_LDR2PSEUDOHDR):
        train_outImgDir_ldr2hdr, test_outImgDir_ldr2hdr = createDirectories(root_dir, name="ldr2hdr", dir="outputImg")
        train_gradcam, test_gradcam = createDirectories(root_dir, name="gradcam", dir="outputImg")
    """Init Dataset"""
    ldr2hdr_train_ds = configureDataset(TRAIN_DIR, train=True)
    ldr2hdr_test_ds  = configureDataset(TEST_DIR, train=False)

    """
    Model initialization
    """
    # ldr2hdr
    optimizer_ldr2hdr = tf.keras.optimizers.Adam(LEARNING_RATE) # TODO change SGD
    train_loss_ldr2hdr = tf.keras.metrics.Mean(name= 'train_loss_ldr2hdr', dtype=tf.float32)
    test_loss_ldr2hdr = tf.keras.metrics.Mean(name='test_loss_ldr2hdr', dtype=tf.float32)

    _skynet  = skynet.model(fc_dim=FC_LAYER_DIM)
    # _ldr2hdr = ldr2hdr.model(fc_dim=FC_LAYER_DIM, imshape=IMSHAPE[:2], deconv_method='resize')
    
    """
    Check out the dataset that properly work
    """
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,20))
    # for i, (image, means) in enumerate(train_ds.take(25)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image[i])
    #     plt.axis('off')
    # plt.show()

    """
    CheckPoint Create
    """
    checkpoint_path = utils.createNewDir(root_dir, "checkpoints")
    
    # ldr2hdr
    if LDR2HDR_PRETRAINED_DIR is None:
        checkpoint_path_ldr2hdr = utils.createNewDir(checkpoint_path, "ldr2hdr")
    else: checkpoint_path_ldr2hdr = LDR2HDR_PRETRAINED_DIR
    
    ckpt_ldr2hdr = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            ldr2hdr=_skynet,
                           optimizer=optimizer_ldr2hdr,) # TODO Insert Iterator object

    ckpt_manager_ldr2hdr = tf.train.CheckpointManager(ckpt_ldr2hdr, checkpoint_path_ldr2hdr, max_to_keep=5)

    #  if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_ldr2hdr.latest_checkpoint and not TRAIN_LDR2PSEUDOHDR:
        ckpt_ldr2hdr.restore(ckpt_manager_ldr2hdr.latest_checkpoint)
        print('Latest ldr2hdr checkpoint has restored!!')

    with tf.device('/GPU:0'):

        """
        ldr to pseudo-hdr
        """
        @tf.function
        def train_step(src_hdrs):
            
            hdrs = hdr_logCompression(src_hdrs) # to [-1..1]

            with tf.GradientTape() as tape_ldr2hdr:
                
                outImg, A_k, y_c= _skynet(hdrs, training=True)
                
                # pred_hdrs = hdr_logDecompression(outImg)
                # outRndr = _skynet.render_scene(pred_hdrs, training=True)
                
                l1_loss = tf.reduce_mean(tf.abs(hdrs - outImg))
                # rndr_loss = tf.reduce_mean(tf.square(rndrs - outRndr))
                # combine_loss = l1_loss
            
            grads = tf.gradients(y_c, A_k)
            gradients_ldr2hdr = tape_ldr2hdr.gradient(l1_loss, _skynet.trainable_variables)
            optimizer_ldr2hdr.apply_gradients(zip(gradients_ldr2hdr, _skynet.trainable_variables))
            train_loss_ldr2hdr(l1_loss)

            return outImg, A_k, y_c, grads

        @tf.function
        def test_step(test_src_hdrs):
            
            test_hdrs = hdr_logCompression(test_src_hdrs) # to [-1..1]

            outImg_test,  A_k, y_c = _skynet(test_hdrs, training= False)
            
            # pred_test_hdrs = hdr_logDecompression(outImg_test)
            # outRndr_test = _skynet.render_scene(pred_test_hdrs, training=False)

            l1_loss = tf.reduce_mean(tf.abs(test_hdrs - outImg_test))
            # rndr_loss = tf.reduce_mean(tf.square(rndrs - outRndr_test))
                       
            grads = tf.gradients(y_c, A_k)
            test_loss_ldr2hdr(l1_loss)

            return outImg_test, A_k, y_c, grads
    
    print("시작")
    
    """
    Sub module 2. train ldr2hdr (get pseudo-hdr)
    &
    Sub module 3. train pseudo-HDR encoder (get sky parameters) then compare the output to the ground-truth sky parameters, and generate real HDR with skynet decoder
    """
    
    if(TRAIN_LDR2PSEUDOHDR):
        isFirst = True

        for epoch in range(LDR2HDR_EPOCHS):

            start = time.perf_counter()

            train_loss_ldr2hdr.reset_states()
            test_loss_ldr2hdr.reset_states()
            
            for step, (hdrs, rndrs, poses) in enumerate(tqdm(ldr2hdr_train_ds)):
                
                ldrs = tf.py_function(_tone_mapping, [hdrs], [tf.float32])[0]
                
                pseudoHDR, A_k, y_c, grads = train_step(hdrs)

            # TODO test
            ldrs = (ldrs + 1.)*127.5
            ldrs = ldrs.numpy()
            ldrs = ldrs[:,:,:,::-1].astype(np.uint8)
            
            cam = gc.grad_cam(y_c, A_k, grads, ldrs)
            # cv2.imwrite("gradcam.jpg", cam)
            gradcam_show(cam, train_gradcam, epoch)

            with train_summary_writer_ldr2hdr.as_default():
                tf.summary.scalar('loss', train_loss_ldr2hdr.result(), step=epoch+1)
            
            for step, (hdrs, rndrs, poses) in enumerate(tqdm(ldr2hdr_test_ds)):
                
                ldrs = tf.py_function(_tone_mapping, [hdrs], [tf.float32])[0]
                
                pseudoHDR, A_k, y_c, grads = test_step(hdrs)
                
            # TODO test
            ldrs = (ldrs + 1.)*127.5
            ldrs = ldrs.numpy()
            ldrs = ldrs[:,:,:,::-1].astype(np.uint8)
            camt = gc.grad_cam(y_c, A_k, grads, ldrs)
            # cv2.imwrite("gradcam.jpg", cam)
            gradcam_show(camt, test_gradcam, epoch)

            with test_summary_writer_ldr2hdr.as_default():
                tf.summary.scalar('loss', test_loss_ldr2hdr.result(), step=epoch+1)
            
            if isFirst:
                isFirst = False
                groundtruth_dir = utils.createNewDir(test_outImgDir_ldr2hdr, "groundTruth")
                if not os.listdir(groundtruth_dir):
                    for i in range(hdrs.get_shape()[0]):
                        utils.writeHDR(hdrs[i].numpy(), "{}/{}_gt.{}".format(groundtruth_dir,i,HDR_EXTENSION), hdrs.get_shape()[1:3])

            if (epoch+1) % 10 == 0:
                pseudoHDR = hdr_logDecompression(pseudoHDR)
                for i in range(pseudoHDR.get_shape()[0]):
                    pseudoHDR_epoch_dir = utils.createNewDir(test_outImgDir_ldr2hdr, "{}Epoch_pseudoHDR".format(epoch+1))
                    utils.writeHDR(pseudoHDR[i].numpy(), "{}/{}.{}".format(pseudoHDR_epoch_dir,i,HDR_EXTENSION), pseudoHDR.get_shape()[1:3])
            
            ckpt_ldr2hdr.epoch.assign_add(1)

            if int(ckpt_ldr2hdr.epoch) % 5 == 0:
                save_path =  ckpt_manager_ldr2hdr.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt_ldr2hdr.epoch), save_path))
            
            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss_ldr2hdr.result(), test_loss_ldr2hdr.result()))
            
            print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))
    
    print("끝")