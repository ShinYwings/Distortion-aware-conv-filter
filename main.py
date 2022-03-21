import tensorflow as tf
import os
import sys
import model as model
import time
from datetime import datetime as dt
import optimizer_alexnet
import threading
# import progressbar
import math

# Hyper parameters
LEARNING_RATE = 0.02
NUM_EPOCHS = 90
NUM_CLASSES = 10    # IMAGENET 2012   # 모델에는 따로 선언해줌
MOMENTUM = 0.9 # SGD + MOMENTUM
BATCH_SIZE = 64 # 128 batches occurs OOM in my computer

DATASET_DIR = "/media/shin/2nd_m.2/ILSVRC2012"

# Input으로 넣을 데이터 선택
indexsub = 440
# indexsub = 441    # q95로 하면 1 더 빼줌
TRAIN_DATASET =  "/media/shin/2nd_m.2/ILSVRC2012/class10_tfrecord_train"
TEST_DATASET = "/media/shin/2nd_m.2/ILSVRC2012/class10_tfrecord_val"

RUN_TRAIN_DATASET =  TRAIN_DATASET
RUN_TEST_DATASET = TEST_DATASET


# hands-on 에서는 r=2 a = 0.00002, b = 0.75, k =1 이라고 되어있음... 
# 문서에는 5, 1e-4, 0.75 2
LRN_INFO = (5, 1e-04, 0.75, 2) # radius, alpha, beta, bias   
INPUT_IMAGE_SIZE = 227 #WIDTH, HEIGHT    # cropped by 256x256 images
WEIGHT_DECAY = 5e-4
# Fixed
IMAGENET_MEAN = [122.10927936917298, 116.5416959998387, 102.61744377213829] # rgb format
ENCODING_STYLE = "utf-8"
AUTO = tf.data.experimental.AUTOTUNE

# widgets = [' [', 
#          progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
#          '] ', 
#            progressbar.Bar('/'),' (', 
#            progressbar.ETA(), ') ', 
#           ]
with tf.device('/CPU:0'):
    def img_preprocessing(q, images, labels, train = None):
        test_images = list()
        test_labels = list()
                
        for i in range(0,len(labels)):

            cropped_intend_image = image_cropping(images[i], training=train)

            for j in cropped_intend_image:
                test_images.append(j)
                test_labels.append(labels[i])

        q.append((test_images, test_labels))

def image_cropping(image , training = None):  # do it only in test time
    
    INPUT_IMAGE_SIZE = 227

    cropped_images = list()

    if training:

        intend_image = image
        
        horizental_fliped_image = tf.image.flip_left_right(intend_image)

        ran_crop_image1 = tf.image.random_crop(intend_image,size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
        ran_crop_image2 = tf.image.random_crop(horizental_fliped_image, 
                                    size=[INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3])
        cropped_images.append(tf.subtract(ran_crop_image1, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(ran_crop_image2, IMAGENET_MEAN))
        
    else:
        
        horizental_fliped_image = tf.image.flip_left_right(image)

        # for original image
        topleft = image[:227,:227]
        topright = image[29:,:227]
        bottomleft =image[:227,29:]
        bottomright = image[29:,29:]
        center = image[15:242, 15:242]

        cropped_images.append(tf.subtract(topleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(topright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(bottomleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(bottomright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(center, IMAGENET_MEAN))
        
        # for horizental_fliped_image
        horizental_fliped_image_topleft = horizental_fliped_image[:227,:227]
        horizental_fliped_image_topright = horizental_fliped_image[29:,:227]
        horizental_fliped_image_bottomleft = horizental_fliped_image[:227,29:]
        horizental_fliped_image_bottomright = horizental_fliped_image[29:,29:]
        horizental_fliped_image_center = horizental_fliped_image[15:242, 15:242]

        cropped_images.append(tf.subtract(horizental_fliped_image_topleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_topright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_bottomleft, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_bottomright, IMAGENET_MEAN))
        cropped_images.append(tf.subtract(horizental_fliped_image_center, IMAGENET_MEAN))
    
    return cropped_images

def get_logdir(root_logdir):
    run_id = dt.now().strftime("run_%Y_%m_%d-%H_%M_%S")
    
    return os.path.join(root_logdir, run_id)

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    raw_image= example['image']
    label= example['label']

    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.cast(image, tf.float32)
    #440은 imgnet metadata 상에 나와있는 index number 임. index 0부터 시작하게 만들려고 뺌
    label = tf.cast(tf.subtract(label,indexsub), tf.int32)
    return image, label

if __name__ == "__main__":
    
    root_dir=os.getcwd()
    dataset_dir=os.path.abspath(DATASET_DIR)
    sys.path.append(root_dir)
    sys.path.append(dataset_dir)

    """Path for tf.summary.FileWriter and to store model checkpoints"""
    filewriter_path = os.path.join(root_dir, "tensorboard")
    checkpoint_path = os.path.join(root_dir, "checkpoints")

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)
    
    root_logdir = os.path.join(filewriter_path, "logs")
    logdir = get_logdir(root_logdir)
    train_logdir = os.path.join(logdir, "train")
    val_logdir = os.path.join(logdir, "val") 

    train_tfrecord_list = list()
    test_tfrecord_list = list()

    train_dirs = os.listdir(RUN_TRAIN_DATASET)
    test_dirs = os.listdir(RUN_TEST_DATASET)
    
    for train_dir in train_dirs:
        dir_path = os.path.join(RUN_TRAIN_DATASET, train_dir)
        a =tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        train_tfrecord_list.extend(a)
    
    for test_dir in test_dirs:
        dir_path = os.path.join(RUN_TEST_DATASET, test_dir)
        b = tf.data.Dataset.list_files(os.path.join(dir_path, '*.tfrecord'))
        test_tfrecord_list.extend(b)

    train_buf_size = len(train_tfrecord_list)
    test_buf_size= len(test_tfrecord_list)
    print("train_buf_size", train_buf_size)
    print("test_buf_size", test_buf_size)
    train_ds = tf.data.TFRecordDataset(filenames=train_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    test_ds = tf.data.TFRecordDataset(filenames=test_tfrecord_list, num_parallel_reads=AUTO, compression_type="GZIP")
    train_ds = train_ds.shuffle(buffer_size=13000)
    train_ds = train_ds.map(_parse_function, num_parallel_calls=AUTO)
    test_ds = test_ds.map(_parse_function, num_parallel_calls=AUTO)
    train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    
    """
    Input Pipeline
    
    experimental: API for input pipelines
    cardinality: size of a set
        > in DB, 중복도가 낮으면 카디널리티가 높다. 중복도가 높으면 카디널리티가 낮다.
    """
    """
    [3 primary operations]
        1. Preprocessing the data within the dataset
        2. Shuffle the dataset
        3. Batch data within the dataset
    
    drop_ramainder: 주어진 dataset을 batch_size 나눠주고 
                    batch_size 만족 못하는 나머지들을 남길지 버릴지
    
    shuffle: Avoid local minima에 좋음
    
    prefetch(1): 데이터셋은 항상 한 배치가 미리 준비되도록 최선을 다합니다.
                 훈련 알고리즘이 한 배치로 작업을 하는 동안 이 데이터셋이 동시에 다음 배치를 준비
                 합니다. (디스크에서 데이터를 읽고 전처리)
    """

    learning_rate_fn = optimizer_alexnet.AlexNetLRSchedule(initial_learning_rate = LEARNING_RATE, name="performance_lr")
    _optimizer = optimizer_alexnet.AlexSGD(learning_rate=learning_rate_fn, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, name="alexnetOp")
    # _optimizer = tf.keras.optimizers.Adam()

    _model = model.mAlexNet(LRN_INFO, NUM_CLASSES)
    # 모델의 손실과 성능을 측정할 지표, 에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 결과 출력
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name= 'train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    top5_train_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    top5_test_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_test_accuracy')
    
    # NaN 발생이유 LR이 너무 높거나, 나쁜 초기화...

    train_summary_writer = tf.summary.create_file_writer(train_logdir)
    val_summary_writer = tf.summary.create_file_writer(val_logdir)
    
    print('tensorboard --logdir={}'.format(logdir))

    prev_test_accuracy = -1.

    with tf.device('/GPU:1'):
        @tf.function
        def train_step(images, labels):

            with tf.GradientTape() as tape:

                predictions = _model(images, training = True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, _model.trainable_variables)
            #apply gradients 가 v1의 minimize를 대체함
            _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)
            top5_train_accuracy(labels, predictions)

        @tf.function
        def test_step(test_images, test_labels):
            test_predictions = _model(test_images, training =False)
            t_loss = loss_object(test_labels, test_predictions)

            test_loss(t_loss)
            test_accuracy(test_labels, test_predictions)
            top5_test_accuracy(test_labels, test_predictions)
                # tf.cond(tf.less_equal(test_accuracy.result(),prev_test_accuracy.read_value()),
                #     learning_rate_fn.cnt_up_num_of_statinary_loss,
                #     lambda: None)
                # prev_test_accuracy.assign(test_accuracy.result())
                
    @tf.function
    def performance_lr_scheduling():
        learning_rate_fn.cnt_up_num_of_statinary_loss()

    @tf.function
    def termination_lr_scheduling():
        learning_rate_fn.turn_on_last_epoch_loss()

    print("시작")
    for epoch in range(NUM_EPOCHS):

        start = time.perf_counter()

        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        top5_train_accuracy.reset_states()
        top5_test_accuracy.reset_states()
        
        # bar = progressbar.ProgressBar(max_value= math.ceil(train_buf_size/BATCH_SIZE), widgets=widgets)
        # test_bar = progressbar.ProgressBar(max_value= math.ceil(test_buf_size/BATCH_SIZE), widgets=widgets)
        # bar.start()
        # test_bar.start()

        q = list()
        isFirst = True
        for step, (images, labels) in enumerate(train_ds):

            if isFirst:
                t = threading.Thread(target=img_preprocessing, args=(q, images, labels, True))
                t.start()
                t.join()
                isFirst = False
            else:
                train_images, train_labels = q.pop()
                batch_length = len(train_labels)
                train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                train_batch_ds = train_batch_ds.shuffle(batch_length)
                train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
                
                t = threading.Thread(target=img_preprocessing, args=(q, images, labels, True))
                t.start()
                for batch_size_images, batch_size_labels in train_batch_ds:
                    train_step(batch_size_images, batch_size_labels)
                t.join()
            
            if (epoch == (NUM_EPOCHS -1)) and step == 126:
                termination_lr_scheduling()
            # bar.update(step)
        
        # Last step
        train_images, train_labels = q.pop()
        batch_length = len(train_labels)
        
        train_batch_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_batch_ds = train_batch_ds.shuffle(batch_length)
        train_batch_ds = train_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        
        for batch_size_images, batch_size_labels in train_batch_ds:

            train_step(batch_size_images, batch_size_labels)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('accuracy', train_accuracy.result()*100, step=epoch+1)
            tf.summary.scalar('top5_accuracy', top5_train_accuracy.result()*100, step=epoch+1)

        q2 = list()
        isFirst = True
        for step, (images, labels) in enumerate(test_ds):

            if isFirst:
                t = threading.Thread(target=img_preprocessing, args=(q2, images, labels,False))
                t.start()
                t.join()
                isFirst = False
            else:
                test_images, test_labels = q2.pop()
                batch_length = len(test_labels)
                test_batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_batch_ds = test_batch_ds.shuffle(batch_length)
                test_batch_ds = test_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
                t = threading.Thread(target=img_preprocessing, args=(q2, images, labels, False))
                t.start()
                for batch_test_images, batch_test_labels in test_batch_ds:
                    test_step(batch_test_images, batch_test_labels)
                t.join()

            # test_bar.update(step)

        # Last step
        test_images, test_labels = q2.pop()
        batch_length = len(test_labels)
        test_batch_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_batch_ds = test_batch_ds.shuffle(batch_length)
        test_batch_ds = test_batch_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        
        for batch_test_images, batch_test_labels in test_batch_ds:
            test_step(batch_test_images, batch_test_labels)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch+1)
            tf.summary.scalar('accuracy', test_accuracy.result()*100, step=epoch+1)
            tf.summary.scalar('top5_accuracy', top5_test_accuracy.result()*100, step=epoch+1)
        print('Epoch: {}, Loss: {}, Accuracy: {}, top5_train_accuracy: {}, Test Loss: {}, Test Accuracy: {}, top5 Test Accuracy: {}'.format(epoch+1,train_loss.result(),
                            train_accuracy.result()*100, top5_train_accuracy.result()*100, test_loss.result(),test_accuracy.result()*100, top5_test_accuracy.result()*100))
        
        print("Spends time({}) in Epoch {}".format(epoch+1, time.perf_counter() - start))

        if prev_test_accuracy >= test_accuracy.result():
            performance_lr_scheduling()
        prev_test_accuracy = test_accuracy.result()
        

    print("끝")