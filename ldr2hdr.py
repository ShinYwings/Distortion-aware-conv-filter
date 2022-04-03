import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_addons as tfa
import ops

import filter as df

class transportMatrix(tf.keras.layers.Layer):
    def __init__(self, output_square_size=64, channel=3):
        super(transportMatrix, self).__init__()
        initializer = tf.random_normal_initializer()

        # 4096 = image h, w (32 x 128)
        self.oss = output_square_size
        self.oss_pow = tf.cast(output_square_size * output_square_size, tf.int32)
        self.w = tf.Variable(initial_value=initializer(shape=[self.oss_pow, self.oss_pow, channel], dtype=tf.float32), trainable=True)
    
    def call(self, x):
        input_b, _, _, input_c = x.get_shape().as_list() #bhwc
        x = tf.reshape(x, [input_b, self.oss_pow, 1, input_c])
        tm = tf.einsum("jkl,ikml->ijml",self.w, x)
        
        output = tf.reshape(tm, [input_b, self.oss, self.oss, input_c])

        return output

class resBlock(Model):

    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, strides=1):
        super(resBlock, self).__init__()

        self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.norm1 = tfa.layers.InstanceNormalization(axis=3,
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.elu1 = ops.elu()
        
        self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.norm2 = tfa.layers.InstanceNormalization(axis=3,
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.elu2 = ops.elu()
        

        if filter_in == filter_out:
            self.identity = lambda x : x
        else:
            self.identity = ops.conv2d(filter_out, k_h=1, k_w=1, strides=1)
    
    def call(self, x, training=False):

        conv1 = self.conv1(x)
        norm1   = self.norm1(conv1, training=training)
        elu1 = self.elu1(norm1)

        conv2 = self.conv2(elu1)
        norm2   = self.norm2(conv2, training=training)
        elu2 = self.elu2(norm2)

        return tf.add(self.identity(x), elu2)

class resLayer(Model):

    def __init__(self, filters, filter_in, k_h, k_w, strides):
        super(resLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in]+ list(filters), filters):
            self.sequence.append(resBlock(f_in, f_out, k_h=k_h, k_w=k_w, strides=strides))
    
    def call(self, x, training=False):
        for unit in self.sequence:
            x=unit(x, training=training)
        return x

class model(Model):
    def __init__(self, fc_dim="fc_dim", imshape="imshape", deconv_method="deconv_method"):

        super(model, self).__init__()

        self.deconv_method = deconv_method
        self.im_height = imshape[0]
        self.im_width = imshape[1]

        # Encoder 
        self.conv1 = ops.conv2d(output_channels=64, k_h=7, k_w=7, strides=2)
        self.bn1 = ops.batch_normalization()
        self.elu1 = ops.elu()
        
        self.conv2 = ops.conv2d(output_channels=128, k_h=5, k_w=5, strides=2)
        self.bn2 = ops.batch_normalization()
        self.elu2 = ops.elu()
        
        self.conv3 = ops.conv2d(output_channels=256, k_h=3, k_w=3, strides=2)
        self.bn3 = ops.batch_normalization()
        self.elu3 = ops.elu()
        
        self.conv4 = ops.conv2d(output_channels=256, k_h=3, k_w=3, strides=2)
        self.bn4 = ops.batch_normalization()
        self.elu4 = ops.elu()

        self.fc = ops.fc2d(fc_dim)
        self.bn5 = ops.batch_normalization()
        self.elu5 = ops.elu()
        self.dropout = ops.dropout(0.5)

        # Decoder
        # Original
        # self.defc = ops.dfc2d(out_height=self.conv4.get_shape()[1].value,
        #                         out_width=self.conv4.get_shape()[2].value,
        #                             out_channels=self.conv4.get_shape()[3].value)
        self.defc = ops.dfc2d(out_height=tf.cast(tf.divide(self.im_height, 16), dtype=tf.int32),
                                out_width=tf.cast(tf.divide(self.im_width, 16), dtype=tf.int32),
                                    out_channels=256)
        self.bn6 = ops.batch_normalization()
        self.elu6 = ops.elu()

        self.deconv4 = ops.deconv2d(output_channels=256, output_imshape=[tf.divide(self.im_height,8), tf.divide(self.im_width,8)], k_h=3, k_w=3, method=self.deconv_method)
        self.bn7 = ops.batch_normalization()
        self.elu7 = ops.elu()

        self.deconv3 = ops.deconv2d(output_channels=128, output_imshape=[tf.divide(self.im_height,4), tf.divide(self.im_width,4)], k_h=3, k_w=3, method=self.deconv_method)
        self.bn8 = ops.batch_normalization()
        self.elu8 = ops.elu()

        self.deconv2 = ops.deconv2d(output_channels=64, output_imshape=[tf.divide(self.im_height,2), tf.divide(self.im_width,2)], k_h=5, k_w=5, method=self.deconv_method)
        self.bn9 = ops.batch_normalization()
        self.elu9 = ops.elu()

        self.deconv1 = ops.deconv2d(output_channels=64, output_imshape=[self.im_height, self.im_width], k_h=7, k_w=7, method=self.deconv_method)
        self.bn10 = ops.batch_normalization()
        self.elu10 = ops.elu()

        self.out = ops.conv2d(output_channels=3, k_h=1, k_w=1, strides=1)
        self.tanh = ops.tanh()

        # 64 =  sqrt(128 * 64) 
        self.tm = transportMatrix(output_square_size=64, channel=3)

    def __call__(self, x, training="training"):

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1, training)
        elu1 = self.elu1(bn1)

        conv2 = self.conv2(elu1)
        bn2 = self.bn2(conv2, training)
        elu2 = self.elu2(bn2)

        conv3 = self.conv3(elu2)
        bn3 = self.bn3(conv3, training)
        elu3 = self.elu3(bn3)

        conv4 = self.conv4(elu3)
        bn4 = self.bn4(conv4, training)
        elu4 = self.elu4(bn4)

        fc1 = self.fc(elu4)
        bn5 = self.bn5(fc1, training)
        elu5 = self.elu5(bn5)
        latentVector = self.dropout(elu5, training)

        defc = self.defc(latentVector)
        defc = tf.add(defc, conv4)
        bn6 = self.bn6(defc, training)
        elu6 = self.elu6(bn6)

        deconv4 = self.deconv4(elu6)
        deconv4 = tf.add(deconv4, conv3)
        bn7 = self.bn7(deconv4, training)
        elu7 = self.elu7(bn7)

        deconv3 = self.deconv3(elu7)
        deconv3 = tf.add(deconv3, conv2)
        bn8 = self.bn8(deconv3, training)
        elu8 = self.elu8(bn8)

        deconv2 = self.deconv2(elu8)
        deconv2 = tf.add(deconv2, conv1)
        bn9 = self.bn9(deconv2, training)
        elu9 = self.elu9(bn9)

        deconv1 = self.deconv1(elu9)
        bn10 = self.bn10(deconv1, training)
        elu10 = self.elu10(bn10)

        out = self.out(elu10)
        out = self.tanh(out)

        return out

    def render_scene(self, x, training="training"):
        
        # HERE,,,, log decompression...?
        img = self.tm(x)

        return img

class skyencoder(Model):
    def __init__(self, fc_dim="fc_dim"):

        super(skyencoder, self).__init__()

        # encodes pseudoHDR to sky parameter in skynet manner
        self.conv1_skynet = ops.conv2d(output_channels=64, k_h=7, k_w=7, strides=2)
        self.norm1_skynet = tfa.layers.InstanceNormalization(axis=3,
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.actv1_skynet  = ops.elu()

        self.res1_skynet = resLayer((64,64), 64, k_h=3, k_w=3, strides=1)
        self.mp1_skynet  = ops.maxpool2d(kernel_size=3, strides=2)

        self.res2_skynet = resLayer((16,16), 32, k_h=3, k_w=3, strides=1)
        self.mp2_skynet  = ops.maxpool2d(kernel_size=3, strides=2)
        
        self.flat_skynet = tf.keras.layers.Flatten()
        self.fc_skynet = tf.keras.layers.Dense(fc_dim)

    def __call__(self, x, training="training"):

        conv1 = self.conv1_skynet(x)
        norm1 = self.norm1_skynet(conv1)
        actv1 = self.actv1_skynet(norm1)

        res1 = self.res1_skynet(actv1, training)
        mp1 = self.mp1_skynet(res1)

        res2 = self.res2_skynet(mp1, training)
        mp2 = self.mp2_skynet(res2)
        
        flat = self.flat_skynet(mp2)
        fc = self.fc_skynet(flat)

        return fc