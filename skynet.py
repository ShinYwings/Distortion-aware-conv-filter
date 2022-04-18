import tensorflow as tf
from tensorflow.keras import Model
import ops

import tensorflow_addons as tfa

import filter as df

# class transportMatrix(tf.keras.layers.Layer):
#     def __init__(self, output_square_size=64, channel=3):
#         super(transportMatrix, self).__init__()
#         initializer = tf.random_normal_initializer()

#         # 4096 = image h, w (32 x 128)
#         self.oss = output_square_size
#         self.oss_pow = tf.cast(output_square_size * output_square_size, tf.int32)
#         self.w = tf.Variable(initial_value=initializer(shape=[self.oss_pow, self.oss_pow, channel], dtype=tf.float32), trainable=True)
    
#     def call(self, x):
#         input_b, _, _, input_c = x.get_shape().as_list() #bhwc
#         x = tf.reshape(x, [input_b, self.oss_pow, 1, input_c])
#         tm = tf.einsum("jkl,ikml->ijml",self.w, x)
        
#         output = tf.reshape(tm, [input_b, self.oss, self.oss, input_c])

#         return output

class resBlock(Model):

    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, strides=1, dilation_rate=1):
        super(resBlock, self).__init__()
        # self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv1 = df.DistortionConvLayer(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)  # out 24
        self.norm1 = tfa.layers.InstanceNormalization(axis=3,
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.elu1 = ops.elu()
        
        # self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv2 = df.DistortionConvLayer(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)  # out 24
        
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

    def __init__(self, filters, filter_in, k_h, k_w, strides=1, dilation_rate=2):
        super(resLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in]+ list(filters), filters):
            self.sequence.append(resBlock(f_in, f_out, k_h=k_h, k_w=k_w, strides=strides, dilation_rate=dilation_rate))
    
    def call(self, x, training=False):
        for unit in self.sequence:
            x=unit(x, training=training)
        return x

class model(Model):
    def __init__(self, fc_dim=64, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=2):
        super(model, self).__init__()

        """skynet + fully conv layers"""
        self.conv1 = ops.conv2d(output_channels=64, k_h=7, k_w=7, strides=2)
        """
        Currently, only stride == 1 is supported.
        mission : make distortion offset hopping
        """
        # self.conv1 = df.DistortionConvLayer(64, kernel_size=7, strides=2, dilation_rate=5)  # out 24
        self.norm1 = tfa.layers.InstanceNormalization(axis=3,
                                   center=True,
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.actv1  = ops.elu()

        self.res1 = resLayer((32,32), 64, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)
        self.mp1  = ops.maxpool2d(kernel_size=da_kernel_size, strides=2)

        self.res2 = resLayer((16,16), 32, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)
        # self.mp2  = ops.maxpool2d(kernel_size=3, strides=2)
        
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(fc_dim)

        self.defc = ops.dfc2d(out_height=int(im_height/ 4),
                                out_width=int(im_width/ 4),
                                    out_channels=16)

        self.us1 = ops.deconv2d(output_channels=32, output_imshape=[int(im_height/ 2), int(im_width/ 2)], k_h=3, k_w=3, method='resize')
        
        self.norm2 = tfa.layers.InstanceNormalization(axis=3,
                                center=True, 
                                scale=True,
                                beta_initializer="random_uniform",
                                gamma_initializer="random_uniform")
        self.actv2 = ops.elu()

        self.res3 = resLayer((32,32,32),32, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)
        
        self.us2 = ops.deconv2d(output_channels=64, output_imshape=[int(im_height), int(im_width)], k_h=3, k_w=3,  method='resize')
        self.norm3 = tfa.layers.InstanceNormalization(axis=3,
                                center=True, 
                                scale=True,
                                beta_initializer="random_uniform",
                                gamma_initializer="random_uniform")
        self.actv3 = ops.elu()

        self.res4 = resLayer((64,64), 64, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)

        # self.us3 = ops.deconv2d(output_channels=64, output_imshape=[im_height, im_width], k_h=3, k_w=3, method='resize')
        # self.norm4 = tfa.layers.InstanceNormalization(axis=3,
        #                         center=True, 
        #                         scale=True,
        #                         beta_initializer="random_uniform",
        #                         gamma_initializer="random_uniform")
        # self.actv4 = ops.elu()

        # self.res5 = resLayer((64,64), 64, h=32 , w=128 , k_h=7, k_w=7, strides=1)

        # self.conv2 = ops.conv2d(output_channels=3, k_h=3, k_w=3, strides=1)
        self.conv2 = df.DistortionConvLayer(3, kernel_size=da_kernel_size, strides=1, dilation_rate=dilation_rate)  # out 24
        
        self.tanh = ops.tanh()

        # 64 =  sqrt(128 * 64) 
        # self.tm = transportMatrix(output_square_size=64, channel=3)

    # def encode(self, x, training="training"):
    #     conv1 = self.conv1(x)
    #     norm1 = self.norm1(conv1)
    #     actv1 = self.actv1(norm1)

    #     res1 = self.res1(actv1, training)
    #     mp1 = self.mp1(res1)
        
    #     res2 = self.res2(mp1, training)
    #     mp2 = self.mp2(res2)
        
    #     flat = self.flat(mp2)
    #     fc = self.fc(flat)

    #     return fc

    # def decode(self, fc, training="training"):
    #     defc = self.defc(fc)

    #     res3 = self.res3(defc, training)

    #     us1 = self.us1(res3)
    #     norm2 = self.norm2(us1)
    #     actv2 = self.actv2(norm2)
    #     res4 = self.res4(actv2, training)

    #     us2 = self.us2(res4)
    #     norm3 = self.norm3(us2)
    #     actv3 = self.actv2(norm3)
    #     res5 = self.res5(actv3, training)

    #     conv2 = self.conv2(res5)
    #     output = self.tanh(conv2)
        
    #     return output
    
    def __call__(self, x, training="training"):
        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        actv1 = self.actv1(norm1)

        res1 = self.res1(actv1, training)
        mp1 = self.mp1(res1)
        
        res2 = self.res2(mp1, training)
        # mp2 = self.mp2(res2)
        
        flat = self.flat(res2)
        fc = self.fc(flat)

        defc = self.defc(fc)

        us1 = self.us1(defc)
        norm2 = self.norm2(us1)
        actv2 = self.actv2(norm2)

        res3 = self.res3(actv2, training)

        us2 = self.us2(res3)
        norm3 = self.norm3(us2)
        actv3 = self.actv3(norm3)

        res4 = self.res4(actv3, training)

        # us3 = self.us3(res4)
        # norm4 = self.norm4(us3)
        # actv4 = self.actv4(norm4)

        # res5 = self.res5(actv4, training)

        conv2 = self.conv2(res4)
        output = self.tanh(conv2)
        
        return output , res2, fc

    # def render_scene(self, x, training="training"):
        
    #     # HERE,,,, log decompression...?
    #     img = self.tm(x)

    #     return img