import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class DistortionConvLayer(Layer):
    """Only support "channel last" data format"""
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.

        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super(DistortionConvLayer, self).__init__()
        self.kernel = None
        self.bias = None
        self.filters = filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        _, h, w, input_dim = input_shape
        k_h, k_w = self.kernel_size
        kernel_shape = [int(k_h*k_w*input_dim), self.filters]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
       
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            trainable=True,
            dtype=self.dtype)
        
        self.offset = DistortionConvLayer.distortion(h, w, dilation_rate=self.dilation_rate[0], skydome=True)

        super(DistortionConvLayer, self).build(input_shape=input_shape)

    def call(self, inputs, training=None, **kwargs):
        
        # add padding if needed
        inputs = self._pad_input(inputs)       
        offset = self.offset

        # some length
        batch_size = inputs.get_shape()[0]
        channel_in = inputs.get_shape()[-1]
        in_h, in_w = [i for i in inputs.get_shape()[1: 3]]  # input feature map size
        out_h, out_w = [i for i in offset.get_shape()[1: 3]]  # output feature map size
        # out_h, out_w = in_h, in_w  # output feature map size
        
        filter_h, filter_w = self.kernel_size
        
        # get x, y axis offset
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        # y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        # y, x = [tf.tile(i, [batch_size, 1, 1, self.num_deformable_group, 1]) for i in [y, x]]
        # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.cast(i, dtype=tf.float32) for i in [y, x]]

        # x
        # TensorShape([1, 416, 1664, 9])
        # x_off
        # TensorShape([1, 416, 1664, 9])

        # add offset
        # y, x = y + y_off, x + x_off
        y = tf.add_n([y, y_off])
        x = tf.add_n([x, x_off])
        y = tf.clip_by_value(y, 0, in_h - 1)
        
        # consider 360 degree
        x= tf.where( x < 0 , tf.add(x, in_w), x)
        x= tf.where( x > in_w - 1 , tf.subtract(x, in_w), x)

        
        # y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1]) for i in [y, x]] # a pixel in the output feature map has several same offsets 
        
        # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), dtype=tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1

        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]

        p0, p1, p2, p3 = [tf.cast(DistortionConvLayer._get_pixel_values_at_point(inputs, i)
                            , dtype=tf.float32) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, dtype=tf.float32) for i in [x0, x1, y0, y1]]
        
        # weights
        # w0 = (y1 - y) * (x1 - x)
        # w1 = (y1 - y) * (x - x0)
        # w2 = (y - y0) * (x1 - x)
        # w3 = (y - y0) * (x - x0)
        w0 = tf.multiply(tf.subtract(y1, y), tf.subtract(x1, x))
        w1 = tf.multiply(tf.subtract(y1, y), tf.subtract(x, x0))
        w2 = tf.multiply(tf.subtract(y, y0), tf.subtract(x1, x))
        w3 = tf.multiply(tf.subtract(y, y0), tf.subtract(x, x0))

        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]

        # bilinear interpolation
        # pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])
        pixels = tf.add_n([tf.multiply(w0, p0), tf.multiply(w1, p1), 
                        tf.multiply(w2, p2), tf.multiply(w3, p3)])
        # current shape at this line : [b, h, w, grid size(3x3=9)*out_channels, in_channels(3)]


        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h* out_w, filter_h* filter_w* channel_in])

        # 1, 32, 128, 9, 2 ( b, h, w, grid # * out_channels, in_channels
        #                     =>   b, h, w, 3, 3, out_channels, in_channels)
        # pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        # pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        # pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group*channel_in])
        
        
        # current shape at this line : [b, 3h, 3w, out_channels, in_channels]
        
        # copy channels to same group
        
        """
         return same value when # filters is equal to num_deformable_group 
        """
        # feat_in_group = self.filters // self.num_deformable_group
        # pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        # pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])
        ## current shape at this line : [b, h*h_filter, w*w_filter, out_channels * in_channels]

        # current pixels shape at this line : [b, 3h, 3w, out_channels, in_channels]
        # self.kernel = [h, w, out_channels, in_channels]
        out = tf.matmul(pixels, self.kernel)
        
        # add the output feature maps in the same group
        # out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        # out = tf.reduce_sum(out, axis=-1)
        
        out = tf.nn.bias_add(out, self.bias)

        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters])
        
        return tf.nn.relu(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.

        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [tf.cast(i, dtype=tf.int32) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'VALID')
                for i in [x, y]]   # shape [1, 1, feat_w - kernel_size + 1, feat_h * kernel_size]    [0 1 2 0 1 2 0 1 2]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values

        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))

        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    @staticmethod
    def distortion(h,w, dilation_rate=1., skydome=True):

        pi = np.math.pi
        
        unit_w = tf.divide(2*pi, w)
        unit_h = tf.divide(pi, h*2 if skydome else h)

        rho = tf.math.tan(unit_w) * dilation_rate

        v = tf.constant([0.,1.,0.])

        # R_grid should be upside down because image pixel coordinate is orientied from top left
        # r_grid = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        r_grid = [[1,-1],[1,0],[1,1],[0,-1],[0,0],[0,1],[-1,-1],[-1,0],[-1,1]]
        
        x = int(w*0.5)
        
        kernel = list()

        for y in range(0,h):

            # radian
            theta = (x - 0.5*w) * unit_w
            phi   = (h - y) * unit_h if skydome else (h*0.5 - y) * unit_h

            x_u = tf.math.cos(phi)*tf.math.cos(theta)
            y_u = tf.math.sin(phi)
            z_u = tf.math.cos(phi)*tf.math.sin(theta)
            p_u = tf.constant([x_u.numpy(), y_u.numpy(), z_u.numpy()])

            t_x = tf.linalg.cross(v, p_u)
            t_y = tf.linalg.cross(p_u,t_x)
            
            r_sphere = list()
            for r in r_grid:
                r_sphere.append(tf.multiply(rho, tf.add(r[0]*t_x, r[1]*t_y)))
            r_sphere = tf.squeeze(r_sphere)
            p_ur = tf.add(p_u, r_sphere)
            
            k = list()
            for ur_i in p_ur:
                # ur_i = ur_ii.numpy()
                if ur_i[0] > 0:
                    theta_r = tf.math.atan2(ur_i[2], ur_i[0])
                elif ur_i[0] < 0:
                    if ur_i[2] >=0:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) + pi
                    else:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) - pi
                else:
                    if ur_i[2] > 0:
                        theta_r = pi*0.5
                    elif ur_i[2] < 0:
                        theta_r = -pi*0.5
                    else:
                        print("undefined coordinates")
                        exit(0)
                        
                phi_r = tf.math.asin(ur_i[1])

                x_r = (tf.divide(theta_r, pi) + 1)*0.5*w
                y_r = (1. - tf.divide(2*phi_r, pi))*h if skydome else (0.5 - tf.divide(phi_r, pi))*h

                k.append([x_r, y_r])

            offset = tf.subtract(k, k[4])
            kernel.append(offset)

        kernel = tf.convert_to_tensor(kernel)
        kernel = tf.stack([kernel] * w)
        kernel = tf.transpose(kernel, [1, 0, 2, 3])   # 32, 128, 9, 2 ( h, w, grid #, (x,y) )
        kernel = tf.expand_dims(kernel, 0)  # 1, 32, 128, 9, 2 ( b, h, w, grid #, (x,y) )
        
        return kernel