import numpy as np
import cv2
import tensorflow as tf

def distortion_tf(h,w, dilation_rate=1., skydome=True):

    pi = np.math.pi
    
    unit_w = tf.divide(2*pi, w)
    unit_h = tf.divide(pi, h*2 if skydome else h)

    rho = tf.math.tan(unit_w) * dilation_rate

    v = tf.constant([0.,1.,0.])

    r_grid = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    
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

    return tf.convert_to_tensor(kernel)
    
class DistortionAwareConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                    filters,
                    distortion = "offset",
                    output_channels="output_channels",
                    strides="strides",
                    k_h="k_h",
                    k_w= "k_w", 
                    padding="SAME",
                    dilation_rate=1.,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    use_bias = True,
                    **kwargs):
        super(DistortionAwareConv2D, self).__init__()  # ==super().__init()

        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.

        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """

        assert(distortion is not None, "distortion-aware_filter must have an offset")

        self.filters = filters
        self.output_channels = output_channels
        self.k_w = k_w
        self.k_h = k_h
        self.strides = [1,strides, strides,1]
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.use_bias = use_bias
        
        self.distortion = distortion

    # TODO create the offset when build the layer?
    def build(self, input_shape):
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channels = input_shape[3]
        self.in_dim = in_height * in_width * in_channels
        
        input_dim = int(input_shape[-1])

        super(DistortionAwareConv2D, self).build(input_shape) 

        self.kernel = self.add_weight(
            name='kernel',
            shape=input_dim,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,), # TODO filter definition ?
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,)
        
    def call(self, input, training=None):

        # get x, y axis offset
        offset = self.distortion
        offset = tf.stack([offset] * _w)
        offset = tf.transpose(offset, [1, 0, 2, 3])
        offset = tf.expand_dims(offset, 0)

        input = self._pad_input(input)

        # some length
        # batch_size = tf.cast(input.get_shape()[0], dtype=tf.int32)
        # channel_in = tf.cast(input.get_shape()[-1], dtype=tf.int32)
        # in_h, in_w = [tf.cast(i, dtype=tf.int32) for i in input.get_shape()[1: 3]]  # input feature map size
        b, h, w, c = input.get_shape().as_list()[0: 4]

        # out_h, out_w = [tf.cast(i, dtype=tf.int32) for i in offset.get_shape()[1: 3]]  # output feature map size 
        out_h, out_w = h, w
        filter_h, filter_w = self.k_h, self.k_w

        # get x, y axis offset
        offset = tf.reshape(offset, [b, out_h, out_w, -1, 2])

        # TODO For each height  all offset is the same.
        x_off, y_off = offset[:, :, 0], offset[:, :, 1]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices([h, w])
        tf.print("y : ", y.get_shape())
        tf.print("y : ", x.get_shape())
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [b, 1, 1]) for i in [y, x]]
        y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.to_float(i) for i in [y, x]]

        # Add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, h - 1)

        # TODO change method to consider 360 degree
        x = tf.clip_by_value(x, 0, w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1

        # clip
        y0, y1 = [tf.clip_by_value(i, 0, h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DistortionAwareConv2D._get_pixel_values_at_point(input, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]

        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)

        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [b, out_h, out_w, filter_h, filter_w, 1, c])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [b, out_h * filter_h, out_w * filter_w, 1, c])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [b, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        
        # add the output feature maps in the same group
        out = tf.reshape(out, [b, out_h, out_w, self.filters, c])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)




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
        x, y = [tf.image.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
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


@tf.function
def test(offset, image):

    Ws= tf.constant(0.)
    bs = 2 * Ws
    cost = Ws + bs
    g = tf.gradients(cost, [Ws, bs])
    dCost_dW, dCost_db = g

    return dCost_dW, dCost_db