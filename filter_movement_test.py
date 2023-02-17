import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def _pad_input(inputs, kernel_size=3, strides=1, dilation_rate=1):
    """Check if input feature map needs padding, because we don't use the standard Conv() function.

    :param inputs:
    :return: padded input feature map
    """
    # When padding is 'same', we should pad the feature map.
    # if padding == 'same', output size should be `ceil(input / stride)`
    
    in_shape = inputs.get_shape().as_list()[1: 3]
    padding_list = []
    for i in range(2):
        dilated_filter_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
        same_output = (in_shape[i] + strides - 1) // strides
        valid_output = (in_shape[i] - dilated_filter_size + strides) // strides
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

def _get_conv_indices(feature_map_size, kernel_size=3, strides=1, dilation_rate=1):
    """the x, y coordinates in the window when a filter sliding on the feature map

    :param feature_map_size:
    :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
    """
    feat_h, feat_w = [tf.cast(i, dtype=tf.int32) for i in feature_map_size[0: 2]]

    x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
    x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
    x, y = [tf.image.extract_patches(i,
                                            [1, kernel_size, kernel_size, 1],
                                            [1, strides, strides, 1],
                                            [1, dilation_rate, dilation_rate, 1],
                                            'VALID')
            for i in [x, y]]   # shape [1, 1, feat_w - kernel_size + 1, feat_h * kernel_size]    [0 1 2 0 1 2 0 1 2]
    
    # y, x = tf.meshgrid(tf.range(feat_h), tf.range(feat_w))
    # y, x = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [y, x]]  # shape [1, h, w, 1]
    # y, x = [tf.image.extract_patches(i,
    #                                         [1, kernel_size, kernel_size, 1],
    #                                         [1, strides, strides, 1],
    #                                         [1, dilation_rate, dilation_rate, 1],
    #                                         'VALID')
    #         for i in [y, x]]   # shape [1, 1, feat_w - kernel_size + 1, feat_h * kernel_size]    [0 1 2 0 1 2 0 1 2]
    
    return y, x

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

def make_grid(kernel_size):
    # R_grid should be upside down because image pixel coordinate is orientied from top left
    assert kernel_size % 2 == 1, "kernel_size must be odd number, current kernel size : {}".format(kernel_size)
    grid = []
    r = kernel_size // 2
    
    for y in range(r, -r-1, -1):
        for x in range(r, -r-1, -1):
            grid.append([x,y])

    return grid


def distortion(h,w, kernel_size= 3, strides = 1, dilation_rate=1, skydome=True):

    pi = np.math.pi
    n = kernel_size // 2
    middle = n * (kernel_size + 1)
    
    unit_w = tf.divide(2 * pi, w)
    unit_h = tf.divide(pi, h * 2 if skydome else h)

    rho = tf.math.tan(unit_w) * dilation_rate

    v = tf.constant([0., 1., 0.])

    r_grid= make_grid(kernel_size=kernel_size)
    
    x = int(w * 0.5)
    
    kernel = list()

    for y in range(0,h):

        # radian
        theta = (x - 0.5 * w) * unit_w
        phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h

        x_u = tf.math.cos(phi)*tf.math.cos(theta)
        y_u = tf.math.sin(phi)
        z_u = tf.math.cos(phi)*tf.math.sin(theta)
        p_u = tf.constant([x_u.numpy(), y_u.numpy(), z_u.numpy()])

        t_x = tf.linalg.cross(v, p_u)
        t_y = tf.linalg.cross(p_u,t_x)
        
        r_sphere = list()
        for r in r_grid:
            r_sphere.append(tf.multiply(rho, tf.add(r[0] * t_x, r[1] * t_y)))
        r_sphere = tf.squeeze(r_sphere)
        p_ur = tf.add(p_u, r_sphere)
        
        k = list()
        for ur_i in p_ur:
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
                    raise Exception("undefined coordinates")
                    
            phi_r = tf.math.asin(ur_i[1])

            x_r = (tf.divide(theta_r, pi) + 1)*0.5*w
            y_r = (1. - tf.divide(2*phi_r, pi))*h if skydome else (0.5 - tf.divide(phi_r, pi))*h

            k.append([y_r, x_r])

        offset = tf.subtract(k, k[middle])
        kernel.append(offset)

    kernel = tf.convert_to_tensor(kernel)
    
    kernel = tf.stack([kernel] * w)
    kernel = tf.transpose(kernel, [1, 0, 2, 3])   # 32, 128, 9, 2 ( h, w, grid #, (y,x) )
    kernel = tf.expand_dims(kernel, 0)  # 1, 32, 128, 9, 2 ( b, h, w, grid #, (y,x) )
    
    # y, x = kernel[:,:,0], kernel[:,:,1]
    # y, x = [tf.stack([i] * w) for i in [y,x]]
    # y, x = [tf.transpose(i, [1, 0, 2]) for i in [y,x]] # 32, 128, 9 ( h, w, grid # )
    # y, x = [tf.expand_dims(i, 0) for i in [y,x]] # 1, 32, 128, 9, 2 ( b, h, w, grid #, (y,x) )
    
    return kernel

def run(src, kernel_size="kernel_size", strides="strides", dilation_rate="dilation_rate", skydome = False):
    
    batch_size, h, w, _ = src.get_shape()

    filter_h, filter_w = kernel_size, kernel_size # kernel_size
    
    # y_off, x_off = distortion(h, w, kernel_size= kernel_size, strides = strides, dilation_rate=dilation_rate, skydome=skydome)
    offset = distortion(h, w, kernel_size= kernel_size, strides = strides, dilation_rate=dilation_rate, skydome=skydome)
    
    # TODO DEBUG
    # debug(src[0].numpy(), offset[0].numpy(), gridoffset=True)
    # exit(0)
    
    # add padding if needed
    # dilation_rate must be 1
    img = _pad_input(src, kernel_size=kernel_size, strides=strides, dilation_rate=1)       
    
    # some length
    in_h, in_w, channel_in = img.get_shape()[1:4] # input feature map size
    out_h, out_w = offset.get_shape()[1: 3]  # output feature map size
    
    # get x, y axis offset
    y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
    
    # input feature map gird coordinates
    # dilation_rate must be 1
    y, x = _get_conv_indices([in_h, in_w], kernel_size=kernel_size, strides=strides, dilation_rate=1)
    y, x = [tf.cast(i, dtype=tf.float32) for i in [y, x]]
    
    # TODO DEBUG
    # res = tf.stack((y,x), axis=-1)
    # debug(img[0].numpy(), res[0].numpy())
    # exit(0)
    
    # x
    # TensorShape([1, 416, 1664, 9])
    # x_off
    # TensorShape([1, 416, 1664, 9])

    # add offset
    y = tf.add_n([y, y_off])
    x = tf.add_n([x, x_off])
    y = tf.clip_by_value(y, 0, in_h - 1)
    
    # consider 360 degree
    x= tf.where( x < 0 , tf.add(x, in_w), x)
    x= tf.where( x > in_w - 1 , tf.subtract(x, in_w), x)
    
    # a pixel in the output feature map has several same offsets 
    y, x = [tf.tile(i, [batch_size, 1, 1, 1]) for i in [y, x]] 

    # TODO DEBUG
    res = tf.stack((y,x), axis=-1)
    return res , img

    ######### below script code does not need in this test section. 

    # get four coordinates of points around (x, y)
    y0, x0 = [tf.cast(tf.floor(i), dtype=tf.int32) for i in [y, x]]
    y1, x1 = y0 + 1, x0 + 1

    # clip
    y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]] 
    # x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]
    # consider 360 degree
    x0, x1 = [tf.where( i < 0 , tf.add(i, in_w), i) for i in [x0, x1]]
    x0, x1 = [tf.where( x > in_w - 1 , tf.subtract(x, in_w), i) for i in [x0, x1]]
    
    # get pixel values
    indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]

    p0, p1, p2, p3 = [tf.cast(_get_pixel_values_at_point(img, i)
                        , dtype=tf.float32) for i in indices]

    # cast to float
    x0, x1, y0, y1 = [tf.cast(i, dtype=tf.float32) for i in [x0, x1, y0, y1]]
    
    # weights
    w0 = tf.multiply(tf.subtract(y1, y), tf.subtract(x1, x))
    w1 = tf.multiply(tf.subtract(y1, y), tf.subtract(x, x0))
    w2 = tf.multiply(tf.subtract(y, y0), tf.subtract(x1, x))
    w3 = tf.multiply(tf.subtract(y, y0), tf.subtract(x, x0))

    # expand dim for broadcast
    w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]

    # bilinear interpolation
    pixels = tf.add_n([tf.multiply(w0, p0), tf.multiply(w1, p1), 
                    tf.multiply(w2, p2), tf.multiply(w3, p3)])
    # current shape at this line : [b, h, w, grid size(3x3=9)*out_channels, in_channels(3)]

    # reshape the "big" feature map
    pixels = tf.reshape(pixels, [batch_size, out_h* out_w, filter_h* filter_w* channel_in])

    # return pixels, img

def debug(newimg, offset, gridoffset = False):
    
    h, w, _, _ = offset.shape
    
    if gridoffset:
        for i in range(0, h, 30):
            for j in range(0, w, 3):
                tmp = newimg.copy() 
                grid = offset[i,j,:,:]
                for g in grid:
                    y,x = g
                    tmp = cv2.circle(tmp, (int(x+j),int(y+i)), 2, (0,0,255), 2)
                
                cv2.imshow("tmp", tmp)
                cv2.waitKey(1)
    
    else:
        for i in range(h-31, h, 10):
            for j in range(0, w, 3):
                tmp = newimg.copy() 
                grid = offset[i,j,:,:]
                for g in grid:
                    y,x = g
                    tmp = cv2.circle(tmp, (int(x), int(y)), 2, (0,0,255), 2)
                
                cv2.imshow("tmp", tmp)
                cv2.waitKey(1)
            
if __name__=="__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="test a distortion aware conv filter")
    parser.add_argument('--img', type=str, default="skydome.jpg")
    parser.add_argument('--skydome', type=str, default="True")

    args = parser.parse_args()
    
    image = cv2.imread(args.img)
    
    h,w,_ = image.shape
    
    kernel_size=3
    strides=1 
    dilation_rate=50
    # newsize = (int(w/4), int(h/4))
    # image = cv2.resize(image, dsize=newsize)
    
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    pixels, newimg = run(image, kernel_size, strides, dilation_rate, skydome = str2bool(args.skydome))
    
    debug(newimg[0].numpy(), pixels[0].numpy())