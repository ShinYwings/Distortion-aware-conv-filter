import numpy as np
import cv2
# import distortion_filter as df
import tensorflow as tf
import sys

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def distortion(h,w, dilation_rate=1, skydome=True):

    pi = np.math.pi
    
    unit_w = 2*pi/w
    unit_h = pi/h*0.5 if skydome else pi/h 

    rho = np.math.tan(unit_w) * dilation_rate

    v = np.matrix([0,1,0])

    # r_grid = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    
    # r_grid = np.array([[1,-1],[1,0],[1,1],[0,-1],[0,0],[0,1],[-1,-1],[-1,0],[-1,1]])
    r_grid = np.array([[1,1],[0,-1],[-1,1],
                        [1,0],[0,0],[-1,0],
                        [1,-1],[0,1],[-1,-1]])
    x = int(w*0.5)

    kernel = []

    for y in range(0,h):

        # radian
        theta = (x - 0.5*w) * unit_w
        phi   = (h - y) * unit_h if skydome else (h*0.5 - y) * unit_h

        x_u = np.math.cos(phi)*np.math.cos(theta)
        y_u = np.math.sin(phi)
        z_u = np.math.cos(phi)*np.math.sin(theta)
        p_u = np.array([x_u, y_u, z_u])

        t_x = np.cross(v, p_u)
        t_y = np.cross(p_u,t_x)
        
        r_sphere = list()
        for r in r_grid:
            r_sphere.append(rho * (r[0]*t_x + r[1]*t_y))

        r_sphere = np.squeeze(r_sphere)
        p_ur = p_u + r_sphere
        
        k = []
        for ur_i in p_ur:
            if ur_i[0] > 0:
                theta_r = np.math.atan2(ur_i[2], ur_i[0])
            elif ur_i[0] < 0:
                if ur_i[2] >=0:
                    theta_r = np.math.atan2(ur_i[2], ur_i[0]) + pi
                else:
                    theta_r = np.math.atan2(ur_i[2], ur_i[0]) - pi
            else:
                if ur_i[2] > 0:
                    theta_r = pi*0.5
                elif ur_i[2] < 0:
                    theta_r = -pi*0.5
                else:
                    print("undefined coordinates")
                    exit(0)
                    
            phi_r = np.math.asin(ur_i[1])

            x_r = (np.divide(theta_r, pi) + 1)*0.5*w
            y_r = (1. - np.divide(2*phi_r, pi))*h if skydome else (0.5 - np.divide(phi_r, pi))*h

            k.append([x_r, y_r])

        offset = np.subtract(k, k[4])
        kernel.append(offset)

    return kernel

def _pad_input(inputs, filter_size=3, dilation_rate=1):
    """Check if input feature map needs padding, because we don't use the standard Conv() function.

    :param inputs:
    :return: padded input feature map
    """
    # When padding is 'same', we should pad the feature map.
    # if padding == 'same', output size should be `ceil(input / stride)`
    strides = [1,1]
    in_shape = inputs.get_shape().as_list()[1: 3]
    # in_shape = inputs.shape[1: 3]
    padding_list = []
    for i in range(2):
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation_rate - 1)
        same_output = (in_shape[i] + strides[i] - 1) // strides[i]
        valid_output = (in_shape[i] - dilated_filter_size + strides[i]) // strides[i]
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

def _get_conv_indices(feature_map_size, filter_size):
    """the x, y coordinates in the window when a filter sliding on the feature map

    :param feature_map_size:
    :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
    """
    feat_h, feat_w = [tf.cast(i, dtype=tf.int32) for i in feature_map_size[0: 2]]

    x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
    x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
    x, y = [tf.image.extract_patches(i,
                                        [1, filter_size, filter_size, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        'VALID')
            for i in [x, y]]   # shape [1, 1, feat_w - stride + 1, feat_h * stride]
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

def test(img):

    _h,_w,_ = img.shape

    offset = distortion(_h,_w,1,True)

    tf.print(np.shape(img))

    input = _pad_input(tf.convert_to_tensor([img]))

    
    filters = 96
    num_deformable_group = 1

    # some length
    # batch_size = tf.cast(input.get_shape()[0], dtype=tf.int32)
    # channel_in = tf.cast(input.get_shape()[-1], dtype=tf.int32)
    # in_h, in_w = [tf.cast(i, dtype=tf.int32) for i in input.get_shape()[1: 3]]  # input feature map size
    b, h, w, c = input.get_shape().as_list()[0: 4]

    # out_h, out_w = [tf.cast(i, dtype=tf.int32) for i in offset.get_shape()[1: 3]]  # output feature map size 
    out_h, out_w = _h, _w
    filter_h, filter_w = 3, 3

    # get x, y axis offset
    offset = tf.stack([offset] * _w)
    offset = tf.transpose(offset, [1, 0, 2, 3])
    offset = tf.expand_dims(offset, 0)

    tf.print(offset[0,0,0], summarize=-1)
    tf.print(offset.get_shape())

    x_off, y_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
    
    # input feature map gird coordinates
    y, x = _get_conv_indices([h, w], 3)
    tf.print(x_off[0,1,1], y_off[0,1,1], summarize=-1)
    tf.print("XX", x_off.get_shape())
    tf.print(x[0,1,1], y[0,1,1], summarize=-1)
    tf.print("XX", x.get_shape())
    # y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
    # y, x = [tf.tile(i, [b, 1, 1, 1, 1]) for i in [y, x]]
    # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
    y, x = [tf.cast(i, dtype=tf.float64) for i in [y, x]]

    # Add offset
    y = tf.add_n([y, y_off])
    x = tf.add_n([x, x_off])

    y = tf.clip_by_value(y, 0, h - 1)
    
    # consider 360 degree
    x= tf.where( x < 0 , tf.add(x, w), x)
    x= tf.where( x > w - 1 , tf.subtract(x, w), x)
    
    # y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
    # y, x = [tf.tile(i, [b, 1, 1, filters, 1]) for i in [y, x]] # a pixel in the output feature map has several same offsets 
    # y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
    # tf.print("y,x: ", y.get_shape(), "\n", x.get_shape())

    # get four coordinates of points around (x, y)
    y0, x0 = [tf.cast(tf.floor(i), dtype=tf.int32) for i in [y, x]]
    y1, x1 = y0 + 1, x0 + 1

    # clip
    y0, y1 = [tf.clip_by_value(i, 0, h - 1) for i in [y0, y1]]
    x0, x1 = [tf.clip_by_value(i, 0, w - 1) for i in [x0, x1]]

    # get pixel values
    indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
    p0, p1, p2, p3 = [tf.cast(_get_pixel_values_at_point(input, i), dtype=tf.float64) for i in indices]

    # cast to float
    x0, x1, y0, y1 = [tf.cast(i, dtype=tf.float64) for i in [x0, x1, y0, y1]]

    # weights
    w0 = tf.multiply(tf.subtract(y1, y), tf.subtract(x1, x))
    w1 = tf.multiply(tf.subtract(y1, y), tf.subtract(x, x0))
    w2 = tf.multiply(tf.subtract(y, y0), tf.subtract(x1, x))
    w3 = tf.multiply(tf.subtract(y, y0), tf.subtract(x, x0))

    # expand dim for broadcast
    w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
    
    # bilinear interpolation (verified by shin)
    pixels = tf.add_n([tf.multiply(w0, p0), tf.multiply(w1, p1), 
                        tf.multiply(w2, p2), tf.multiply(w3, p3)])

    tf.print("pixels : ", pixels.get_shape() , output_stream=sys.stdout)

    # reshape the "big" feature map
    pixels = tf.reshape(pixels, [b, out_h* out_w, filter_h* filter_w* c])
    
    # pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
    # pixels = tf.reshape(pixels, [b, out_h * filter_h, out_w * filter_w, c])

    # copy channels to same group
    # pixels = tf.tile(pixels, [1, 1, 1, 1, filters])
    # pixels = tf.reshape(pixels, [b, out_h * filter_h, out_w * filter_w, -1])

    print("pixels shape : ", pixels.get_shape())

    return pixels
    # # depth-wise conv
    # out = tf.nn.conv2d(pixels, kernel, [1, filter_h, filter_w, 1], 'VALID')
    
    # # add the output feature maps in the same group
    # out = tf.reshape(out, [b, out_h, out_w, filters, c])
    # out = tf.reduce_sum(out, axis=-1)
    # if self.use_bias:
    #     out += self.bias

def debug(newimg ,kernel_offset):
    w= 1664
    for i in kernel_offset[0]:
        pos = [w/2, 0] + i
        pos = list(map(int, pos))
        newimg = cv2.circle(newimg, pos, 5, (0,0,255), 2)
    cv2.imshow("newimg", newimg)
    cv2.waitKey(0)

    for i in kernel_offset[30]:
        pos = [w/2, 30] + i
        pos = list(map(int, pos))
        newimg = cv2.circle(newimg, pos, 5, (0,0,255), 2)
    cv2.imshow("newimg", newimg)
    cv2.waitKey(0)

    for i in kernel_offset[200]:
        pos = [100, 200] + i
        pos = list(map(int, pos))
        newimg = cv2.circle(newimg, pos, 5, (0,0,255), 2)
    cv2.imshow("newimg", newimg)
    cv2.waitKey(0)

    for i in kernel_offset[400]:
        pos = [w/2, 400] + i
        pos = list(map(int, pos))
        newimg = cv2.circle(newimg, pos, 5, (0,0,255), 2)
    cv2.imshow("newimg", newimg)
    cv2.waitKey(0)

   
if __name__ == "__main__":

    img = cv2.imread("output.jpg", -1)
    img = cv2.resize(img, (416, 104))

    # h, w, c = img.shape

    newimg = img[...]

    a = test(img)

    # kernel_offset = distortion(h,w, dilation_rate=100, skydome=True)
    
    # print("kernel_offset: ",np.shape(kernel_offset))

    # print("kernel_offset: ",np.shape(np.reshape(kernel_offset, [416, 1, 1, -1, 2])))
    
    DEBUG = False