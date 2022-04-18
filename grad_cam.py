from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return  tf.divide(x, (K.sqrt(K.mean(K.square(x))) + 1e-5))

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

"""
relu modification of the model
"""
# def modify_backprop(model, name):
#     g = tf.get_default_graph()
#     with g.gradient_override_map({'Relu': name}):

#         # get layers that have an activation
#         layer_dict = [layer for layer in model.layers[1:]
#                       if hasattr(layer, 'activation')]

#         # replace relu activation
#         for layer in layer_dict:
#             if layer.activation == keras.activations.relu:
#                 layer.activation = tf.nn.relu

#         # re-instanciate a new model
#         new_model = VGG16(weights='imagenet')
#     return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(y_c, A_k, image, grads):

    # nb_classes = 1000
    # target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    # model.add(Lambda(target_layer,
    #                  output_shape = target_category_loss_output_shape))

    # y_c : class_index에 해당하는 CNN 마지막 layer op(softmax, linear, ...)의 입력
    
    # TODO Insert 64 params?
    # y_c = K.sum(model.layers[-1].output)

    # A_k: activation conv layer의 출력 feature map
    # TODO conv? max pool?
    # A_k =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    
    # TODO norm 해줘도 되고 안해줘도 되는듯?

    # tf.print("\n#########grads\n",grads, summarize = -1)
    # tf.print("\n#########grads\n",tf.shape(grads))

    # grads = normalize(K.gradients(y_c, A_k)[0])
    
    # grads = tf.gradients(y_c, A_k)[0]
    
    # gradient_function = K.function([image], [A_k, grads])
    
    # batch size가 포함되어 shape가 (1, width, height, k)이므로
    # (width, height, k)로 shape 변경
    # 여기서 width, height는 activation conv layer인 A_k feature map의 width와 height를 의미함
    # output, grads_val = gradient_function([image])
    
    # TODO 이게 맞아? ** 0번째 배치 지우기 **
    # output, grads_val = output[:, :], grads_val[:, :, :, :]

    
    # tf.print("\n###############\ngrads\n###############\n",tf.shape(grads))
    output, grads_val = A_k.numpy(), grads

    # global average pooling 연산
    # gradient의 width, height에 대해 평균을 구해서(1/Z) weights(a^c_k) 계산
    weights = np.mean(grads_val, axis = (1, 2))

    # activation conv layer의 출력 feature map(conv_output)과
    # class_index에 해당하는 weights(a^c_k)를 k에 대응해서 weighted combination 계산
    
    # feature map(conv_output)의 (width, height)로 초기화
    # cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    param_size = np.shape(grads_val)[0]
    cam_shape = [param_size, output.shape[0], output.shape[1]]
    cam = np.zeros(shape=cam_shape, dtype = np.float32)

    # print("\n###############\noutput\n###############\n",np.shape(output))
    # Linear Combination
    # for each batch, it has a weight value of 16

    batch_size = output.shape[0]
    new_cam = []

    for j, batch in enumerate(weights): # Batch is the index # of sky paramters
        # print("\n###############\nbatch\n###############\n",np.shape(batch))
        for i, w in enumerate(batch):
            cam[j, :, :] += w * output[:, :, i]

        new_cam.append(cv2.resize(cam[j], (128, 32)) )

    # 계산된 weighted combination 에 ReLU 적용
    new_cam = np.maximum(new_cam, 0)
    new_cam = np.expand_dims(new_cam, axis=-1)
    heatmap = new_cam / (np.max(new_cam) + 1e-10)
    # print("\n###############\ncam\n###############\n",np.shape(cam))
    # print("\n###############\nnew_cam\n###############\n",np.shape(new_cam))
    #Return to BGR [0..255] from the preprocessed image

    # for i in range(len(image)):
    #     image[i] -= np.min(image[i])
    #     image[i] = np.minimum(image[i], 255)

    # new_cam = new_cam.astype(np.uint8)
    image = [image] * param_size
    
    heatmap = 255*heatmap
    heatmap = heatmap.astype(np.uint8)
    res = []

    # import matplotlib.pyplot as plt
    # plt.figure()
    for i in range(len(heatmap)):
        ans  = cv2.applyColorMap(heatmap[i], cv2.COLORMAP_JET)
        ans = ans[:,:,::-1]
        # plt.subplot(1, 2, 1)
        # plt.imshow(image[i], interpolation='nearest')
        # plt.subplot(1, 2, 2)
        # plt.imshow(ans, interpolation='nearest')
        # plt.show()
        ans = ans.astype(np.float32) + image[i].astype(np.float32)
        res.append(255 * ans / np.max(ans))
    # return np.uint8(cam), heatmap
    return np.uint8(res)

# cam, heatmap = grad_cam(output, grads_val, img)
# cv2.imwrite("gradcam.jpg", cam)

# register_gradient()
# saliency_fn = compile_saliency_function(guided_model)
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))