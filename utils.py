import OpenEXR
import Imath
import numpy as np
import os
from datetime import datetime as dt
import cv2

from numpy.random import uniform

def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_var(timestamp = dt.now().strftime("%Y-%m-%d-%H:%M:%S"))
def getTimestamp():
    return getTimestamp.timestamp

# If there is not input name, create the directory name with timestamp
def createNewDir(root_path, name=None):

    if name == None:
        print("[utils.py, createNewDir()] DirName is not defined in the arguments, define as timestamp")
        newpath = os.path.join(root_path, getTimestamp())
    else:
        newpath = os.path.join(root_path, name)

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath

def createTrainValidationDirpath(root_dir, createDir = False):
    
    if createDir == True:
        train_dir = createNewDir(root_dir, "train")
        val_dir = createNewDir(root_dir, "val")
    
    else:
        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val") 

    return train_dir, val_dir

def writeHDR(arr, outfilename, imgshape):

    ext_name = outfilename.split(".")[1]
    if ext_name == "exr":
        '''Align_ratio (From HDRUNET)''' 
        # align_ratio = (2 ** 16 - 1) / arr.max()
        # arr = np.round(arr * align_ratio).astype(np.uint16)
        
        '''write HDR image using OpenEXR'''
        # Convert to strings
        R, G, B = [x.astype('float16').tostring() for x in [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]]

        im_height, im_width = imgshape

        HEADER = OpenEXR.Header(im_width, im_height)
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(outfilename, HEADER)
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()

    if ext_name == "hdr":
        bgr = arr[...,::-1].copy()
        cv2.imwrite(outfilename, bgr)

def openexr2np(path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    image = OpenEXR.InputFile(path)

    dw = image.header()['dataWindow']
    size = (dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1)
    (redstr, greenstr, bluestr) = image.channels("RGB",pt)
    
    red = np.frombuffer(redstr, dtype = np.float32)
    green = np.frombuffer(greenstr, dtype = np.float32)
    blue = np.frombuffer(bluestr, dtype = np.float32)

    for i in [red,green,blue]:
        i.shape=(size[1],size[0])

    red = np.expand_dims(red,axis=2)
    green = np.expand_dims(green, axis=2)
    blue = np.expand_dims(blue, axis=2)

    color = np.concatenate([red,green,blue],axis=2)

    return color

"""
Below codes are used for "Randomised Tone Mapping"

Tonemapping function is not working on the python script (numeric error?)

it seems that numerical error has occured while convert to python data
(it only happens with Laval Sky Database)
"""

def map_range(x, low=0, high=1):
    # Shrink the range of pixel values to [0..1] or something
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

class Exposure(object):
    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma

class PercentileExposure(object):
    def __init__(self, randomize=False):
        
        self.gamma = uniform(1.8, 2.2)
        self.low_perc = uniform(0, 15)
        self.high_perc = uniform(85, 100)
    
    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        img = map_range(np.clip(x, low, high)) ** (1 / self.gamma)
        img *= 255
        return img.astype(np.int8)

# class BaseTMO(object):
#     def __call__(self, img):
#         return self.op.process(img)

# class Reinhard(BaseTMO):
#     def __init__(
#         self,
#         intensity=-1.0,
#         light_adapt=0.8,
#         color_adapt=0.0,
#         gamma=2.0,
#         randomize=False,
#     ):
#         if randomize:
#             gamma = uniform(1.8, 2.2)
#             intensity = uniform(-1.0, 1.0)
#             light_adapt = uniform(0.8, 1.0)
#             color_adapt = uniform(0.0, 0.2)

#         self.op = cv2.createTonemapReinhard(
#             gamma=gamma,
#             intensity=intensity,
#             light_adapt=light_adapt,
#             color_adapt=color_adapt,
#         )

# class Mantiuk(BaseTMO):
#     def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
#         if randomize:
#             gamma = uniform(1.8, 2.2)
#             scale = uniform(0.65, 0.85)

#         self.op = cv2.createTonemapMantiuk(
#             saturation=saturation, scale=scale, gamma=gamma
#         )

# # ALM
# class Drago(BaseTMO):
#     def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
#         if randomize:
#             gamma = uniform(1.8, 2.2)
#             bias = uniform(0.7, 0.9)

#         self.op = cv2.createTonemapDrago(
#             saturation=saturation, bias=bias, gamma=gamma
#         )

# TMO_DICT = {
#     'exposure': Exposure,
#     'reinhard': Reinhard,
#     'mantiuk': Mantiuk,
#     'drago': Drago,
# }

# def tone_map(img, tmo_name, **kwargs):
#     return TMO_DICT[tmo_name](**kwargs)(img)

# TRAIN_TMO_DICT = {
#     'exposure': PercentileExposure,
#     'reinhard': Reinhard,
#     'mantiuk': Mantiuk,
#     'drago': Drago,
# }

# def random_tone_map(x):
#     tmos = list(TRAIN_TMO_DICT.keys())
#     choice = np.random.randint(0, len(tmos))

#     tmo = TRAIN_TMO_DICT[tmos[choice]](randomize=True)
    
#     ldr = tmo(x)
#     if(np.isnan(ldr.any())):
#         ldr = np.nan_to_num(ldr, nan=0.0)

#     return map_range(ldr, low=0, high=2) - 1.