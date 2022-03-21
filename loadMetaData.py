import numpy as np
import tensorflow as tf 
import pickle as pk
import os
import sys

def load_ILSVRC2012_metadata():

    file_name = r"imgnet_meta.txt"
    dir_list, index_list, name_list = list(),list(),list()
    with open(file_name, "r") as f:

        raw_data = f.readlines()

    for i in raw_data:

        # dir, index, name
        _dir, _index, _name = i[:-1].split(" ")
        dir_list.append(_dir)
        index_list.append(int(_index))
        name_list.append(_name)

    return (dir_list, index_list, name_list)
