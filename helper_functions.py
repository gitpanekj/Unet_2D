import tensorflow as tf
import numpy as np
from typing import Optional, List
import tensorflow.keras.backend as K

def _gather_channels(x: np.array, indexes: Optional[List[int]]=None) -> np.array:
    """ Return n channels according to specified indexes 

        Only supported for array axes order format (B,Z,Y,X,C)
    """
    if not isinstance(indexes, type(None)):
        x = tf.gather(x, indices=indexes, axis=-1)
    return x

def threshold_if_specified(x: np.array, threshold: Optional[float]=None) -> np.array:
    """ Return binary output according to threshold
    
        Only supported for axis oreder format (B,(Z),Y,X,C)
    """
    if threshold != None:
        x = K.greater(x, threshold)
        x = K.cast(x, tf.float32)
    return x

def calc_average(x, class_weights: Optional[List[float]]=None):
    """ Calculates weighted average """
    if class_weights:
        x = K.sum(x * class_weights)
        return x

    return K.mean(x)

@tf.function
def normalization(x, keepdims=True):
    x = (x - K.min(x, keepdims=keepdims))/(K.max(x, keepdims=keepdims) - K.min(x, keepdims=keepdims))
    return x

@tf.function
def standardization(x, keepdims=True):
    x = (x - K.mean(x, keepdims=keepdims))/K.var(x, keepdims=keepdims)

@tf.function
def min_max_scale(x, keepdims=True):
    min_ = K.min(x, keepdims=keepdims)
    return (x-min_)/(K.max(x, keepdims=keepdims)-min_)

def one_hot_encode(data, classes):
    n_classes = len(classes)
    data = np.squeeze(data)
    encoded = np.ndarray(data.shape + (n_classes,), dtype=np.uint8)
    labels = np.eye(n_classes)

    for i in classes:
        encoded[data == i] = labels[i]
    return encoded