# libraries
from tensorflow.keras.losses import Loss
from tensorflow import float32
import numpy as np
from typing import Optional, List
from tensorflow.keras import backend as K


# repo imports
from models.internals.metrics import JaccardIndex, DiceScore
from helper_functions import _gather_channels, calc_average, threshold_if_specified



class JaccardLoss(Loss):
    """ Calculate Jaccard loss (1 - IoU ("Intersection over Union"))

        Calculates Jaccard loss on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Loss.

        JaccardScore(y,y_hat) = (intersection/union = (tp)/(tp+fp+fn))
        JaccardLoss = 1. - JaccardScore
    """
    def __init__(self,
                 class_indexes: Optional[List[int]]=None,
                 class_weights: Optional[List[float]]=None,
                 threshold: Optional[float]=None,
                 dimensions: Optional[int]=2) -> None:
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Jaccards loss is calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexes) must equal len(class_weights)
            if None, calculate normal average
        threshold : float
            value used to threshold y_pred
            if None, do not perform thresholding
        """

        super(JaccardLoss, self).__init__(name="jaccard_loss")
        self.class_indexes = class_indexes
        self.class_weights = class_weights
        self.threshold = threshold
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]
        self.epsilon = 1e-5 # avoids division by zero

    def call(self, y_true: np.array, y_pred: np.array) -> float:
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = K.cast(y_pred, np.float32)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)
        y_true = K.cast(y_true, y_pred.dtype)

        intersection = K.sum(y_pred * y_true, axis=self.axes)
        union = K.sum(y_true + y_pred, axis=self.axes) - intersection

        score = (intersection + self.epsilon) / (union + self.epsilon)
        score = calc_average(score, class_weights=self.class_weights)
        return 1. - score


class DiceLoss(Loss):
    """ Calculate Dice loss (also (1 - F1 score))

        Calculates Dice loss on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Loss.

        DiceScore(y,y_hat) = 2*((precision*recall)/(precision+recall))\
            = ( ((1 + beta^2)*tp)/(1 + beta^2)*tp + beta^2*fn + fp)
        DiceLoss = 1. - DiceScore
    """

    def __init__(self,
                 class_indexes: Optional[List[int]]=None,
                 class_weights: Optional[List[int]]=None,
                 beta: int=1,
                 threshold: Optional[float]=None,
                 dimensions: Optional[int] = 2) -> None:
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Dice loss is calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexes) must equal len(class_weights)
            if None, calculate normal average
        threshold : float
            value used to threshold y_pred
            if None, do not perform thresholding
        beta : int
            indicates that recall is beta times as important as precision
        """

        super(DiceLoss, self).__init__(name='dice_loss')
        self.class_indexes = class_indexes
        self.class_weights = class_weights
        self.beta = beta
        self.threshold = threshold
        self.epsilon= 1e-5 # avoid division by 0
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]
    
    def call(self, y_true: np.array, y_pred: np.array) -> float:
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)

        y_true = K.cast(y_true, y_pred.dtype)

        tp = K.sum(y_true * y_pred, axis=self.axes)
        fn = K.sum(y_true, axis=self.axes) - tp
        fp = K.sum(y_pred, axis=self.axes) - tp

        score = ((1 + self.beta ** 2) * tp + self.epsilon) / (
                    (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.epsilon)
        score = calc_average(score, class_weights=self.class_weights)
        return 1. - score


class BinaryFocalLoss(Loss):
    """ Calculates binary focal loss

        A modification of standard binary cross entropy.
        Inherits from tf.losses.Loss.

        BCE - binary cross entropy
        focal = -(1-p_t)^gamma * BCE(y_true, y_pred)
    """
    def __init__(self, gamma=0., alpha=0.5):
        super(BinaryFocalLoss, self).__init__(name='binary_focal_loss')
        self.gamma = gamma
        self.alpha = alpha
        """
        Parameters
        ----------
        gamma : float
            The higher gamma is the lower loss will.
            relatively correct prediction return.
            If gamma is 0 focal normal CE is calculated.
        """
    
    def call(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        y_true = K.cast(y_true, y_pred.dtype)

        positive_loss = (-y_true*K.pow(1 - y_pred, self.gamma)*K.log(y_pred))
        negative_loss = -(1 - y_true)*K.pow(y_pred, self.gamma)*K.log(1 - y_pred)

        return K.mean(positive_loss + negative_loss)


class CategoricalFocalLoss(Loss):
    """ Calculates categorical focal loss

        A modification of standard categorical cross entropy.
        Inherits from tf.losses.Loss.

        CCE - categorical cross entropy
        focal = -(1-y_pred)^gamma * CCE(y_true, y_pred)
    """
    def __init__(self, gamma, class_weights: Optional[List[float]]=None, class_indexes: Optional[List[int]]=None):
        """
        Parameters
        ----------
        gamma : float
            The higher gamma is the lower loss will
            relatively correct prediction return.
            If gamma is 0 focal normal CE is calculated
        class_indexes : list[int]
            list of channels(classes) the Jaccards lossis calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexes) must equal len(class_weights)
            if None, calculate normal average
        """
        super(CategoricalFocalLoss, self).__init__(name='categorical_focal_loss')
        self.indexes = class_indexes
        self.gamma = gamma
        self.class_weights = class_weights


    def call(self, y_true: np.array, y_pred: np.array) -> float:
        y_true = _gather_channels(y_true,indexes=self.indexes)
        y_pred = _gather_channels(y_pred,indexes=self.indexes)

        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        y_true = K.cast(y_true, y_pred.dtype)

        loss = -y_true*K.pow(1 - y_pred, self.gamma)*K.log(y_pred)

        axes = [0,1,2,3]
        loss_per_channel = K.mean(loss, axis=axes)
        return calc_average(loss_per_channel, class_weights=self.class_weights)