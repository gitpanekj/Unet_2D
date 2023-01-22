# libraries
import numpy as np
from typing import Optional, List
from helper_functions import _gather_channels, calc_average, threshold_if_specified
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric

# ------------------
# Metric Functions
# ------------------

class JaccardIndex(Metric):
    """ Calculate Jaccard score (also IoU - "Intersection over Union")

        Calculates Jaccard score on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Metric.

        JaccardScore(y,y_hat) = (intersection/union = (tp)/(tp+fp+fn))
    """

    def __init__(self, class_indexes: Optional[List[int]]=None,
                       threshold: Optional[float]=None,
                       class_weights: Optional[List[float]]=None,
                       dimensions: Optional[int] = 2,
                       name='JaccardIndex'):
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Jaccards score is calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexs) must equal len(class_weights)
            if None, calculate normal average
        threshold : float
            value used to threshold y_pred
            if None, do not perform thresholding
        beta : int
            indicates that recall is beta times as important as precision
        """

        super(JaccardIndex, self).__init__(name=name)
        self.jaccard_index = self.add_weight(name='jaccard', initializer='zeros')
        #self.last_result = tf.Variable([0.0])

        self.class_indexes = class_indexes
        self.threshold = threshold
        self.class_weights = class_weights
        self.epsilon = 1e-5
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]

    def update_state(self,y_true,y_pred, sample_weight):
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = K.cast(y_pred, np.float32)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)
        y_true = K.cast(y_true, y_pred.dtype)

        axes = self.axes

        intersection = K.sum(y_pred*y_true, axis=axes)
        union = K.sum(y_true + y_pred, axis=axes) - intersection

        score = (intersection + self.epsilon)/(union+self.epsilon)
        score = calc_average(score, class_weights=self.class_weights)
        self.jaccard_index.assign(score)

    def result(self):
        return self.jaccard_index


class DiceScore(Metric):
    """ Calculate Dice score (also F1 score)

        Calculates Dice score on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Metric.

        DiceScore(y,y_hat) = 2*((precision*recall)/(precision+recall))\
            = ( ((1 + beta^2)*tp)/(1 + beta^2)*tp + beta^2*fn + fp)
    """

    def __init__(self, beta: float=1,
                       class_indexes: Optional[List[int]]=None,
                       threshold: Optional[float]=None,
                       class_weights: Optional[List[float]]=None,
                       dimensions: Optional[int] = 2,
                       name="dice score"):
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Dice score is calculated on
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

        super(DiceScore, self).__init__(name='DiceScore')
        self.dice_score = self.add_weight(name='dice', initializer='zeros')
        self.beta = beta
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.class_weights = class_weights
        self.epsilon = 1e-5
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]
    
    def update_state(self, y_true, y_pred, sample_weight):
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)

        y_true = K.cast(y_true, y_pred.dtype)

        axes = self.axes

        tp = K.sum(y_true*y_pred, axis=axes)
        fn = K.sum(y_true, axis=axes) - tp
        fp = K.sum(y_pred, axis=axes) - tp

        score = ((1+self.beta**2) * tp + self.epsilon)/\
                ((1+self.beta**2)*tp + self.beta**2*fn + fp + self.epsilon)
        score = calc_average(score, class_weights=self.class_weights) 
        self.dice_score.assign(score)

    def result(self):
        return self.dice_score


class Precision(Metric):
    """ Calculate Precision

        Calculates Precision score on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Metric.

        Precision = (tp)/(tp+fp)
    """

    def __init__(self, class_indexes: Optional[List[int]]=None,
                       threshold: Optional[float]=None,
                       class_weights: Optional[List[float]]=None,
                       dimensions: Optional[int]=2,
                       name="precision"):
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Precision is calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexes) must equal len(class_weights)
            if None, calculate normal average
        threshold : float
            value used to threshold y_pred
            if None, do not perform thresholding
        """

        super(Precision, self).__init__(name='Precision')
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.class_weights = class_weights
        self.epsilon = 1e-5
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]
    
    def update_state(self, y_true, y_pred, sample_weight):
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)

        y_true = K.cast(y_true, y_pred.dtype)

        axes = self.axes

        tp = K.sum(y_true*y_pred, axis=axes)
        score = (tp + self.epsilon)/(K.sum(y_pred, axis=axes) + self.epsilon)
        score = calc_average(x=score, class_weights=self.class_weights)
        self.precision.assign(score)

    def result(self):
        return self.precision


class Recall(Metric):
    """ Calculate Recall

        Calculates Recall score on specified channels.
        Threshold inputs if specified.
        Calculate weighted average if specified.

        Inherits from tf.losses.Metric.

        Recall = (tp)/(tp+fn)
    """

    def __init__(self, class_indexes: Optional[List[int]]=None,
                       threshold: Optional[float]=None,
                       class_weights: Optional[List[float]]=None,
                       dimensions: Optional[int]=2,
                       name="recall"):
        """
        Parameters
        ----------
        class_indexes : list[int]
            list of channels(classes) the Precision is calculated on
            if None use all the channels
        class_weights : list[float]
            list of weights used to calculate weighted average
            len(class_indexes) must equal len(class_weights)
            if None, calculate normal average
        threshold : float
            value used to threshold y_pred
            if None, do not perform thresholding
        """

        super(Recall, self).__init__(name='Recall')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.class_weights = class_weights
        self.epsilon = 1e-5 
        self.axes = [0,1,2] if dimensions==2 else [0,1,2,3]
    
    def update_state(self, y_true, y_pred, sample_weight):
        y_true = _gather_channels(y_true, indexes=self.class_indexes)
        y_pred = _gather_channels(y_pred, indexes=self.class_indexes)

        y_pred = threshold_if_specified(y_pred, threshold=self.threshold)

        y_pred = K.cast(y_pred, np.float32)
        y_true = K.cast(y_true, y_pred.dtype)

        axes = self.axes

        tp = K.sum(y_true*y_pred, axis=axes)
        score = (tp + self.epsilon)/(K.sum(y_true, axis=axes) + self.epsilon)
        score = calc_average(x=score, class_weights=self.class_weights)
        self.recall.assign(score)

    def result(self):
        return self.recall