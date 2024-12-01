from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_analysis as tfma
import segmentation_models as sm
from config import *

# Setting framework

sm.set_framework('tf.keras')
sm.framework()

# Keras MeanIoU
# ----------------------------------------------------------------------------------------------

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    '''
    Summary:
        MyMeanIOU inherit tf.keras.metrics.MeanIoU class and modifies update_state function.
        Computes the mean intersection over union metric.
        iou = true_positives / (true_positives + false_positives + false_negatives)
    Arguments:
        num_classes (int): The possible number of labels the prediction task can have
    Return:
        Class objects
    '''

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Summary:
        This functions get dice coefficient metric
    Arguments:
        y_true (float32): true label
        y_pred (float32): predicted label
        smooth (int): smoothness
    Return:
        dice coefficient metric
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_score(y_true, y_pred):
    return dice_coef(y_true, y_pred)
# Metrics
# ----------------------------------------------------------------------------------------------

def get_metrics():
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        empty
    Return:
        metrics directories
    """

    m = MyMeanIOU(num_classes)
    return {
        'MyMeanIOU': m,
        # 'f1-score': tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        # 'precision': sm.metrics.precision,
        # 'recall': sm.metrics.recall,
        'dice_coef_score':dice_coef_score
        # "keras_MIOU" : tf.keras.metrics.MeanIoU(num_classes=2)
    }
