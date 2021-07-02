import numpy as np
from sklearn.metrics import confusion_matrix


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_mIoU(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = (intersection + smooth ) / (union.astype(np.float32) + smooth)
        iou_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union, iou_list
    
    def init_op(self):
        self.overall_confusion_matrix = None





class RunningDice():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Dice 
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=0):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        if len(ground_truth.shape) > 1 or len(prediction.shape) > 1:
            ground_truth = ground_truth.flatten()
            prediction = prediction.flatten()
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_dice(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set

        intersection_over_union = (2*intersection + smooth ) / (union.astype(np.float32) + smooth)
        dice_list = [round(case,4) for case in intersection_over_union]
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union, dice_list
    
    def init_op(self):
        self.overall_confusion_matrix = None



def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = -1.0 * np.ones((num_classes),dtype=np.float32)
    for i in range(num_classes):
        if i+1 not in y_true and i+1 not in y_pred:
            continue
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list[i] = round(dice,4)

    dice_list = np.where(dice_list == -1.0, np.nan, dice_list)
    
    return dice_list, round(np.nanmean(dice_list),4)