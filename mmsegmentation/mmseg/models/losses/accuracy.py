# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None, ignore_index=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh,
                        self.ignore_index)


def multi_class_accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy for each class separately.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...).
        target (torch.Tensor): The ground truth, shape (N, num_class, ...).
        ignore_index (int | None): The label index to be ignored. Default: None.
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        list[float]: List of accuracies, one for each class.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    num_classes = pred.size(1)
    assert pred.shape == target.shape, "Prediction and target shapes must match."

    accuracies = []
    eps = torch.finfo(torch.float32).eps

    for class_idx in range(num_classes):
        # Extract predictions and ground truth for the current class
        pred_class = pred[:, class_idx, ...]  # Predictions for the class
        target_class = target[:, class_idx, ...]  # GT for the class

        # Flatten for computation (if spatial dimensions exist)
        pred_class = pred_class.reshape(pred_class.size(0), -1)  # Shape: (N, num_elements)
        target_class = target_class.reshape(target_class.size(0), -1)  # Shape: (N, num_elements)

        pred_value, pred_label = pred_class.topk(maxk, dim=1)  # Top-k predictions
        pred_label = pred_label.transpose(0, 1)  # Shape: (maxk, N, num_elements)
        correct = pred_label.eq(target_class.unsqueeze(0))  # Compare with GT

        if thresh is not None:
            correct = correct & (pred_value > thresh).t()  # Apply threshold
        if ignore_index is not None:
            correct = correct[:, target_class != ignore_index]  # Ignore specified indices

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
            if ignore_index is not None:
                total_num = target_class[target_class != ignore_index].numel() + eps
            else:
                total_num = target_class.numel() + eps
            res.append(correct_k.mul_(100.0 / total_num))

        accuracies.append(res[0] if return_single else res)
    return accuracies  # List of accuracies for each class


class MultiClassAccuracy(nn.Module):
    """Accuracy calculation module for multiple classes."""

    def __init__(self, topk=(1,), thresh=None, ignore_index=None):
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Predictions of models, shape (N, num_class, ...).
            target (torch.Tensor): Ground truth, shape (N, num_class, ...).

        Returns:
            list[float]: The accuracies for each class.
        """
        return multi_class_accuracy(pred, target, self.topk, self.thresh, self.ignore_index)
