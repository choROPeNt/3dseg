import importlib

import numpy as np
import torch
import torch.nn as nn

from skimage import measure
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error

from .losses import compute_per_channel_dice
from .seg_metrics import AveragePrecision, Accuracy
from .utils import get_logger, expand_as_one_hot, convert_to_numpy


logger = get_logger('EvalMetric')


class F1Score(nn.Module):
    """
    Compute F1 Score for multi-channel 3D segmentation.
    
    Supports:
    - probabilities (model outputs after activiation)
    - Thresholded binary masks (0s and 1s)
    - Skipping specific channels
    
    Args:
        threshold (float): Threshold for converting probabilities to binary (default: 0.5).
        smooth (float): Smoothing factor to avoid division by zero.
        skip_channels (list[int]): List of channel indices to skip during F1 calculation.
    """
    def __init__(self, threshold=0.5, smooth=1e-6, skip_channels=None,**kwargs):
        super(F1Score, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
        self.skip_channels = skip_channels if skip_channels else []

    
    def forward(self, pred, target):
        """
        Compute F1 Score.

        Args:
            pred (torch.Tensor): Predicted mask (logits or binary), shape [batch, channels, depth, height, width].
            target (torch.Tensor): Ground truth mask (binary), shape [batch, channels, depth, height, width].

        Returns:
            torch.Tensor: Mean F1 Score across valid channels.
        """
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        assert pred.min() >= 0.0 and pred.max() <= 1.0, \
                f"Input `pred` must be probabilities or binary values in [0, 1], got range ({pred.min().item()}, {pred.max().item()})"

        # Convert probabilities to binary masks
        pred = (pred > self.threshold).float()
        target = target.float()

        # Compute per-channel F1 score
        f1_per_channel = []
        num_channels = pred.shape[1]  # Channel dimension is at index 1

        for ch in range(num_channels):
            if ch in self.skip_channels:
                continue  # Skip this channel

            # Compute True Positives (TP), False Positives (FP), False Negatives (FN)
            tp = (pred[:, ch] * target[:, ch]).sum()  # Sum over spatial dimensions
            fp = (pred[:, ch] * (1 - target[:, ch])).sum()
            fn = ((1 - pred[:, ch]) * target[:, ch]).sum()

            # Compute F1 Score for this channel (batch-wise)
            f1 = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

            # Take the mean F1 score over the batch
            f1_per_channel.append(f1.mean())

        if len(f1_per_channel) == 0:
            return torch.tensor(0.0, device=pred.device)  # Return 0 if all channels are skipped

        # Compute mean F1 across valid channels
        return torch.stack(f1_per_channel).mean()


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]
        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()
            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. 
        Returns byte tensor of the same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)

        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / \
            torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class AdaptedRandError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_segm()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_segm` method.

    Args:
        use_last_target (bool): use only the last channel from the target to compute the ARand
    """

    def __init__(self, use_last_target=False, ignore_index=None, **kwargs):
        self.use_last_target = use_last_target
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Compute ARand Error for each input, target pair in the batch and return the mean value.

        Args:
            input (torch.tensor): 5D (NCDHW) output from the network
            target (torch.tensor): 4D (NDHW) ground truth segmentation

        Returns:
            average ARand Error across the batch
        """

        def _arand_err(gt, seg):
            n_seg = len(np.unique(seg))
            if n_seg == 1:
                return 0.
            return adapted_rand_error(gt, seg)[0]

        # converts input and target to numpy arrays
        input, target = convert_to_numpy(input, target)
        if self.use_last_target:
            target = target[:, -1, ...]  # 4D
        else:
            # use 1st target channel
            target = target[:, 0, ...]  # 4D

        # ensure target is of integer type
        target = target.astype(np.int32)

        if self.ignore_index is not None:
            target[target == self.ignore_index] = 0

        per_batch_arand = []
        for _input, _target in zip(input, target):
            n_clusters = len(np.unique(_target))
            # skip ARand eval if there is only one label in the patch due to the zero-division error in Arand impl
            # xxx/skimage/metrics/_adapted_rand_error.py:70: RuntimeWarning: invalid value encountered in double_scalars
            # precision = sum_p_ij2 / sum_a2
            if n_clusters == 1:
                logger.info('Skipping ARandError computation: only 1 label present in the ground truth')
                per_batch_arand.append(0.)
                continue

            # convert _input to segmentation CDHW
            segm = self.input_to_segm(_input)
            assert segm.ndim == 4

            # compute per channel arand and return the minimum value
            per_channel_arand = [_arand_err(_target, channel_segm) for channel_segm in segm]
            per_batch_arand.append(np.min(per_channel_arand))

        # return mean arand error
        mean_arand = torch.mean(torch.tensor(per_batch_arand))
        logger.info(f'ARand: {mean_arand.item()}')
        return mean_arand

    def input_to_segm(self, input):
        """
        Converts input tensor (output from the network) to the segmentation image. E.g. if the input is the boundary
        pmaps then one option would be to threshold it and run connected components in order to return the segmentation.

        :param input: 4D tensor (CDHW)
        :return: segmentation volume either 4D (segmentation per channel)
        """
        # by deafult assume that input is a segmentation volume itself
        return input


class BoundaryAdaptedRandError(AdaptedRandError):
    """
    Compute ARand between the input boundary map and target segmentation.
    Boundary map is thresholded, and connected components is run to get the predicted segmentation
    """

    def __init__(self, thresholds=None, use_last_target=True, ignore_index=None, input_channel=None, invert_pmaps=True,
                 save_plots=False, plots_dir='.', **kwargs):
        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, save_plots=save_plots,
                         plots_dir=plots_dir, **kwargs)

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel
        self.invert_pmaps = invert_pmaps

    def input_to_segm(self, input):
        if self.input_channel is not None:
            input = np.expand_dims(input[self.input_channel], axis=0)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # threshold probability maps
                predictions = predictions > th

                if self.invert_pmaps:
                    # for connected component analysis we need to treat boundary signal as background
                    # assign 0-label to boundary mask
                    predictions = np.logical_not(predictions)

                predictions = predictions.astype(np.uint8)
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label(predictions, background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAdaptedRandError(AdaptedRandError):
    def __init__(self, input_channels, thresholds=None, use_last_target=True, ignore_index=None, invert_channels=None,
                 **kwargs):

        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, **kwargs)
        assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
        self.input_channels = input_channels
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        if invert_channels is None:
            invert_channels = []
        self.invert_channels = invert_channels

    def input_to_segm(self, input):
        # pick only the channels specified in the input_channels
        results = []
        for i in self.input_channels:
            c = input[i]
            # invert channel if necessary
            if i in self.invert_channels:
                c = 1 - c
            results.append(c)

        input = np.stack(results)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label((predictions > th).astype(np.uint8), background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if target.dim() == 5:
            if self.use_last_target:
                target = target[:, -1, ...]  # 4D
            else:
                # use 1st target channel
                target = target[:, 0, ...]  # 4D

        input1 = input2 = input
        multi_head = isinstance(input, tuple)
        if multi_head:
            input1, input2 = input

        input1, input2, target = convert_to_numpy(input1, input2, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp1, inp2, tar in zip(input1, input2, target):
            if multi_head:
                inp = (inp1, inp2)
            else:
                inp = inp1

            segs = self.input_to_seg(inp, tar)  # expects 4D
            assert segs.ndim == 4
            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            logger.info(f'Batch: {i_batch}. Max Average Precision for channel: {np.argmax(segs_aps)}')
            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 0-index
        :param input: input instance segmentation
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BlobsBoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction, boundary prediction and ground truth instance segmentation.
    Segmentation mask is computed as (P_mask - P_boundary) > th followed by a connected component
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds

    def input_to_seg(self, input, target=None):
        # input = P_mask - P_boundary
        input = input[0] - input[1]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            seg = measure.label(np.logical_not(input > th).astype(np.uint8), background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class PSNR:
    """
    Computes Peak Signal to Noise Ratio. Use e.g. as an eval metric for denoising task
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return peak_signal_noise_ratio(target, input)


class MSE:
    """
    Computes MSE between input and target
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(input, target)


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('torch3dseg.utils.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)


def get_evaluation_metrics(config):
    """
    Returns a dict of evaluation metric instances keyed by their name.
    :param config: (dict) a top-level configuration object containing the 'eval_metrics' key
    :return: a dict {name: metric_instance}
    """

    def _metric_class(class_name):
        m = importlib.import_module('torch3dseg.utils.metrics')
        return getattr(m, class_name)

    assert 'eval_metrics' in config, 'Could not find evaluation metrics configuration'
    metrics_config = config['eval_metrics']

    metrics = {}
    for metric_cfg in metrics_config:
        metric_name = metric_cfg['name']
        metric_class = _metric_class(metric_name)
        kwargs = {k: v for k, v in metric_cfg.items() if k != 'name'}
        metrics[metric_name] = metric_class(**kwargs)

    return metrics