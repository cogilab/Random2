import numpy as np

def reliability_diagram(pred, conf, num_bin=10):
    """
    Computes reliability diagram metrics for the given predictions and confidences.

    Args:
        pred (array-like): Boolean array where each value indicates the correctness of a prediction.
        conf (array-like): Array of confidence scores for each prediction.
        num_bin (int, optional): Number of bins to partition the confidence scores. Default is 10.

    Returns:
        tuple: A tuple containing:
            - acc (list): List of accuracy values for each bin.
            - conf_bin (list): List of average confidence values for each bin.
            - num_sample (list): List of the number of samples in each bin.
    """
    bin_size = 1 / num_bin
    acc = []
    conf_bin = []
    num_sample = []

    for i in range(num_bin):
        bin_idx = (conf >= i*bin_size) & (conf < (i+1)*bin_size)
        if bin_idx.sum() == 0:
            acc.append(0)
        else:
            acc.append(np.mean(pred[bin_idx]))

        conf_bin.append(np.mean(conf[bin_idx]))
        num_sample.append(bin_idx.sum())

    conf_bin = np.arange(0, 1, bin_size) + bin_size / 2

    return acc, conf_bin, num_sample

def acc_conf_diff(acc, conf_bin):
    """
    Calculates the absolute difference between accuracy and confidence for each bin.

    Args:
        acc (list): List of accuracy values for each bin.
        conf_bin (list): List of average confidence values for each bin.

    Returns:
        np.ndarray: Array of absolute differences between accuracy and confidence.
    """
    return np.abs(np.array(acc) - np.array(conf_bin))

def ece(acc, conf_bin, num_sample):
    """
    Computes the Expected Calibration Error (ECE) for the model.

    Args:
        acc (list): List of accuracy values for each bin.
        conf_bin (list): List of average confidence values for each bin.
        num_sample (list): List of the number of samples in each bin.

    Returns:
        float: The ECE value.
    """
    return np.abs(np.array(acc) - np.array(conf_bin)).dot(np.array(num_sample)) / np.sum(num_sample)

def mce(acc, conf_bin):
    """
    Computes the Maximum Calibration Error (MCE) for the model.

    Args:
        acc (list): List of accuracy values for each bin.
        conf_bin (list): List of average confidence values for each bin.

    Returns:
        float: The MCE value.
    """
    return np.max(np.abs(np.array(acc) - np.array(conf_bin)))
