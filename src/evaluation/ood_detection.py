import numpy as np

def cdf(data, bins=20, range=(0, 1)):
    """
    Computes the cumulative distribution function (CDF) of the given data.

    Args:
        data (array-like): Input data for which the CDF is computed.
        bins (int, optional): Number of bins for the histogram. Default is 20.
        range (tuple, optional): The lower and upper range of the bins.

    Returns:
        tuple: A tuple containing the CDF values and the bin edges.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=True)
    cdf = np.cumsum(hist)  # Calculate cumulative sum
    cdf = cdf / cdf[-1]  # Normalize by the last value to make the CDF range [0, 1]
    return cdf, bin_edges

def roc_curve(a, b):
    """
    Computes the ROC curve points for two datasets.

    Args:
        a (array-like): Scores or probabilities for the positive class.
        b (array-like): Scores or probabilities for the negative class.

    Returns:
        tuple: A tuple of false positive rates (FP) and true positive rates (TP).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b), "Length of input arrays must be the same."

    n = len(a)
    tp = np.zeros(n)
    fp = np.zeros(n)

    # Sort thresholds based on combined scores from both classes
    thresholds = np.sort(np.concatenate([a, b]))
    for i, threshold in enumerate(thresholds):
        tp[i] = np.sum(a >= threshold) / len(a)  # True Positive Rate
        fp[i] = np.sum(b >= threshold) / len(b)  # False Positive Rate
    
    return fp, tp

def auc(fp, tp):
    """
    Calculates the area under the ROC curve.

    Args:
        fp (array-like): False positive rates.
        tp (array-like): True positive rates.

    Returns:
        float: The calculated area under the curve (AUC).
    """
    return np.trapz(tp, fp)  # Integrate the ROC curve
