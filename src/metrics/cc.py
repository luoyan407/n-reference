import numpy as np
from PIL import Image
import scipy.io
import scipy.ndimage


def calc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def compute_score(sals, gts, image_size=(480, 640), sigma=-1.0):
    """
    Computes CC score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : salmap predictions with "image name" key and ndarray as values
    :returns: average_score: float (mean CC score computed by averaging scores for all the images)
    """

    assert(len(gts) == len(sals))
    score = []
    for i in range(len(sals)):
        
        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)
            
        gtmap = Image.open(gts[i])
        gtmap = np.array(gtmap, dtype=np.float)

        height_sal, width_sal = (salmap.shape[0],salmap.shape[1])
        height_fx,width_fx = (gtmap.shape[0],gtmap.shape[1])
        if image_size is None:
            salmap = scipy.ndimage.zoom(salmap, 
                (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        else:
            salmap = scipy.ndimage.zoom(salmap, 
                (float(image_size[0])/height_sal, float(image_size[1])/width_sal), order=3)
            gtmap = scipy.ndimage.zoom(gtmap, 
                (float(image_size[0])/height_fx, float(image_size[1])/width_fx), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap /np.max(salmap)

        score.append(calc_score(gtmap,salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)