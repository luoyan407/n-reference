import numpy as np
from PIL import Image
import scipy.io
import scipy.ndimage


def calc_score(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    # return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

    # y,x = np.where(gtsAnn==1)
    # scores = []
    # for i in range(x.shape[0]):
    #     scores.append(salMap[y[i],x[i]])
    # return np.nanmean(np.array(scores))
    return np.sum(salMap*gtsAnn)/np.sum(gtsAnn)

def compute_score(sals, gts, image_size=(480, 640), sigma=-1.0, fxt_field_in_mat='fixationPts'):
    """
    Computes NSS score for a given set of predictions and fixations
    :param gts : dict : fixation points with "image name" key and list of points as values
    :param sals : dict : saliency map predictions with "image name" key and ndarray as values
    :param image_size: [height, width]
    :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
    """
    assert(len(gts) == len(sals))

    score = []
    for i in range(len(sals)):
        salmap = Image.open(sals[i])
        salmap = np.array(salmap, dtype=np.float)
        if len(salmap.shape) == 3:
            salmap = np.mean(salmap, axis=2)

        mat = scipy.io.loadmat(gts[i])
        fixations = mat[fxt_field_in_mat]
        fixations = fixations.astype(np.bool)
        if image_size is not None:
            fixations = Image.fromarray(fixations)
            fixations = fixations.resize((image_size[1],image_size[0]), resample=Image.NEAREST)
            fixations = np.array(fixations)

        height_sal, width_sal = (salmap.shape[0],salmap.shape[1])
        if image_size is None:
            height_fx,width_fx = (fixations.shape[0],fixations.shape[1])
            salmap = scipy.ndimage.zoom(salmap, 
                (float(height_fx)/height_sal, float(width_fx)/width_sal), order=3)
        else:
            height_fx,width_fx = (image_size[0],image_size[1])
            salmap = scipy.ndimage.zoom(salmap, 
                (float(image_size[0])/height_sal, float(image_size[1])/width_sal), order=3)
        if sigma > 0:
            salmap = scipy.ndimage.filters.gaussian_filter(salmap, sigma)
        salmap -= np.min(salmap)
        salmap = salmap /np.max(salmap)

        score.append(calc_score(fixations,salmap))
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)
