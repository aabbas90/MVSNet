# import openmesh as om 
import numpy as np 
from IO import *
from graphicsHelper import *
import argparse
import trimesh 
import trimesh.proximity
import scipy.interpolate
from scipy import ndimage
from refinementHelper import *
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refine mesh generated from MVSNet.')
    parser.add_argument('--MVSNetOutDir', default='../TEST_DATA_FOLDER_SIMULATED/', type=str,
                        help='MVSNet out path')
    parser.add_argument('--NumViews', default=1, type=int,
                        help='num neighbouring views for multi-view refinement')
    args = parser.parse_args()
    _, viewGraph = gen_pipeline_mvs_list(args.MVSNetOutDir, args.NumViews)
    truncatedViewGraph = viewGraph 
    cameras = LoadInputs(args.MVSNetOutDir)
    image1 = cameras[0].image.astype(float)
    validMask = np.zeros_like(image1)
    validMask[image1 > 0] = 1
    noise = 30 * np.random.rand(image1.shape[0], image1.shape[1], image1.shape[2])
    noise = np.multiply(noise, validMask)
    image2 = (image1 + noise)
    initialImage2 = image2.copy()
    validImage1 = np.multiply(image1, validMask)
    validImage2 = np.multiply(image2, validMask)
    numItr = 15
    dT = 0.2
    for itr in range(50):
        ccArr = GetCorrelation(image1, validMask, image2, validMask)
        cc = np.mean(ccArr[validMask > 0])
        print("cc: {}".format(cc))
        alpha, beta, gamma, maskInt = GetCorrelationGradientWeights(image1, validMask, image2, validMask)
        del2M = np.multiply(np.multiply(alpha, image1) + np.multiply(beta, image2) + gamma, maskInt)
        image2 = image2 - dT * del2M

    minV = np.min(del2M)
    maxV = np.max(del2M)
    del2M = np.divide(del2M - minV, maxV - minV)
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=2)
    ax0[0].imshow(image1.astype(int))
    ax1[0].imshow(image2.astype(int))
    ax1[1].imshow(initialImage2.astype(int))
    ax0[1].imshow(np.subtract(image2, image1).astype(float))
    multi = MultiCursor(fig.canvas, (ax0[0], ax1[0], ax0[1], ax1[1]), color='r', lw=1, horizOn=True, vertOn=True)
    plt.show()