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
    parser.add_argument('--MVSNetOutDir', default='../TEST_DATA_FOLDER/', type=str,
                        help='MVSNet out path')
    parser.add_argument('--NumViews', default=1, type=int,
                        help='num neighbouring views for multi-view refinement')
    args = parser.parse_args()
    _, viewGraph = gen_pipeline_mvs_list(args.MVSNetOutDir, args.NumViews)
    cameras = LoadInputs(args.MVSNetOutDir, scaleFactor=1.0)
    mesh = trimesh.load(args.MVSNetOutDir + "mesh5.off")
    truncatedViewGraph = [[46, 47]]
    for currentPair in truncatedViewGraph:
        i = currentPair[0]
        j = currentPair[1]
        rayInt = rayMeshInt.RayMeshIntersector(mesh)
        Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
        image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
        Wj, index_ray_j, index_tri_j = cameras[j].GetIntersectionWithLocation(rayInt)
        image_i_j, validMask_i_j = cameras[i].GetImageAtWorldCoords(Wj, index_tri_j, index_ray_j, rayInt, cameras[i].image)
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=2)
        ax0[0].imshow(image_i.astype(int))
        ax1[0].imshow(image_i_j.astype(int))
        ax0[1].imshow(validMask_i.astype(float))
        ax1[1].imshow(validMask_i_j.astype(float))
        multi = MultiCursor(fig.canvas, (ax0[0], ax1[0], ax0[1], ax1[1]), color='r', lw=1, horizOn=True, vertOn=True)
        plt.show()
        cv2.imwrite("../TEST_DATA_FOLDER_SIMULATED/" + "/images/00000046.jpg", image_i)
        cv2.imwrite("../TEST_DATA_FOLDER_SIMULATED/" + "/images/00000047.jpg", image_i_j)