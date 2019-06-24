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
    truncatedViewGraph = viewGraph[-7:-6]
    i = truncatedViewGraph[0][0]
    j = truncatedViewGraph[0][1]
    cameras = LoadInputs(args.MVSNetOutDir)
    mesh = trimesh.load(args.MVSNetOutDir + "mesh5.off")
    stepSize = 10.0
    di = mesh.vertices - cameras[i].C
    vertexNormals = ComputeVertexNormals(mesh)
    alignedN = AlignNormals(vertexNormals, di)
    
    meshcopy1 = mesh.copy()
    meshcopy1.vertices = meshcopy1.vertices - stepSize * alignedN
    rayInt = rayMeshInt.RayMeshIntersector(meshcopy1)
    Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
    image_i_1, validMask_i_1 = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
    image_j_i_1, validMask_j_i_1 = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)
    cc1 = GetCorrelation(image_i_1, validMask_i_1, image_j_i_1, validMask_j_i_1)

    meshcopy2 = mesh.copy()
    meshcopy2.vertices = meshcopy2.vertices
    rayInt = rayMeshInt.RayMeshIntersector(meshcopy2)
    Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
    image_i_2, validMask_i_2 = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
    image_j_i_2, validMask_j_i_2 = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)
    cc2 = GetCorrelation(image_i_2, validMask_i_2, image_j_i_2, validMask_j_i_2)
    
    meshcopy3 = mesh.copy()
    meshcopy3.vertices = meshcopy3.vertices + stepSize * alignedN
    rayInt = rayMeshInt.RayMeshIntersector(meshcopy3)
    Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
    image_i_3, validMask_i_3 = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
    image_j_i_3, validMask_j_i_3 = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)
    cc3 = GetCorrelation(image_i_3, validMask_i_3, image_j_i_3, validMask_j_i_3)
    
    ccMax = np.max(np.concatenate((cc1, cc2, cc3)))
    cc1 = (1.0 + cc1 / ccMax) / 2.0
    cc2 = (1.0 + cc2 / ccMax) / 2.0
    cc3 = (1.0 + cc2 / ccMax) / 2.0
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=3)
    ax0[0].imshow(image_i_1.astype(int))
    ax0[1].imshow(image_j_i_1.astype(int))
    ax0[2].imshow(cc1.astype(float))

    ax1[0].imshow(image_i_2.astype(int))
    ax1[1].imshow(image_j_i_2.astype(int))
    ax1[2].imshow(cc2.astype(float))
    
    ax2[0].imshow(image_i_3.astype(int))
    ax2[1].imshow(image_j_i_3.astype(int))
    ax2[2].imshow(cc3.astype(float))
    
    multi = MultiCursor(fig.canvas, (ax0[0], ax0[1], ax0[2], ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]), color='r', lw=1, horizOn=True, vertOn=True)
    plt.show()