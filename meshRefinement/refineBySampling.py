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
    truncatedViewGraph = viewGraph[-7:]
    cameras = LoadInputs(args.MVSNetOutDir)
    mesh = trimesh.load(args.MVSNetOutDir + "mesh5.off")
    numSteps = 20
    stepSize = 1.0
    sumWeights = np.zeros((mesh.vertices.shape[0], 1))
    PCscores = np.zeros((mesh.vertices.shape[0], 1))
    bestPCscores = np.zeros((mesh.vertices.shape[0], 1))
    bestPCindex = np.zeros((mesh.vertices.shape[0], 1))
    vertexNormals = ComputeVertexNormals(mesh)
    di = mesh.vertices - cameras[truncatedViewGraph[0][0]].C
    alignedN = AlignNormals(vertexNormals, di)

    for step in range(numSteps):
        disp = float(step - int(numSteps / 2)) * stepSize
        meshcopy = mesh.copy()
        meshcopy.vertices = mesh.vertices + disp * alignedN
        rayInt = rayMeshInt.RayMeshIntersector(meshcopy)

        #vertexNormals = trimesh.triangles.normals(meshcopy.triangles) #mesh.vertex_normals
        for currentPair in truncatedViewGraph:
            i = currentPair[0]
            j = currentPair[1]
            Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
            image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
            image_j_i, validMask_j_i = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)
            validInt = np.multiply(validMask_i, validMask_j_i)
            cc = GetCorrelation(image_i, validMask_i, image_j_i, validMask_j_i)
            ccVertex, validVertexIndices = cameras[i].GetColorAtVertices(mesh, Wi, index_ray_i, cc)
            ccSumVertex = np.sum(ccVertex, axis = 1, keepdims = True)
            validBestPCscores = bestPCscores[validVertexIndices]
            validBestindex = bestPCindex[validVertexIndices]
            indices = ccSumVertex > validBestPCscores
            validBestPCscores[indices] = ccSumVertex[indices]
            validBestindex[indices] = disp
            bestPCscores[validVertexIndices] = validBestPCscores
            bestPCindex[validVertexIndices] = validBestindex
            sumWeights[validVertexIndices] = ccSumVertex + sumWeights[validVertexIndices]
            PCscores[validVertexIndices] = PCscores[validVertexIndices] + ccSumVertex * disp 
            print("Refined {} -> {} ".format(i, j))

        print("Completed {} steps".format(step))

    wa_displacement = np.divide(PCscores, sumWeights + 1.0)
    refined = mesh.copy()
    refined.vertices = mesh.vertices + wa_displacement * alignedN
    refined.export(args.MVSNetOutDir + "refinedWA.off")

    refinedMax = mesh.copy()
    refinedMax.vertices = mesh.vertices + bestPCindex * alignedN
    refinedMax.export(args.MVSNetOutDir + "refinedMax.off")