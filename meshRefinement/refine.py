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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refine mesh generated from MVSNet.')
    parser.add_argument('--MVSNetOutDir', default='../TEST_DATA_FOLDER_SIMULATED/', type=str,
                        help='MVSNet out path')
    parser.add_argument('--NumViews', default=1, type=int,
                        help='num neighbouring views for multi-view refinement')
    args = parser.parse_args()
    _, viewGraph = gen_pipeline_mvs_list(args.MVSNetOutDir, args.NumViews)
    truncatedViewGraph = viewGraph #[-7:-3]
    cameras = LoadInputs(args.MVSNetOutDir)
    mesh = trimesh.load(args.MVSNetOutDir + "mesh5.off")
    mesh.vertices = mesh.vertices + 2.0
    mesh.export(args.MVSNetOutDir + "mesh5noisy.off")
    dT = 5.0
    numItr = 20
    for itr in range(numItr):
        for currentPair in truncatedViewGraph:
            i = currentPair[0]
            j = currentPair[1]
            rayInt = rayMeshInt.RayMeshIntersector(mesh)
            Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
            image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
            image_j_i, validMask_j_i = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)
            alpha, beta, gamma, maskInt = GetCorrelationGradientWeights(image_i, validMask_i, image_j_i, validMask_j_i)
            del2M = np.multiply(alpha, image_i) + np.multiply(beta, image_j_i) + gamma
            affectedVertexDist, affectedVertexIndices = trimesh.proximity.ProximityQuery(mesh).vertex(Wi)
            uniqueVertices, indices = np.unique(affectedVertexIndices, return_index = True) #TODO: Better to find closest
            vertexWorldCoord_i = Wi[indices]
            vertexRay_i = index_ray_i[indices]
            vertexIndex_tri_i = index_tri_i[indices]
            gradM, validMask_gradM = cameras[i].GetImageAtWorldCoords(vertexWorldCoord_i, vertexIndex_tri_i, vertexRay_i, rayInt, del2M)
            gradM[np.logical_not(validMask_gradM)] = 0
            validGradM = np.stack((gradM[:,:,0].flatten()[vertexRay_i],gradM[:,:,1].flatten()[vertexRay_i],gradM[:,:,2].flatten()[vertexRay_i]), axis = 1)
            vertexDepths = cameras[i].ComputeDepth(vertexWorldCoord_i)
            vertexNormals = mesh.vertex_normals[uniqueVertices]
#            vertexNormals = mesh.vertex_normals
            JacobianJ = GetProjectionMatrixJacobian(cameras[j], vertexWorldCoord_i)
            di = vertexWorldCoord_i - cameras[i].C
            alignedN = AlignNormals(vertexNormals, di)
            gradIjX, gradIjY = GetImageGradient(cameras[j].image)
            gradIjXValid = np.squeeze(cameras[j].GetImageAtWorldCoordsRisky(vertexWorldCoord_i, gradIjX))
            gradIjYValid = np.squeeze(cameras[j].GetImageAtWorldCoordsRisky(vertexWorldCoord_i, gradIjY))
            term = ComputeFullSimilarityTermDerivative(validGradM, gradIjXValid, gradIjYValid, JacobianJ, di, vertexDepths, alignedN)
            ccBefore = GetCorrelation(image_i, validMask_i, image_j_i, validMask_j_i)
            print("Correlation before {}".format(np.sum(ccBefore)))
            mesh.vertices[uniqueVertices, 0:1] = mesh.vertices[uniqueVertices, 0:1] + dT * np.multiply(term, alignedN[:, 0:1])
            mesh.vertices[uniqueVertices, 1:2] = mesh.vertices[uniqueVertices, 1:2] + dT * np.multiply(term, alignedN[:, 1:2])
            mesh.vertices[uniqueVertices, 2:3] = mesh.vertices[uniqueVertices, 2:3] + dT * np.multiply(term, alignedN[:, 2:3])
            mesh.vertex_normals = ComputeVertexNormals(mesh)
            Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
            image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
            image_j_i, validMask_j_i = cameras[j].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[j].image)

            ccAfter = GetCorrelation(image_i, validMask_i, image_j_i, validMask_j_i)
            print("Correlation after {}".format(np.sum(ccAfter)))
            print("Refined {} -> {} ".format(i, j))
 
        print("Completed {} iterations".format(itr))

    mesh.export(args.MVSNetOutDir + "refinedConv.off")
        # absGrad = np.abs(gradM)
        # gradMImage = (absGrad - np.min(absGrad))/np.ptp(absGrad).astype(float)
        # cv2.imshow('gradM',gradMImage) 
        # cv2.waitKey(0)
    # W0, index_ray, index_tri = cameras[44].GetIntersectionWithLocation(rayInt)
    # image = cameras[44].GetImageAtWorldCoords(W0, index_tri, index_ray, rayInt)
    # image = (image - np.min(image))/np.ptp(image).astype(float)
    # cv2.imshow('image1',image)
    # image = cameras[0].GetImageAtWorldCoords(W0, index_tri, index_ray, rayInt)
    # image = (image - np.min(image))/np.ptp(image).astype(float)
    # cv2.imshow('image2',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()