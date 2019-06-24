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
    mesh = trimesh.load(args.MVSNetOutDir + "mesh5.off")
    vertexNormals = ComputeVertexNormals(mesh)
    di = mesh.vertices - cameras[truncatedViewGraph[0][0]].C
    alignedN = AlignNormals(vertexNormals, di)

    for currentPair in truncatedViewGraph:
        i = currentPair[0]
        j = currentPair[1]
        rayInt = rayMeshInt.RayMeshIntersector(mesh)
        Wi, index_ray_i, index_tri_i = cameras[i].GetIntersectionWithLocation(rayInt)
        # image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
        testPattern = np.zeros_like(cameras[i].image)
        x = np.arange(0, cameras[i].image.shape[1] / 1000.0, 0.001).astype(float)
        y = np.arange(0, cameras[i].image.shape[0] / 1000.0, 0.001).astype(float)
        xx, yy = np.meshgrid(x, y, sparse=True)
        #z = (np.sin(xx**2 + yy**2) + 1.0) / (0.01 * xx**2 + 0.01 * yy**2 + 1.0)
        z = (xx**2.0 + yy**2.0)
        testPattern[:,:,0] = Normalize01(z) * 200.0
        # meshColors = np.zeros_like(mesh.vertices)
        # validcolors, validVertexIndices = cameras[i].GetValuesAtWorldCoords(mesh.vertices, rayInt, testPattern)
        image_i, validMask_i = cameras[i].GetImageAtWorldCoords(Wi, index_tri_i, index_ray_i, rayInt, cameras[i].image)
        # pdb.set_trace()
        # plt.imshow(image_i.astype(int))
        # plt.show()
        # validcolors = Normalize01(validcolors) * 200.0
        # meshColors[validVertexIndices, :] = validcolors # * 255.0
        #mesh.vertices[validVertexIndices = mesh.vertices[validVertexIndices, :] + validcolors * 10.0
        meshColors = np.zeros_like(mesh.vertices)
        # validcolors, validVertexIndices = cameras[i].GetValuesAtWorldCoords(mesh.vertices, rayInt, testPattern)
        # meshColors[validVertexIndices, 0] = Normalize01(np.sum(np.power(mesh.vertices[validVertexIndices, :], 2.0), axis = 1)) * 255.0
        # _, affectedVertexIndices = trimesh.proximity.ProximityQuery(mesh).vertex(Wi)
        # uniqueVertices, indices = np.unique(affectedVertexIndices, return_index = True) #TODO: Better to find closest
        # validImageVals = image_i[:,:,0].flatten()[index_ray_i]
        # validImageValsVertices = validImageVals[indices]
        validImageValsVertices, validVertices = cameras[i].GetColorAtVertices(mesh, Wi, index_ray_i, cameras[i].image)
        # vertexWorldCoord_i = Wi[indices]
        meshColors[validVertices, :] = validImageValsVertices # Normalize01(np.sum(np.power(mesh.vertices[uniqueVertices, :], 2.0), axis = 1)) * 255.0        
        mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=meshColors.astype(np.uint8))
        mesh.export(args.MVSNetOutDir + "meshColor.ply")
