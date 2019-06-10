import openmesh as om 
import numpy as np 
import trimesh 
import trimesh.ray.ray_pyembree as rayMeshInt
import scipy.interpolate
from scipy import ndimage
import cv2
import pdb

def PixelCoordToWorldCoord(K, R, t, u, v, depth):
    a = np.multiply(K[2,0], u) - K[0,0]
    b = np.multiply(K[2,1], u) - K[0,1]
    c = np.multiply(depth, (K[0,2] - np.multiply(K[2,2], u)))

    g = np.multiply(K[2,0], v) - K[0,1]
    h = np.multiply(K[2,1], v) - K[1,1]
    l = np.multiply(depth, (K[1,2] - np.multiply(K[2,2], v)))

    y = np.divide(l - np.divide(np.multiply(g, c), a), h - np.divide(np.multiply(g, b), a))
    x = np.divide((c - np.multiply(b,y)), a)
    z = depth
    C = np.concatenate(([x], [y], [z]), axis = 0)
    W = np.matmul(R.T, C - t)
    return W.T

def WorldCoordTopixelCoord(K, R, t, W):
    C = t + np.matmul(R, W.T)
    p = np.matmul(K, C)
    Xp = np.divide(p[0:1, :], p[2:3, :])
    Yp = np.divide(p[1:2, :], p[2:3, :])
    return Xp, Yp

class Camera:
    def __init__(self, name, K, R, t, image):
        self.name = name
        self.K = K
        self.R = R
        self.t = t
        self.image = image
        x = np.linspace(0, image.shape[1], image.shape[1], axis = -1)   
        y = np.linspace(0, image.shape[0], image.shape[0], axis = -1)
        xv, yv = np.meshgrid(x, y)
        depth = np.zeros_like(xv)
        self.C = PixelCoordToWorldCoord(K, R, t, [0], [0], [0])
        temp = PixelCoordToWorldCoord(K, R, t, xv.flatten(), yv.flatten(), depth.flatten() + 1)
        self.rays = temp - self.C
        self.opticalAxisRay = self.GetOpticalAxisRay()

    def GetIntersectionWithLocation(self, rayInt):
        loc, index_ray, index_tri = rayInt.intersects_location(np.tile(self.C, [self.rays.shape[0], 1]), self.rays, multiple_hits=False)
        return loc, index_ray, index_tri

    def GetOpticalAxisRay(self):
        return PixelCoordToWorldCoord(self.K, self.R, self.t, self.K[0, 2], self.K[1, 2], [1]) - self.C

    def GetImageAtPixelCoords(self, X, Y, indices, image):
        imageInterpR = np.zeros((image.shape[0], image.shape[1]))
        imageInterpG = np.zeros((image.shape[0], image.shape[1]))
        imageInterpB = np.zeros((image.shape[0], image.shape[1]))
        validMaskR = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        imageInterpR.flat[indices] = ndimage.map_coordinates(image[:,:,0], [Y, X], order=1)
        imageInterpG.flat[indices] = ndimage.map_coordinates(image[:,:,1], [Y, X], order=1)
        imageInterpB.flat[indices] = ndimage.map_coordinates(image[:,:,2], [Y, X], order=1)
        imageInterp = np.stack((imageInterpR, imageInterpG, imageInterpB), axis = 2)
        validMaskR.flat[indices] = True
        validMask = np.stack((validMaskR, validMaskR, validMaskR), axis = 2)
        return imageInterp, validMask

    def GetImageAtPixelCoordsRisky(self, X, Y, image):
        imageInterpR = ndimage.map_coordinates(image[:,:,0], [Y, X], order=1)
        imageInterpG = ndimage.map_coordinates(image[:,:,1], [Y, X], order=1)
        imageInterpB = ndimage.map_coordinates(image[:,:,2], [Y, X], order=1)
        imageInterp = np.stack((imageInterpR, imageInterpG, imageInterpB), axis = 2)
        return imageInterp

    def GetImageAtWorldCoords(self, W, WTriIndices, imageIndices, rayInt, imageToUse):
        rays = W - self.C
        #TODO: Check if rays need to be rescaled
        WValidTriIndices = rayInt.intersects_first(np.tile(self.C, [rays.shape[0], 1]), rays)
        # Only those intersections are valid which occur at the same triangle as we wanted:
        validWorld = W[WValidTriIndices == WTriIndices, :]
        validXp, validYp = WorldCoordTopixelCoord(self.K, self.R, self.t, validWorld)
        imageValidIndices = imageIndices[WValidTriIndices == WTriIndices]
        image, validMask = self.GetImageAtPixelCoords(validXp, validYp, imageValidIndices, imageToUse)
        return image, validMask

    #TODO: Assumes that all world coordinates are visible!
    def GetImageAtWorldCoordsRisky(self, W, imageToUse):
        validXp, validYp = WorldCoordTopixelCoord(self.K, self.R, self.t, W)
        image = self.GetImageAtPixelCoordsRisky(validXp, validYp, imageToUse)
        return image

    def ComputeDepth(self, W):
        rays = W - self.C
        return np.dot(rays, self.opticalAxisRay.T)
    
def AlignNormals(N, ref):
    dotP = np.sum(np.multiply(N, ref), axis = 1)
    N[dotP < 0, :] = N[dotP < 0, :] * -1
    return N
    