import numpy as np 
import cv2 
from scipy.ndimage import filters
import pdb
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor

def FilterImage(image, sigma):
    return filters.gaussian_filter(image, sigma, truncate = 3.0)

# Computes alpha, beta and gamma to be used as 
# per eq 13 of Multi-View Stereo Reconstruction and 
# Scene Flow:
def GetCorrelationGradientWeights(image1, mask1, image2, mask2, smoothingSigma = 3.0):
    beta_square = 1.0 # from paper page 14
    maskInt = np.multiply(mask1, mask2).astype(float)
    image1 = np.multiply(image1, maskInt).astype(float)
    image2 = np.multiply(image2, maskInt).astype(float)
    u1 = np.multiply(FilterImage(image1, smoothingSigma), maskInt)
    u2 = np.multiply(FilterImage(image2, smoothingSigma), maskInt)
    v1 = np.multiply(FilterImage(np.multiply(image1, image1), smoothingSigma), maskInt) - np.multiply(u1, u1) + beta_square
    v2 = np.multiply(FilterImage(np.multiply(image2, image2), smoothingSigma), maskInt) - np.multiply(u2, u2) + beta_square
    v12 = np.multiply(FilterImage(np.multiply(image1, image2), smoothingSigma), maskInt) - np.multiply(u1, u2)
    v1v2SqrtInv = np.divide(1.0, np.sqrt(np.multiply(v1, v2)))
    cc = np.multiply(v12, v1v2SqrtInv)
    alpha = np.multiply(FilterImage(-v1v2SqrtInv, smoothingSigma), maskInt)
    beta = np.multiply(FilterImage(np.divide(cc, v2), smoothingSigma), maskInt)
    gamma = np.multiply(FilterImage(np.subtract(np.multiply(u1, v1v2SqrtInv), np.divide(np.multiply(u2, cc), v2)), smoothingSigma), maskInt)
    # fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=2)
    # ax0[0].imshow(alpha.astype(float))
    # ax1[0].imshow(beta.astype(float))
    # ax0[1].imshow(gamma.astype(float))
    # ax1[1].imshow(cc.astype(float))
    # multi = MultiCursor(fig.canvas, (ax0[0], ax1[0], ax0[1], ax1[1]), color='r', lw=1, horizOn=True, vertOn=True)
    # pdb.set_trace()
    # plt.show()
    return alpha, beta, gamma, maskInt

# def GetMeanCurvature(mesh):

def GetImageGradient(image):
    image = image.astype(float)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    for d in range(image.shape[2]):
        gx[:,:,d], gy[:,:,d] = np.gradient(image[:,:,d])

    return gx, gy

def GetProjectionMatrixJacobian(cam, W):
    fx = cam.K[0,0]
    fy = cam.K[1,1]
    u0 = cam.K[0,2]
    v0 = cam.K[1,2]
    C = np.matmul(W, cam.R[2,:]) + cam.t[2]
    A = fx * (np.matmul(W, cam.R[0,:]) + cam.t[0]) + u0 * C
    B = fy * (np.matmul(W, cam.R[1,:]) + cam.t[1]) + v0 * C
    Ax = cam.R[0,0] * fx + cam.R[2,0] * u0
    Ay = cam.R[0,1] * fx + cam.R[2,1] * u0
    Az = cam.R[0,2] * fx + cam.R[2,2] * u0
    Bx = cam.R[1,0] * fy + cam.R[2,0] * v0
    By = cam.R[1,1] * fy + cam.R[2,1] * v0
    Bz = cam.R[1,2] * fy + cam.R[2,2] * v0
    Cx = cam.R[2,0]
    Cy = cam.R[2,1]
    Cz = cam.R[2,2]
    ux = np.divide(C * Ax - A * Cx, C * C)
    uy = np.divide(C * Ay - A * Cy, C * C)
    uz = np.divide(C * Az - A * Cz, C * C)
    vx = np.divide(C * Bx - B * Cx, C * C)
    vy = np.divide(C * By - B * Cy, C * C)
    vz = np.divide(C * Bz - B * Cz, C * C)
    return [ux, uy, uz, vx, vy, vz]

def ComputeFullSimilarityTermDerivative(grad2M, gradIjX, gradIjY, JacobianJ, d, zi, N):
    MIjX = np.multiply(grad2M[:,0], gradIjX[:,0]) + \
           np.multiply(grad2M[:,1], gradIjX[:,1]) + \
           np.multiply(grad2M[:,2], gradIjX[:,2])

    MIjY = np.multiply(grad2M[:,0], gradIjY[:,0]) + \
           np.multiply(grad2M[:,1], gradIjY[:,1]) + \
           np.multiply(grad2M[:,2], gradIjY[:,2])

    t1 = np.multiply(MIjX, JacobianJ[0]) + np.multiply(MIjY, JacobianJ[3])
    t2 = np.multiply(MIjX, JacobianJ[1]) + np.multiply(MIjY, JacobianJ[4])
    t3 = np.multiply(MIjX, JacobianJ[2]) + np.multiply(MIjY, JacobianJ[5])

    numer = np.multiply(t1, d[:,0]) + np.multiply(t2, d[:,1]) + np.multiply(t3, d[:,2])
    term = np.divide(np.expand_dims(numer, axis = 1), np.power(zi, 3))
    return term

# Computes image similarity using correlation per eq 12 of Multi-View Stereo Reconstruction and 
# Scene Flow, used for debugging purposes:
def GetCorrelation(image1, mask1, image2, mask2, smoothingSigma = 5.0):
    beta_square = 1.0
    maskInt = np.multiply(mask1, mask2)
    image1 = np.multiply(image1, maskInt).astype(float)
    image2 = np.multiply(image2, maskInt).astype(float)
    u1 = np.multiply(FilterImage(image1, smoothingSigma), maskInt)
    u2 = np.multiply(FilterImage(image2, smoothingSigma), maskInt)
    v1 = np.multiply(FilterImage(np.multiply(image1, image1), smoothingSigma), maskInt) - np.multiply(u1, u1) + beta_square
    v2 = np.multiply(FilterImage(np.multiply(image2, image2), smoothingSigma), maskInt) - np.multiply(u2, u2) + beta_square
    v12 = np.multiply(FilterImage(np.multiply(image1, image2), smoothingSigma), maskInt) - np.multiply(u1, u2)
    v1v2SqrtInv = np.divide(1.0, np.sqrt(np.multiply(v1, v2)))
    cc = np.multiply(v12, v1v2SqrtInv)
    return cc
