import openmesh as om 
import numpy as np 
import os
import pdb
import cv2
from graphicsHelper import Camera

def LoadMesh(path):
    mesh = om.TriMesh()
    om.read_mesh(mesh, path)
    return mesh

def LoadInputs(path, scaleFactor = 0.5):
    # imageList = os.listdir(path + "/depths_mvsnet/")
    imageList = os.listdir(path + "/images/")
    Cameras = []
    for name in imageList:
        if not ".jpg" in name or "init" in name or "prob" in name:
            continue

        # image = cv2.imread(path + "/depths_mvsnet/" + name)
        image = scale_image(cv2.imread(path + "/images/" + name), scaleFactor)
#        K, R, t, _, _ = LoadCameraMatrices(path + "/depths_mvsnet/" + name[:-4] + ".txt")
        K, R, t, _, _ = LoadCameraMatrices(path + "/cams/" + name[:-4] + "_cam.txt", scaleFactor)
        Cameras.append(Camera(name, K, R, t, image))

    return Cameras

def LoadCameraMatrices(path, scale = 1.0):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    file = open(path)
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]
            
    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = words[28]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0

    cam = scale_camera(cam, scale)
    R = cam[0, :3, :3]
    t = np.array(cam[0, :3, 3:4])
    K = cam[1, :3, :3]
    dStart = cam[1, 3, 0]
    dInt = cam[1, 3, 1]
    return K, R, t, dStart, dInt

    
def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal: 
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def gen_pipeline_mvs_list(dense_folder, numViews):
    """ mvs input path list """
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()
    viewGraph = [] 
    # for each dataset
    mvs_list = []
    pos = 1
    for i in range(int(cluster_list[0])):
        paths = []
        # ref image
        ref_index = int(cluster_list[pos])
        pos += 1
        ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
        ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)
        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        currentView = [ref_index]
        for view in range(numViews):
            view_index = int(cluster_list[pos + 2 * view])
            view_image_path = os.path.join(image_folder, ('%08d.jpg' % view_index))
            view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
            paths.append(view_image_path)
            paths.append(view_cam_path)
            currentView.append(view_index)
        viewGraph.append(currentView)
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(paths)
    return mvs_list, viewGraph