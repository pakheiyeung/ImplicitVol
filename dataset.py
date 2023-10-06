# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:16:28 2019

@author: pemb5552
"""
import torch
from torch.utils import data
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from random import uniform
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_erosion
from skimage.transform import resize
import torchio as tio
from scipy.spatial import cKDTree as KDTree
import nibabel as nib

import matplotlib
matplotlib.use("TKAgg")

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])#*360/2/math.pi

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data



def procrustes(X, Y, scaling=True, reflection=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def grid_translation(grid, x, y, z):
    grid_copy = grid.copy()
    direction_row = grid[:,0,0]-grid[:,0,-1]
    direction_col = grid[:,0,0]-grid[:,-1,0]
    normal = np.cross(direction_row, direction_col)

    direction_row = unit_vector(direction_row)*x
    direction_col = unit_vector(direction_col)*y
    normal = unit_vector(normal)*z

    movement = direction_row+direction_col+normal
    grid_copy[0,:,:] = grid_copy[0,:,:]+movement[0]
    grid_copy[1,:,:] = grid_copy[1,:,:]+movement[1]
    grid_copy[2,:,:] = grid_copy[2,:,:]+movement[2]

    return grid_copy, movement

        
class Dataset_volume_video(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, vol_file, **params):
        'Initialization'
        self.set_size = params['set_size']
        self.mode = params['mode']
        
        
        self.create_train(vol_file)
        self.create_test()
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        store  = self.overall_store[index]
        
        img_slice = store['img']
        rot_ground = store['rot_ground']
        trans_ground = store['trans_ground']
        rot_pred = store['rot_pred']
        trans_pred = store['trans_pred']
        
        return index, img_slice, rot_ground, trans_ground, rot_pred, trans_pred

    def create_train(self, vol_file):
        self.overall_store = {}
        self.list_IDs = []
        
        'Import ultasound volume'
        nii_data = nib.load(vol_file)
        self.img_vol = nii_data.get_fdata()
        self.img_vol = self.img_vol[2:162, 15:175, 0:160]   #Just to make the sub-feta001_T2w.nii.gz volume the same size for every dimension, it's not necessary for other volumes
        
        'normalize'
        self.img_vol = self.img_vol/self.img_vol.max()
        
        'Get reference grid'
        self.H, self.W, self.D = self.img_vol.shape    #160, 160, 160
        self.preparing_sampling_grid_ref(self.H)
        
        'Sample slices along the depth'
        for num in range(self.set_size):
            'Get the parameters for sampling'
            r_x = 0    #x axis rotation angle (in angle not radian)
            r_y = 0    #y axis rotation angle (in angle not radian)
            r_z = 0    #z axis rotation angle (in angle not radian)
            t_x = 0    # in-plane x translation
            t_y = 0    # in-plane y translation
            t_z = -self.D//4+num*(self.D/2/self.set_size)    #tranlation along the surface normal, from -40 to 40
            
            'sample single slice'
            img, real_translation = self.sampling_slice(r_x, r_y, r_z, t_x, t_y, t_z)
            t_x_real, t_y_real, t_z_real = real_translation
            
            'Can add noise here to mimic inaccurate pose prediction'
            r_x_pred = r_x
            r_y_pred = r_y
            r_z_pred = r_z
            t_x_pred = t_x_real
            t_y_pred = t_y_real
            t_z_pred = t_z_real

            'store the data'
            store = {'img': img,
                     'rot_ground': np.array((r_x/360*2*math.pi, r_y/360*2*math.pi, r_z/360*2*math.pi)),
                     'trans_ground': np.array((t_x_real, t_y_real, t_z_real)),
                     'rot_pred': np.array((r_x_pred/360*2*math.pi, r_y_pred/360*2*math.pi, r_z_pred/360*2*math.pi)),
                     'trans_pred': np.array((t_x_pred, t_y_pred, t_z_pred))}
            self.overall_store.update({num:store})
            self.list_IDs.append(num)
            
    def create_test(self):
        self.overall_store_test = {}
        
        'Sample slices along the depth'
        for num in range(self.set_size):
            'Get the parameters for sampling'
            r_x = 0    #x axis rotation angle (in angle not radian)
            r_y = 0    #y axis rotation angle (in angle not radian)
            r_z = 90    #z axis rotation angle (in angle not radian)
            t_x = 0    # in-plane x translation
            t_y = 0    # in-plane y translation
            t_z = -self.D//4+num*(self.D/2/self.set_size)    #tranlation along the surface normal, from -40 to 40
            
            'sample single slice'
            img, real_translation = self.sampling_slice(r_x, r_y, r_z, t_x, t_y, t_z)
            t_x_real, t_y_real, t_z_real = real_translation
        
            'store the data'
            store = {'img': img,
                     'rot_ground': np.array((r_x/360*2*math.pi, r_y/360*2*math.pi, r_z/360*2*math.pi)),
                     'trans_ground': np.array((t_x_real, t_y_real, t_z_real)),
                     }
            self.overall_store_test.update({num:store})
    
    
    def preparing_sampling_grid_ref(self, size):
        sampling_xrange = np.arange(-size//2,size//2)
        sampling_yrange = np.arange(-size//2,size//2)
        X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
        grid = np.dstack([X, Y])
        grid = np.concatenate((grid,np.zeros([self.H,self.W,1])),axis=-1)
        rotation = np.array(((1,0,0),(0,1,0),(0,0,1)))
        self.sampling_grid_ref = np.einsum('ji, mni -> jmn', rotation, grid)
        
    def sampling_slice(self, r_x, r_y, r_z, t_x, t_y, t_z):
        'Get sampling grid and real translation'
        rot_matrix = eulerAnglesToRotationMatrix((r_x/360*2*math.pi, r_y/360*2*math.pi, r_z/360*2*math.pi))
        sampling_grid = np.einsum('ji, imn -> jmn', rot_matrix, self.sampling_grid_ref)
        sampling_grid, real_translation = grid_translation(sampling_grid, t_x, t_y, t_z)
        
        sampling_grid += self.H//2  #from (-80,80) to (0,160)
        
        'Sample from volume'
        xx = np.arange(self.W)
        yy = np.arange(self.H)
        zz = np.arange(self.D)
        interp_arr = interpn((xx, yy, zz), self.img_vol, np.transpose(sampling_grid.reshape((3,self.H*self.W))), bounds_error=False, fill_value=0)
        
        return interp_arr.reshape((self.H, self.W)), real_translation
   
        
    def _overall_store(self):
        return self.overall_store
    
    def _sampling_grid_ref(self):
        return self.sampling_grid_ref
    
    def return_test(self):
        return self.overall_store_test
        
    
        



