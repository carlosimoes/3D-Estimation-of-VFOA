#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('../python/')
import sintel_io as sio

# Generate some data
u,v = [np.random.rand(500,1000).astype('float32') for i in range(2)]
u *= 100
v *= 100

# Generate some nonsense camera matrices
I = np.random.rand(3,3)
E = np.random.rand(3,4)


# Test flow
sio.flow_write('./_test_flow.flo',u,v)
u_,v_ = sio.flow_read('./_test_flow.flo')
error = np.sqrt((u-u_)**2 + (v-v_)**2).mean()
print('Flow error: {}'.format(error))

# Test depth
sio.depth_write('./_test_depth.dpt',u)
u_ = sio.depth_read('./_test_depth.dpt')
error = np.abs(u-u_).mean()
print('Depth error: {}'.format(error))

# Test disparity
sio.disparity_write('./_test_disparity.png',v)
v_ = sio.disparity_read('./_test_disparity.png')
error = np.abs(v-v_).mean()
print('Disparity error: {}'.format(error))

# Test cam matrices
sio.cam_write('./_test_cam.cam',I,E)
i_,e_ = sio.cam_read('./_test_cam.cam')
error_e = np.abs(E-e_).mean()
error_i = np.abs(I-i_).mean()
print('Error int: {0}. Error ext: {1}'.format(error_i,error_e))

# Test segmentation
seg = u.astype('int32')
sio.segmentation_write('./_test_segmentation.png',seg)
seg_ = sio.segmentation_read('./_test_segmentation.png')
error = np.abs(seg-seg_).mean()
print('Segmentation error: {}'.format(error))

# Test and display some real data
FLOWFILE = '../../basic/data/out/temple_2/flow/frame_0001.flo'
DEPTHFILE = '../../stereo/data/package/training/depth/temple_2/frame_0001.dpt'
DISPFILE = '../../stereo/data/package/training/disparities/temple_2/frame_0001.png'
CAMFILE = '../../stereo/data/package/training/camdata_left/temple_2/frame_0001.cam'
SEGFILE = '../../segmentation/data/package/training/segmentation/temple_2/frame_0001.png';

# Load data
u,v = sio.flow_read(FLOWFILE)
depth = sio.depth_read(DEPTHFILE)
disp = sio.disparity_read(DISPFILE)
I,E = sio.cam_read(CAMFILE)
seg = sio.segmentation_read(SEGFILE)

# Display data
plt.figure()
plt.subplot(321)
plt.imshow(u,cmap='gray')
plt.title('u')
plt.subplot(322)
plt.imshow(v,cmap='gray')
plt.title('v')
plt.subplot(323)
plt.imshow(depth,cmap='gray')
plt.title('depth')
plt.subplot(324)
plt.imshow(disp,cmap='gray')
plt.title('disparity')
plt.subplot(325)
plt.imshow(seg,cmap='gray')
plt.title('Segmentation')

print(I)
print(E)

plt.show()
