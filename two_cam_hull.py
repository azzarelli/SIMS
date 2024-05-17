import os
import json
import numpy as np
import math
import torch


from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis

cx = 20.0
cy = 10.0
fx = 20.0
fy = 20.0



def focal2fov(focal, pixels): # From 4DGS
    return 2*math.atan(pixels/(2*focal))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

DEBUG = True

fp = "./data/two_cam/"
ext = '.jpg'

mask_fp = fp+'mask/'

mask_colors = ['B']

# Projection Parameters
trans = 0.
scale = 0.

zfar = 100.0
znear = 0.01


# Ensure path exists
if os.path.exists(fp+'transforms_train.json'):
    with open(fp+'transforms_train.json', 'r') as file:
        data = json.load(file)

    image_names = [dir for dir in os.listdir(fp) if ext in dir]

    mask_fps = [mask_fp+dir for dir in image_names]

    image_fps = [fp+dir for dir in image_names]

    frame_data = data['frames']
    selected_data = {
        'h':data['h'],
        'w':data['w'],

        'fovx': focal2fov(data['fl_x'], data['w']),
        'fovy': focal2fov(data['fl_y'], data['h'])
    }
    
    """
    The idea is to get the mask bounding box and projecting the two hulls comprised from each mask bounding box
    """

    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    
    from numpy import array, argwhere
    import matplotlib.patches as patches


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for frame in frame_data:

        # Process frame name:
        fname = frame['file_path'].split('/')[-1]+ext

        if fname in image_names:
            mask_image = Image.open(mask_fps[image_names.index(fname)]) 
            
            tensor_image = transform(mask_image)

            if tensor_image.shape[0] == 3:
                if 'B' in mask_colors: tensor_image = tensor_image[1].unsqueeze(0)

            # Convert to numpy and then make boolean
            rgb_np = tensor_image.squeeze(0).numpy()
            boolmask = (rgb_np>0.)*1

            # Get the AABB bounding box
            B = argwhere(boolmask)
            (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 

            if DEBUG:
                """Display the mask and the resulting AABB bounding box
                """
                rgb_image = tensor_image.permute(1, 2, 0)
                rgb_np = rgb_image.numpy()
                rgb_np = rgb_np.clip(0, 1)
                plt.imshow(boolmask, cmap='Greys')

                # Create a Rectangle patch
                rect = patches.Rectangle((xstart, ystart), xstop - xstart, ystop - ystart, linewidth=1, edgecolor='r', facecolor='none')
                
                # Add the rectangle to the plot
                plt.gca().add_patch(rect)

                plt.show()

            # W2C = np.array(frame['transform_matrix'])
            # R = W2C[:3,:3].transpose()
            # T = W2C[:3, 3]

            # # Get the camera to world
            # C2W = torch.from_numpy(np.linalg.inv(W2C))[:3, :].cpu()


            # world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
            # # .cuda()
            # projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=selected_data['fovx'], fovY=selected_data['fovy']).transpose(0,1)
            # # .cuda()

            # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # camera_center = world_view_transform.inverse()[3, :3]

            # selected_data[fname] = {
            #     'R':R,
            #     'T':T
            # }

            # camera = Cameras(fx=data['fl_x'], fy=data['fl_y'], cx=float(data['w']/2), cy=float(data['h']/2), camera_to_worlds=C2W, camera_type=CameraType.PERSPECTIVE)
            # ray_bundle = camera.generate_rays(camera_indices=0)


            # exit()
            
            # fig = vis.vis_camera_rays(camera)
            # fig.show()

