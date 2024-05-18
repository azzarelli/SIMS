import os
import json
import numpy as np
import math
import torch


from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis



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




def getTrapezoidVertices(fp, ext, znear:float=0.1, zfar:float=10.):
    # Ensure path exists
    assert os.path.exists(fp+'transforms_train.json'), 'Could not find the transorms_train.json file'
    
    mask_fp = fp+'mask/'


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
    
    from numpy import argwhere
    import matplotlib.patches as patches


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    hulls = []

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')


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

            W2C = np.array(frame['transform_matrix'])

            # Get the camera to world
            matrix = np.linalg.inv(W2C) #torch.from_numpy(W2C[:3, :]).cpu() #
            
            # Go from our (OpenCV) to Nerfstudio (OpenGL)
            # Essential y and z axis are flippd bu x-axis remain the same
            T_opencv_to_opengl = np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0,  -1, 0],
                    [0,  0,  0, 1]
                ])

            # Apply the transformation
            C2W = torch.from_numpy(T_opencv_to_opengl @ matrix @ T_opencv_to_opengl.T)[:3, :]
    
            # Get the nerfstudio Ray-Bundles for the selected cameras (to get the origins and direction in 3-D)
            camera = Cameras(fx=data['fl_x'], fy=data['fl_y'], cx=float(data['w']/2), cy=float(data['h']/2), camera_to_worlds=C2W, camera_type=CameraType.PERSPECTIVE)
            ray_bundle = camera.generate_rays(camera_indices=0)

            XY = np.array([[xstart, ystart],
                [xstart, ystop],
                [xstop, ystart],
                [xstop, ystop]
                ])

            # Get the rays at the corners of the 2-D AABB Box
            ray = ray_bundle[XY[:, 1], XY[:, 0]]
            
            if DEBUG:
                for r in ray:
                    ax.quiver(r.origins[0], r.origins[1], r.origins[2], r.directions[0], r.directions[1], r.directions[2], color='r', label='Quiver 1')

            # Get the vertices corresponding to our trapezoid
            top_vertices = ray.origins + znear * ray.directions
            base_vertcies = ray.origins + zfar * ray.directions

            vertices = torch.cat([top_vertices, base_vertcies], dim=0)

            hulls.append(vertices)

    if DEBUG:
        plt.show()


    return hulls

def getConvexHull(hulls):
    """Using some ChatGPT never hurt:
    """
    from scipy.spatial import ConvexHull
    
    points = torch.cat(hulls, dim=0).cpu().tolist()

    assert (len(points)) == 16, 'Inccorect number of vertices (16 are needed, 8 for each projection)'
    hull = ConvexHull(points)

    hull_vertices = hull.points[hull.vertices]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # The simplices are faces
    for simplex in hull.simplices:
        poly = [hull.points[simplex]]
        ax.add_collection3d(Poly3DCollection(poly, alpha=.25, linewidths=1, edgecolors='r'))
    

    for id, hull in enumerate(hulls):
        top = hull[:4]
        base = hull[4:]
        hull = torch.cat([base, top], dim=0)
        print(hull)

        c = 'g-' if id ==0 else 'b-'
        for i in range(4):
            ax.plot([hull[i][0], hull[(i+1)%4][0]], 
                    [hull[i][1], hull[(i+1)%4][1]], 
                    [hull[i][2], hull[(i+1)%4][2]], c)
            ax.plot([hull[i+4][0], hull[(i+1)%4 + 4][0]], 
                    [hull[i+4][1], hull[(i+1)%4 + 4][1]], 
                    [hull[i+4][2], hull[(i+1)%4 + 4][2]], c)
            ax.plot([hull[i][0], hull[i+4][0]], 
                    [hull[i][1], hull[i+4][1]], 
                    [hull[i][2], hull[i+4][2]], c)


    ax.scatter(*zip(*points))
    plt.show()

from argparse import ArgumentParser
import sys

DEBUG = False


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--fp', type=str, default="./data/two_cam/")
    parser.add_argument('--ext', type=str, default=".jpg")


    args = parser.parse_args(sys.argv[1:])


    mask_colors = ['B'] # TODO define a way of selecting a specific colour channel for an image with multiple detected humans

    # Projection Parameters
    # TODO : Maybe makle these arguments?

    zfar = 100.0
    znear = 0.01
    
    DEBUG = args.debug
    hulls = getTrapezoidVertices(args.fp, args.ext)

    

    getConvexHull(hulls)