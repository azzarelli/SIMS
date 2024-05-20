import os
import json
import numpy as np
import math
import torch


from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis

import matplotlib
matplotlib.use('TkAgg')


def focal2fov(focal, pixels): # From 4DGS
    return 2*math.atan(pixels/(2*focal))

def getTrapezoidVertices(fp, ext, znear:float=0.1, zfar:float=20.):
    # Ensure path exists
    assert os.path.exists(fp+'transforms_train.json'), 'Could not find the transorms_train.json file'
    
    mask_fp = fp+'mask/'


    with open(fp+'transforms_train.json', 'r') as file:
        data = json.load(file)

    image_names = [dir for dir in os.listdir(fp) if ext in dir]

    mask_fps = [mask_fp+dir for dir in image_names]

    # image_fps = [fp+dir for dir in image_names]

    frame_data = data['frames']

    # selected_data = {
    #     'h':data['h'],
    #     'w':data['w'],
    #
    #     'fovx': focal2fov(data['fl_x'], data['w']),
    #     'fovy': focal2fov(data['fl_y'], data['h'])
    # }
    
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

    if DEBUG:
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

            XY = np.array([
                [xstart, ystop],
                [xstart, ystart],
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

          
        
            hulls.append((top_vertices, base_vertcies, ray.directions))

    if DEBUG:
        plt.show()


    return hulls

from sympy import Plane, Line3D, Point3D

def ray_plane_intersection(ray_origin, ray_direction, plane_normal, point_on_plane):
    # Check if the ray is parallel to the plane
    denom = np.dot(plane_normal, ray_direction)
    if np.abs(denom) < 1e-6:
        return None  # No intersection, the ray is parallel to the plane
    
    # Calculate the intersection point
    t = np.dot(plane_normal, point_on_plane - ray_origin) / denom
    if t <= 0.001:
        return None  # The intersection is behind the ray origin
    
    intersection_point = ray_origin + t * ray_direction
    return intersection_point

def getHullIntersection(camdata):
    """

    Notes:
        We need three points (xyz) to define a plane. And we construct 4 planes: the Top, left, bottom, right planes
        The input is a list with a dictionary containing each top and base vertices. The top vertices are the near-frustum camera vertices,
        in the order of: top-left, bottom-left, bottom-right, top-right  
    """
      # Construct a line for each point
    # The order is top-left anti-clockwise to top-right

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def ray_plane_intersection(ray_origin, ray_direction, plane_normal, point_on_plane):
        """
        Notes:
            Explanation available here: https://math.stackexchange.com/questions/2934157/how-to-calculate-the-intersection-point-between-a-straight-and-a-triangle-in-py

            1. We determine intersection of ray with plane
            2. We determine whether intersection point is inside triangle
        """

        # Check if the ray is parallel to the plane
        denom = np.dot(ray_direction, plane_normal)

        # Very unlikely...
        if np.abs(denom) < 1e-11:
            return None  # No intersection, the ray is parallel to the plane

        # t = -(o-p).n / d.n
        t = - np.dot(ray_origin - point_on_plane, plane_normal) / denom

        # If t is behind the origin
        if t < 0:
            return None  # The intersection is behind the ray origin

        intersection_point = ray_origin + t * ray_direction

        return intersection_point
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    planes = []
    rays = []
    cols = ['b', 'g']
    for j, data in enumerate(camdata):

        for i, (top, bottom, direct) in enumerate(zip(*data)):
            ax.plot([top[0], bottom[0]], [top[1], bottom[1]], [top[2], bottom[2]], c=cols[j])

            if j == 0:
                # line = Line3D(top.tolist(), direction_ratio=direct.tolist())
                rays.append([top.numpy(), direct.numpy()]) 

            else:
                v0, v1, v2 = top, data[1][(i+1)%4], bottom
                # Compute the normal of the plane defined by the triangle (v0, v1, v2)
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)

                plane_data = {
                    'normal': normal,
                    'point': v1,  # if we split the polygon into triangles, this is a common point shared by triangles
                    'vertices': [top, bottom, data[0][(i + 1) % 4], data[1][(i + 1) % 4]],
                    'triangles': [
                        [top, bottom, data[0][(i + 1) % 4]],
                        [bottom, data[0][(i + 1) % 4], data[1][(i + 1) % 4]]
                    ]

                }
                planes.append(plane_data)

                # planes.append([normal, v1])


                poly3d = [[top.tolist(), data[1][(i+1)%4].tolist(), bottom.tolist()]]
                poly = Poly3DCollection(poly3d, alpha=0.5, facecolors='cyan', linewidths=1, edgecolors='r')
                ax.add_collection3d(poly)


    intersections = []
    for ray in rays:
        for plane in planes:

            normal = plane['normal']
            plane_point = plane['point']


            inter = ray_plane_intersection(ray[0], ray[1], normal, plane_point)

            # If inter is None then the ray is either parallel or intersection occurs behind the camera
            if inter is not None:
                # Now we need to determine if the point lies inside our polygon

                intersections.append(inter)


    ax.scatter(*zip(*intersections), 'r')
    plt.show()

    # Then lets 
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

    getHullIntersection(hulls)