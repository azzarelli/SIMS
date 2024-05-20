import pymeshlab
import numpy as np

# Define vertices for the first trapezoid
vertices_t1 = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # bottom face
    [0.2, 0.2, 1], [0.8, 0.2, 1], [0.8, 0.8, 1], [0.2, 0.8, 1]  # top face
], dtype=np.float64)

# Define faces for the first trapezoid (each face as triangles)
faces_t1 = np.array([
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], # sides
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]  # bottom and top faces
], dtype=np.int32)

# Define vertices for the second trapezoid
vertices_t2 = np.array([
    [0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0],   # bottom face
    [0.7, 0.7, 1], [1.3, 0.7, 1], [1.3, 1.3, 1], [0.7, 1.3, 1]  # top face
], dtype=np.float64)

# Define faces for the second trapezoid (each face as triangles)
faces_t2 = np.array([
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], # sides
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]  # bottom and top faces
], dtype=np.int32)

# Create a PyMeshLab mesh set
ms = pymeshlab.MeshSet()

# Add the first trapezoid to the mesh set
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices_t1, face_matrix=faces_t1))

# Add the second trapezoid to the mesh set
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices_t2, face_matrix=faces_t2))

# Perform boolean intersection
ms.apply_filter("compute_boolean_intersection", meshA=0, meshB=1)

# Extract the resulting intersection mesh
intersection_mesh = ms.current_mesh()

# Check if the intersection is valid and not empty
if intersection_mesh.vertex_matrix().size == 0:
    print("No intersection found.")
else:
    print("Intersection found with vertices:")
    print(intersection_mesh.vertex_matrix())

# Save the result to a file (optional)
ms.save_current_mesh("intersection_result.ply")
