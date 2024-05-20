# Single (actually Double now) Image Man Seqmentation (SIMS)

(Working on it...) Using only two-cameras from a multi-view video set up, this looks to determine an morphologically dillated AABB box (in 3-D) containing the fore-ground target object (hu-man).

We do this by:

1. Segementng and masking the man in two views
2. Constructing a 2-D AABB on the image for the man (essentially a mask)
3. Projecting the 2-D AABB into 3-D (using the (nerfstudio library)[https://docs.nerf.studio/nerfology/model_components/visualize_cameras.html])
4. Finding the intersections of the two projections -> hence defining a volume which contains the target in-frustum human and other geometries from the sahre camera
5. Get the maximums and minimums of these points and construct a 3-D AABB with some dilation (not sure the value might be scene dependant)

Hopefully this could be a fesable solution to a static sparse 360 MVS, whereby we can contrain the in-frustum geometry to exist inside the final 3-D AABB, rather than over-training as it usually does...

