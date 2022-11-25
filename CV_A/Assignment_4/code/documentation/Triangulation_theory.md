# Basic Method of behind estimating the rotation and traslation between 2 cameras

## triangulate3D

- Here we fix one camera (extrinsic matrix = Identity matrix). Now, we know correspondence points in image1 and image2.
- Therefore we can find the Fundamental matrix based on these point correspondences

Now, to estimate the 3D location of these points, we need 
1. Camera matrices (extrinsic*intrinsic) for both cameras M1 and M2
2. Image points (x,y) which correspond to each other
3. [Some theory](https://www.dropbox.com/sh/r569lhrgq9z4x7l/AACGDws-F4Krdwagm1F3-tnja?dl=0&preview=L17+-+Camera+Models%2C+Pose+Estimation+and+Triangulation.pdf)

The above theory is summarized as:
![](/documentation/images/triangulation_setup.png)

![](/documentation/images/triangulation_formula.png)

After finding the 3D points, we will reproject them back onto the image and compare them with our original correspondence points (which we either manually selected or got from some keypoint detector like ORB or BRIEF)

The formula for reprojection error in this case is:
$$
\operatorname{err}=\sum_i\left\|\mathbf{x}_{1 i}, \widehat{\mathbf{x}_{1 i}}\right\|^2+\left\|\mathbf{x}_{2 i}, \widehat{\mathbf{x}_{2 i}}\right\|^2
$$


## findM2
Previsously we saw that we need an M2 to triangulate, but we don't have an M2 yet :/.  \
However, since our first camera is fixed in 3D we can find the camera matrix M2 of our second camera as:

```python
def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s
```

**Note: The above function gives three possible values for M2**

![4 camera orientations](/documentation/images/4cameras.jpg)

Now there are 2 checks we can use to find which is the right camera:
- Determinant(Rotation component of M2) = 1 (so that the rotation belongs to SO(3))
- All Z values should be positive (i.e. the 3D point should be in front of both the cameras right?)

## Combining the above two functions

Now we have point correspondences, M1 and 4 M2's. Therefore we'll try to triangulate points based on  \
the correct criteria for camera orientations. Additionally we'll also try to minimize reprojection error:

```python
# iterate over M1(fixed) and M2(4 possibilites) by passing them to triangulate
    for i in range(M2.shape[2]):
        M2_current = M2[:,:,i]

        # build the C1 and C2:
        pts_in_3d, err = triangulate((K1 @ M1), pts1, (K2 @ M2_current), pts2)    
        if err < err_min and (np.where(pts_in_3d[:,2] < 0)[0].shape[0] == 0):
            print("satisfies the error criteria")
            err_min = err
            best_M2_i = i
            best_pts_3d = pts_in_3d

    if (best_M2_i is not None) and (best_pts_3d is not None):
        print("min err is", err_min)
        
        # return M2, C2, w(3d points), M1, C1
        return M2[:,:,best_M2_i], (K2 @ M2[:,:,best_M2_i]), best_pts_3d, M1, (K1 @ M1) # last entry is C1
```

**Finally we all together have our best_3d_points and the correct M2 matrix**

# Bundle Adjustment
We know that the error in the triangulation is basically difference between the projection of a 3D point and the actual point in 2D on the image. Now, we will move around the 3D points slightly and check in which orientation the reprojection error comes to a global minimum.
The formula for the above operation is shown below:

![](/documentation/images/Bundle_formula.png)

The process  we will follow now is very code specific. An explanation for only this below code is shown, where we will only be minimizing the rotation and translation (M2 matrix) error.

### High level procedure

1. Use the 2D point correspondences to find the Fundamental Matrix (along with RANSAC to find the inlier points)
2. Use the **inliers** to find our best **F** (fundamental matrix)
3. Compute an initial guess for M2 by using our old findM2 function
4. Now, the above function would have given us 3D points **(P_init)** and an **M2_init**
5. Now, we have compiled the following:
   - M1 and K1
   - M2_init and K2
   - F and E *(E = (K2.T @ F) @ K1)*

Having the above content, we will need to derive our reprojection error. We will do this in the RodriguesResidual function:

#### RodriguesResidual: *rodriguesResidual(x, K1, M1, p1, K2, p2)*

- **x** basically contains the translation and rotation of camera2. We can therefore get M2 from x
- We can find the camera matrices ***C1 = K1 @ M1***, ***C2 = K2 @ M2***

![](/documentation/images/generic_projection_eq.png)

- Use the above equation to get p1' and p2'
- Compare p1' and p1, p2' and p2, to get the reprojection error we need in both cameras

![](/documentation/images/reproj_error_residuals.png)

**Now we have a function which will give us reprojection error for a given M2 matrix. Now lets see how we'll use this reporjection error to optimize our M2**

### Optimization of M2

Now that we have a function which will give us reprojection error for any given M2, lets minimize this error by **moving around our 3D points slightly such that our reprojection error (for all points cumulative) reduces**

We do this using the scipy.optimize.minimize function

```python
# just some repackaging/preprocessing to give x to rodriguesResidual
x0 = P_init.flatten()
x0 = np.append(x0, r2_0.flatten())
x0 = np.append(x0, t2_0.flatten())

# optimization step
x_opt, _ = scipy.optimize.minimze(rodriguesResidual, x0, args=(K1, M1, p1, K2, p2))
```

**Finally our x_opt i.e x_optimal will have the correct rotation and translation of camera 2 and the corrected 3D points**