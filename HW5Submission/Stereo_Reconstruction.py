import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def find_match(img1, img2):
    # TO DO
    
    sift1 = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift1.detectAndCompute(img1, None)
    
    sift2 = cv2.xfeatures2d.SIFT_create()
    keypoints2, descriptors2 = sift2.detectAndCompute(img2, None)

    neighbor1 = NearestNeighbors()
    neighbor1.fit(descriptors2)
    neighDist1, neighInd1 = neighbor1.kneighbors(descriptors1, n_neighbors = 2, return_distance = True)

    x1LR = []
    x2LR = []
    
    for index in range(len(neighDist1)):
        if (neighDist1[index,0] / neighDist1[index,1]) < 0.7:
            x1LR.append(keypoints1[index].pt)
            x2LR.append(keypoints2[neighInd1[index,0]].pt)
    x1LR = np.asarray(x1LR)
    x2LR = np.asarray(x2LR)

    neighbor2 = NearestNeighbors()
    neighbor2.fit(descriptors1)
    neighDist2, neighInd2 = neighbor2.kneighbors(descriptors2, n_neighbors = 2, return_distance = True)
    
    x1RL = []
    x2RL = []
    
    for index in range(len(neighDist2)):
        if (neighDist2[index,0] / neighDist2[index,1]) < 0.6:
            x1RL.append(keypoints2[index].pt)
            x2RL.append(keypoints1[neighInd2[index,0]].pt)

    x1RL = np.asarray(x1RL)
    x2RL = np.asarray(x2RL)

    pts1 = []
    pts2 = []

    for i in range(len(x1LR)): 
        for j in range(len(x2RL)):
            if (x1LR[i][0] == x2RL[j][0] and x1LR[i][1] == x2RL[j][1]):                          
                if (x2LR[i][0] == x1RL[j][0] and x2LR[i][1] == x1RL[j][1]):    
                    pts1.append(x1LR[i])
                    pts2.append(x2LR[i])

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    return pts1, pts2



def compute_F(pts1, pts2):
    # TO DO
    RANSACiters = 750          
    iters = 0
    ones = [[1]] * 8
    maxInliers = 0
    
    RANSACthr = 0.03
    
    while iters < RANSACiters:
        iters += 1
        randomSamples = np.random.choice(len(pts1), 8, replace = False)
        randPts1 = np.append(pts1[randomSamples,:], ones, axis = 1)
        randPts2 = np.append(pts2[randomSamples,:], ones, axis = 1)
        A = []
        
        for i in range(len(randomSamples)):
            point = [randPts2[i][0] * randPts1[i][0], randPts2[i][0] * randPts1[i][1], randPts2[i][0], randPts2[i][1] * randPts1[i][0], randPts2[i][1] * randPts1[i][1], randPts2[i][1],  randPts1[i][0],randPts1[i][1], 1 ]
            A.append(point)
        A = np.asarray(A)
        u, s, vT = np.linalg.svd(A)
        v = vT.T
        f = v[:,-1].reshape((3,3))
                
        U,D,VT = np.linalg.svd(f)
        
        D[-1] = 0
        D = np.diag(D)
        Finitial = np.dot(np.dot(U, D), VT)
        inliers = 0
        
        
        for j in range(len(pts1)):
            p1 = np.asarray([pts1[j][0], pts1[j][1], 1])
            p2 = np.asarray([pts2[j][0], pts2[j][1], 1])
            p1 = np.transpose([p1])
            
            pointError = abs(np.dot(np.matmul(p2, Finitial), p1))

            if pointError < RANSACthr:
                inliers += 1
          
        if inliers > maxInliers:
            maxInliers = inliers
            F = Finitial    
    return F


def skew_symm_matrix(pts):
    a,b,c = pts
    matrix = np.array([[0, -c, b],[c, 0, -a], [-b, a, 0]])
    return matrix


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    
    pts3D = []
    for i in range(len(pts1)):

        point1 = np.hstack((pts1[i,:], np.array([1])))
        point2 = np.hstack((pts2[i,:], np.array([1])))
        
        point1skew = skew_symm_matrix(point1)
        point2skew = skew_symm_matrix(point2)
        
        crossP1 = np.dot(point1skew,P1) 
        crossP2 = np.dot(point2skew,P2) 
        
        A = np.vstack((crossP1[0:2,:], crossP2[0:2,:]))
        u,s,vT = np.linalg.svd(A)
        v = vT.T
        pt = v[:,-1]

        ptScaled = pt[:3] / pt[-1]
        pts3D.append(ptScaled)
    pts3D = np.asarray(pts3D)
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    #print("Rs shape", np.shape(Rs))         #4x3x3
    #print("Cs shape", np.shape(Cs))         #4x3x1
    #print('pts3D shape', np.shape(pts3Ds))  #4x292x3
    R,C,pts3D = 0,0,0
    bestNValid = 0
    
    for r in range(np.shape(Rs)[0]):
        nValid = 0
        rotation = np.asarray(Rs[r])
        rotZ = rotation[2:]
        center = np.asarray(Cs[r])
        center = np.reshape(center,-1)

        for i in range(np.shape(pts3Ds)[1]):
            z = np.dot((pts3Ds[r][i,:] - center),  np.transpose([rotZ[0]]))
            if z > 0:
                nValid += 1
        
        if nValid > bestNValid:
            bestNValid = nValid
            bestInd = r  
    
    R = Rs[bestInd]
    C = Cs[bestInd]
    pts3D = pts3Ds[bestInd]
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    #print("K shape", np.shape(K))           #3x3
    #print("R shape", np.shape(R))           #3x3
    #print('C shape', np.shape(C))           #3x1
    
    RX = C / np.linalg.norm(C)
    RX = np.reshape(RX, (1,3))
    RX = RX[0]
    rztilde = np.array([0,0,1])
    RZnumerator = rztilde - RX*np.dot(rztilde, np.transpose([RX]))
    RZdenominator = np.linalg.norm(RZnumerator)
    RZ = RZnumerator / RZdenominator
    RY = np.cross(RZ, RX)
    Rrect = np.vstack((RX,RY,RZ))
    
    H1 = np.dot(np.dot(K,Rrect), np.linalg.inv(K))
    H2 = np.dot(np.dot(np.dot(K,Rrect), np.transpose(R)),np.linalg.inv(K))
    return H1, H2



def dense_match(img1, img2):
    # TO DO
    sift = cv2.xfeatures2d.SIFT_create()
    stride = 1
    size = 3
    keypoints1 = [cv2.KeyPoint(x, y, size) for y in range(0, img1.shape[0], stride) 
                                         for x in range(0, img1.shape[1], stride)]
    kps1,denseFeature1 = sift.compute(img1, keypoints1)
    
    denseFeature1 = np.reshape(denseFeature1, (np.shape(img1)[0], np.shape(img1)[1], 128))
    
    keypoints2 = [cv2.KeyPoint(x, y, size) for y in range(0, img2.shape[0], stride) 
                                         for x in range(0, img2.shape[1], stride)]
    kps2,denseFeature2 = sift.compute(img2, keypoints2)
    
    denseFeature2 = np.reshape(denseFeature2, (np.shape(img2)[0], np.shape(img2)[1], 128))
    
    disparity = np.zeros(np.shape(img1))

    for v in range(np.shape(img1)[0]):
        for u in range(np.shape(img1)[1]):  
            if img1[v,u] != 0:
                diff = []
                denseFeat1 = denseFeature1[v, u]
        
                for w in range(np.shape(img1)[1]):
                    denseFeat2 = denseFeature2[v, w]
                    diff.append(np.linalg.norm(denseFeat1 - denseFeat2))
                
                minDiff = np.argmin(diff)
                disparity[v,u] = np.abs(minDiff - u)
 
    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.2, markersize=3)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    #visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    #visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    #visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)
    
    
    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    #visualize_camera_poses(Rs, Cs)
    
    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    #visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)
    
    
    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)
    
    
    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    #visualize_img_pair(img_left_w, img_right_w)
    
    
    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    #visualize_disparity_map(disparity)
    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
    