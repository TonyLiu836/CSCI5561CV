import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

def find_match(img1, img2):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    neighbor = NearestNeighbors()
    #neighbor.fit(descriptors1)
    #neighDist, neighInd = neighbor.kneighbors(descriptors2, n_neighbors = 2, return_distance = True)
    neighbor.fit(descriptors2)
    neighDist, neighInd = neighbor.kneighbors(descriptors1, n_neighbors = 2, return_distance = True)

    x1 = []
    x2 = []

    for index in range(len(neighDist)):
        if (neighDist[index,0] / neighDist[index,1]) < 0.7:
            x1.append(keypoints1[index].pt)
            x2.append(keypoints2[neighInd[index,0]].pt)

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    maxInliers = 0
    bestAffine = 0
    for itera in range(ransac_iter):
        rand = np.random.choice(np.shape(x1)[0], 3, replace=False)

        x = np.array([[x1[rand[0],0], x1[rand[1],0], x1[rand[2],0]],
                      [x1[rand[0],1], x1[rand[1],1], x1[rand[2],1]],
                      [1,1,1]])

        xPrime = np.array([[x2[rand[0],0], x2[rand[1],0], x2[rand[2],0]],
                           [x2[rand[0],1], x2[rand[1],1], x2[rand[2],1]]])
        
        affineParams = np.zeros((2,3))
        xInv = np.linalg.pinv(x)
        affineParams = np.dot(xPrime, xInv)
        affineTrans = np.vstack((affineParams, np.array([0,0,1])))
        inliers = 0
        
        for pt in range(np.shape(x1)[0]):
            x1Vect = np.concatenate((np.array(x1[pt]), np.array([1])))
            x1Vect = np.transpose([x1Vect])
            estim = np.dot(affineTrans, x1Vect)
            x2Vect = np.concatenate((np.array(x2[pt]), np.array([1])))
            x2Vect = np.transpose([x2Vect])
            error = np.sqrt(np.sum((x2Vect - estim)**2))
            if error <= ransac_thr:
                inliers += 1
        
        if inliers > maxInliers:
            maxInliers = inliers
            bestAffine = affineTrans

    A = bestAffine
    return A


def warp_image(img, A, output_size):
    # To do
    img_warped = np.zeros(output_size)

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            targetPt= np.array([j,i,1])
            sourcePt = np.floor(np.dot(A,targetPt))
            sourcePtY = sourcePt[0]
            sourcePtX = sourcePt[1]
            img_warped[i,j] = img[np.int(sourcePtX), np.int(sourcePtY)]
    return img_warped


def get_differential_filter():
    # To do
    filter_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])   #sobel x
    filter_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])   #sobel y
    
    return filter_x, filter_y

def filter_image(im, filter):
    # To do
    im_filtered = np.zeros(np.shape(im))
    im_padded = np.pad(im, ((1,1),(1,1)), 'constant')
    filter_size = np.shape(filter)[0]       #assuming only square filters     
    
    m = np.shape(im_padded)[0]
    n = np.shape(im_padded)[1]

    for i in range(m-filter_size+1):        #sliding window to filter image
        for j in range(n-filter_size+1):
            im_filtered[i,j] = np.sum(np.multiply(im_padded[i:i+filter_size, j:j+filter_size], filter))
    
    return im_filtered


def findSteepestDescent(template, templateDX, templateDY):
    '''
    jac= 2x6
    dTemplate = 452x292 for both dx and dy combined
    steepestDesc = dTemplate * jac 
    for each (u,v) in steepestestDesc, 
        grab both the dx and dy value to form [dx, dy]
        multiply [dx,dy] with jac[u,v] to get 1x6 vect
        store 1x6 vect at steepestDesc[u,v]
    '''
    steepestDescent = np.zeros((np.shape(template)[0],np.shape(template)[1],6))
    
    for u in range(np.shape(template)[0]):
        for v in range(np.shape(template)[1]):
            
            Jacobian = np.array([[v, u, 1, 0, 0, 0],
                                 [0, 0, 0, v, u, 1]])
            
            gradPt = np.array([templateDX[u,v], templateDY[u,v]])
            steepest = np.array(np.dot(gradPt, Jacobian))
            steepestDescent[u,v,:] = steepest
    return steepestDescent

    
def align_image(template, target, A):
    # To do

    filterX, filterY = get_differential_filter()   
    templateDY = filter_image(template, filterY)
    templateDX = filter_image(template, filterX)

    hessian = np.zeros((6,6))
    steepestDescentImg = findSteepestDescent(template,templateDX, templateDY)
    
    for u in range(np.shape(template)[0]):
        for v in range(np.shape(template)[1]):
            steepestDes = steepestDescentImg[u,v,:]
            steepestDesTrans = np.transpose([steepestDes])
            hessian += np.dot(steepestDesTrans, [steepestDes])

    invHessian = np.linalg.pinv(hessian)
    errors = []
    affine = A
    
    #while(np.linalg.norm(p) > 1.733):        
    for a in range(150):
        targetWarped = warp_image(target, affine, [np.shape(template)[0], np.shape(template)[1]])
        imgError = targetWarped - template
        errorVal = np.linalg.norm(imgError)
        errors.append(errorVal)
        F = np.zeros((6,1))
        
        for x in range(np.shape(template)[0]):
            for y in range(np.shape(template)[1]):
                steepestDes = steepestDescentImg[x,y,:]
                steepestDesTrans = np.transpose([steepestDes])
                F += steepestDesTrans * imgError[x,y]

        deltaP = np.dot(invHessian, F)
        deltaP = np.transpose(deltaP) + np.array([1,0,0,0,1,0])
        affineDeltaP = np.vstack((np.array(np.reshape(deltaP,(2,3))), np.array([0,0,1])))
        invWdeltaP = np.linalg.pinv(affineDeltaP)
        affine = np.dot(affine,invWdeltaP)
    
    A_refined = affine
    
    return A_refined, np.asarray(errors)

def track_multi_frames(template, img_list):
    # To do
    A_list = []
    ransac_thr = 1
    ransac_iter = 125
    updatedTemplate = template

    for i in range(len(img_list)):
        x1, x2 = find_match(updatedTemplate, img_list[i])
        A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
        A_refined, errors = align_image(updatedTemplate, img_list[i], A)
        A_list.append(A_refined)
        updatedTemplate = warp_image(img_list[i], A_refined, np.shape(updatedTemplate))
        updatedTemplate = cv2.normalize(updatedTemplate, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #not normalizing updatedTemplate gives error from CV2
    
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    #visualize_find_match(template, target_list[0], x1, x2)
    
    ransac_thr = 1
    ransac_iter = 125
    
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    img_warped = warp_image(target_list[0], A, template.shape)
    img_diff = np.abs(template - img_warped)
    error = np.sqrt(np.sum(img_diff ** 2))
    #plt.imshow(template, cmap='gray', vmin=0, vmax=255)
    #plt.axis('off')
    #plt.show()
    
    #plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    #plt.axis('off')
    #plt.show()
    #plt.imshow(img_diff, cmap='jet')
    #plt.axis('off')
    #plt.show()
    
    
    A_refined, errors = align_image(template, target_list[0], A)
    #visualize_align_image(template, target_list[0], A, A_refined, errors)
    
    #A_list = track_multi_frames(template, target_list)
    #visualize_track_multi_frames(template, target_list, A_list)
    

