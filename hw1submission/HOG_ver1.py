
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def get_gradient(im_dx, im_dy):
    # To do
    m = np.shape(im_dx)[0]
    n = np.shape(im_dy)[1]
    
    grad_angle = np.zeros(((m,n)))
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    
    for i in range(m):
        for j in range(n):
            angle = np.arctan2(im_dy[i,j], im_dx[i,j])
            if angle < 0:
                grad_angle[i,j] = angle + np.pi
            else:
                grad_angle[i,j] = angle
    grad_angle = np.degrees(grad_angle)
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    m = np.shape(grad_mag)[0]
    n = np.shape(grad_mag)[1]
    
    M = m // cell_size
    N = n // cell_size

    ori_histo = np.zeros((M,N,6))    

    for i in range (M):
        for j in range(N):
            for x in range(cell_size):
                for y in range(cell_size):
                    indexM = cell_size*i + x
                    indexN = cell_size*j + y
                    
                    if (grad_angle[indexM,indexN] >= 165 and grad_angle[indexM, indexN] < 180) or (grad_angle[indexM,indexN] >= 0 and grad_angle[indexM, indexN] < 15):
                        ori_histo[i,j,0] += grad_mag[indexM, indexN]
                    
                    elif grad_angle[indexM,indexN] >= 15 and grad_angle[indexM,indexN] < 45:
                        ori_histo[i,j,1] += grad_mag[indexM, indexN]
                    
                    elif grad_angle[indexM,indexN] >= 45 and grad_angle[indexM,indexN] < 75:
                        ori_histo[i,j,2] += grad_mag[indexM, indexN]
                        
                    elif grad_angle[indexM,indexN] >= 75 and grad_angle[indexM,indexN] < 105:
                        ori_histo[i,j,3] += grad_mag[indexM, indexN]
                        
                    elif grad_angle[indexM,indexN] >= 105 and grad_angle[indexM,indexN] < 135:
                        ori_histo[i,j,4] += grad_mag[indexM, indexN]
                        
                    elif grad_angle[indexM,indexN] >= 135 and grad_angle[indexM,indexN] < 165:
                        ori_histo[i,j,5] += grad_mag[indexM, indexN]
   
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    M = np.shape(ori_histo)[0]
    N = np.shape(ori_histo)[1]
    e = 0.001
    ori_histo_normalized = np.zeros((M-(block_size-1), N-(block_size-1), 6*block_size**2))

    for i in range(M - block_size+1):
        for j in range(N - block_size+1):
            h = np.zeros((1,6))
            histMag = np.sqrt(np.sum(ori_histo[i:i+block_size, j:j+block_size,:] ** 2 + e**2))
            targetCell = ori_histo[i:i+block_size,j:j+block_size,:]
            h = targetCell.flatten('C')       #flatten replaces 2 for loops needed to convert matrice to vector
            '''
            for m in range(block_size):
                for n in range(block_size):            
                    #print("h",h)
                    #print("shape of ori histo in loop=",np.shape([ori_histo[i+m][j+n][:]]))
                    h = np.concatenate([h, [ori_histo[i+m][j+n][:]]],-1)
                    #h = np.dstack((h, ori_histo[i+m][j+n][:]))                                             
                    #print("shape of h=",np.shape(h))
            
            h = h[0,6:]                                        
            '''
            ori_histo_normalized[i][j][:] = h / histMag
    
    ori_histo_normalized = ori_histo_normalized.flatten('C')
    
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    im /= im.max()
    filterX, filterY = get_differential_filter()        
    
    imageDX = filter_image(im, filterX)
    imageDY = filter_image(im, filterY)

    grad_mag, grad_angle = get_gradient(imageDX, imageDY)
    
    block_size = 2
    cell_size = 8 
    
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    
    hog = get_block_descriptor(ori_histo, block_size)
    # visualize to verify
    
    visualize_hog(im, hog, cell_size, block_size)
    
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    bounding_boxes = []
    stride = 9              #sliding window with stride of 1 is too slow

    I_templateH = box_size = np.shape(I_template)[0]
    I_templateW = np.shape(I_template)[1]
    
    templateHog = extract_hog(I_template)
    normTemplateHogVect = (templateHog - np.mean(templateHog))

    I_targetH = np.shape(I_target)[0]
    I_targetW = np.shape(I_target)[1]
    BBinitial = []
    
    for i in range(0, (I_targetH - I_templateH+1), stride):
        for j in range(0, (I_targetW - I_templateW+1), stride):
            targetCell = extract_hog(I_target[i:i+I_templateH, j:j+I_templateW])
            normTargetCellVect = (targetCell - np.mean(targetCell))
            s = np.dot(normTemplateHogVect, normTargetCellVect) / (np.linalg.norm(normTemplateHogVect) * np.linalg.norm(normTargetCellVect))
            if s > 0.4:
                BBinitial.append((j,i,s))
            
    BBsorted = sorted(BBinitial, key=lambda BBinitial:BBinitial[2], reverse = False)  

    bounding_boxes = NMS(BBsorted, box_size)

    return bounding_boxes


def NMS(BBsorted, box_size):
    boundingBoxes = []
    while len(BBsorted) > 0:                    
        currentBB = BBsorted.pop(-1)
        boundingBoxes.append(currentBB)
        for index in reversed(range(len(BBsorted))):                #go from back of array to avoid skipping box after deleting index
            IOU = find_IOU(currentBB, BBsorted[index], box_size)
            if IOU > 0.5:
                BBsorted.remove(BBsorted[index])
    boundingBoxes = np.asarray(boundingBoxes)
    return boundingBoxes
        
        
def find_IOU(BB1, BB2, box_size):
    
    BB1x1 = BB1[0]
    BB1x2 = BB1[0] + box_size
    BB1y1 = BB1[1]
    BB1y2 = BB1[1] + box_size
    
    BB2x1 = BB2[0]
    BB2x2 = BB2[0] + box_size
    BB2y1 = BB2[1]
    BB2y2 = BB2[1] + box_size

    area1 = abs(BB1x2 - BB1x1) * abs(BB1y2- BB1y1)
    area2 = abs(BB2x2 - BB2x1) * abs(BB2y2 - BB2y1)
    
    intersectWidth = min(abs(BB1x2-BB2x1), abs(BB1x1-BB2x2))
    intersectHeight = min(abs(BB1y2-BB2y1), abs(BB1y1-BB2y2))

    if intersectWidth > box_size or intersectHeight > box_size: #for when bounding boxes dont overlap
        intersectArea = 0
    else:    
        intersectArea = intersectWidth * intersectHeight
    
    IOU = intersectArea / (area1 + area2 - intersectArea)

    #print("area1 = ", area1, "   area2=", area2, "    intersection area=", intersectArea, '   total area =', (area1 + area2 - intersectArea), "    IOU=", IOU)
    return IOU
    

def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)
    
    I_target= cv2.imread('target.png', 0)                               
    #MxN image
    I_template = cv2.imread('template.png', 0)      
    bounding_boxes=face_recognition(I_target, I_template)                  
    I_target_c= cv2.imread('target.png')   

    #visualize_face_detection(I_target_c, bounding_boxes, template.shape[0])     #********* original code (gives error about parameter "template.shape[0]")
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])           #modified function call
    



