"""
load and preprocessing step for the depth map
hand detection codes are copied from PosePrior module
see https://cvarlab.icg.tugraz.at/projects/hand_detection
"""
import numpy as np, scipy.ndimage, cv2, math, struct, matplotlib, matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
from util import Camera

MSRA_size = namedtuple('MSRA_size', ['cols', 'rows', 'left', 'top', 'right', 'bottom'])

class DepthMap(object):
    # searcing depth range of hand
    max_depth, min_depth = 1500, 10
    # cropped_size
    size3 = [250, 250, 250]
    size2 = [128, 128]
    invariant_depth = float(size3[0])/float(size2[0])*Camera.focal_x

    def __init__(self, dataset, path):
        if dataset.upper() == 'ICVL':
            self.loadICVL(path)
        elif dataset.upper() == 'MSRA':
            self.loadMSRA(path)
        elif dataset.upper() == 'NYU':
            self.loadNYU(path)

    '''
    loading module
    '''
    def loadICVL(self, path):
        img = Image.open(path)
        if len(img.getbands()) != 1:
            raise ValueError('ICVL input should be with 1 channel')
        self.dmData = np.asarray(img,np.float32)
        return self.dmData

    def loadMSRA(self, path):
        f = open(path, 'rb')
        shape = [struct.unpack('i', f.read(4))[0] for i in range(6)]
        shape = MSRA_size(*shape)

        # initial data from MSRA is cropped
        cropDmData = np.fromfile(f, dtype=np.float32)

        crop_rows, crop_cols = shape.bottom - shape.top, shape.right - shape.left
        cropDmData = cropDmData.reshape(crop_rows, crop_cols)

        # expand the cropped dm to full-size make later process in a uniformed way
        self.dmData = np.zeros((shape.rows, shape.cols), np.float32)
        np.copyto(self.dmData[shape.top:shape.bottom, shape.left:shape.right], cropDmData)
        return self.dmData

    def loadNYU(self, path):
        img = Image.open(path)
        if len(img.getbands()) != 3:
            raise ValueError('NYU input should be with 3 channel')
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        self.dmData = np.asarray(dpt, np.float32)
        return self.dmData

    '''
    hand detector module
    '''
    def CoM(self, dpt):
        '''
        Calculate the center of mass
        :dpt: depthmap
        '''
        num = np.count_nonzero(dpt)
        xyc = scipy.ndimage.center_of_mass(dpt > 0)
        zc = dpt.sum()
        com = np.zeros((3), np.float)
        if num != 0:
            com[0], com[1], com[2] = xyc[1], xyc[0], zc/float(num)
        return com

    def Detector(self, com=None):
        '''
        First try to use the cv2.findContours to find mass and calculate the centroid according to the
        if not find contour with area>200, calculate the arbitrary contour, crop it using crop3D
        '''
        dpt = self.dmData.copy()
        dpt[dpt < self.min_depth] = 0
        dpt[dpt > self.max_depth] = 0

        # in the training case, we are given the groudtruth label of the palm/hand center
        if com is not None:
            return self.cropArea3D(dpt, com)

        # calculate the com based on the contour    
        steps = 20
        dz = (self.max_depth - self.min_depth)/float(steps)
        for i in range(steps):
            part = dpt.copy()
            part[part < i*dz + self.min_depth] = 0
            part[part > (i+1)*dz + self.min_depth] = 0
            part[part != 0] = 10  # binaralize
            ret, thresh = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype(dtype=np.uint8)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) > 200:
                    # centroid according to the contour
                    M = cv2.moments(contours[c])
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    # crop
                    xstart = int(max(cx-100, 0))
                    xend = int(min(cx+100, dpt.shape[1]))
                    ystart = int(max(cy-100, 0))
                    yend = int(min(cy+100, dpt.shape[0]))

                    cropped = dpt[ystart:yend, xstart:xend].copy()
                    cropped[cropped < i*dz + self.min_depth] = 0.
                    cropped[cropped > (i+1)*dz + self.min_depth] = 0.
                    com = self.CoM(cropped)
                    if np.allclose(com, 0.):
                        com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                    com[0] += xstart
                    com[1] += ystart

                    # refine iteratively
                    for k in range(5):
                        # calculate boundaries
                        zstart = com[2] - self.size3[2] / 2.
                        zend = com[2] + self.size3[2] / 2.
                        xstart  = int(max(math.floor((com[0]*com[2]/Camera.current.focal_x-self.size3[0]/2.)/com[2]*Camera.current.focal_x), 0.))
                        xend    = int(min(math.floor((com[0]*com[2]/Camera.current.focal_x+self.size3[0]/2.)/com[2]*Camera.current.focal_x), 
                                          dpt.shape[1]))
                        ystart  = int(max(math.floor((com[1]*com[2]/Camera.current.focal_y-self.size3[1]/2.)/com[2]*Camera.current.focal_y), 0.))
                        yend    = int(min(math.floor((com[1]*com[2]/Camera.current.focal_y+self.size3[1]/2.)/com[2]*Camera.current.focal_y), 
                                          dpt.shape[0]))

                        # crop
                        cropped = dpt[ystart:yend, xstart:xend].copy()
                        cropped[cropped < zstart] = 0.
                        cropped[cropped > zend] = 0.

                        com = self.CoM(cropped)
                        if np.allclose(com, 0.):
                            com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                        com[0] += xstart
                        com[1] += ystart

                    return self.cropArea3D(dpt, com)

        # if not find, use the camera center as the arbitrary center
        com = np.zeros((3), np.float)
        com[0], com[1], com[2] = Camera.current.center_x, Camera.current.center_y, 500.0
        return self.cropArea3D(dpt, com)

    def cropArea3D(self, dpt, com):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        if com is None:
            com = self.CoM(dpt)

        # 3D crop
        zstart = com[2] - self.size3[2] / 2.
        zend = com[2] + self.size3[2] / 2.
        xstart = int(math.floor((com[0] * com[2] / Camera.focal_x - self.size3[0] / 2.) / com[2]*Camera.focal_x))
        xend = int(math.floor((com[0] * com[2] / Camera.focal_x + self.size3[0] / 2.) / com[2]*Camera.focal_x))
        ystart = int(math.floor((com[1] * com[2] / Camera.focal_y - self.size3[1] / 2.) / com[2]*Camera.focal_y))
        yend = int(math.floor((com[1] * com[2] / Camera.focal_y + self.size3[1] / 2.) / com[2]*Camera.focal_y))
        cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
        cropped = np.pad(cropped,\
            ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, dpt.shape[0])), (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, dpt.shape[1]))),\
            mode='constant', constant_values=0)
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.  # backface is at 0, it is set later
        cropped[cropped != 0] -= (zstart-10)
        cropped /= self.size3[2]
        # crop offset
        trans = np.asmatrix(np.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        # depth resize
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (self.size2[0], hb * self.size2[0] / wb)
        else:
            sz = (wb * self.size2[1] / hb, self.size2[1])
        roi = cropped
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        scale = np.asmatrix(np.eye(3, dtype=float))
        if roi.shape[0] > roi.shape[1]:
            scale = scale * sz[1] / float(roi.shape[1])
        else:
            scale = scale * sz[0] / float(roi.shape[1])
        scale[2, 2] = 1

        # normalization
        ret = np.ones(self.size2, np.float32) * 0  # use background as filler
        xstart = int(math.floor(self.size2[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(self.size2[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.asmatrix(np.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off*scale*trans, Camera.to3D(com)

    def DetectCenter(self):
        '''
        First try to use the cv2.findContours to find mass and calculate the centroid according to the
        if not find contour with area>200, calculate the arbitrary contour, crop it using crop3D
        '''

        dpt = self.dmData.copy()
        min_depth = max(10, dpt.min())
        max_depth = min(1500, dpt.max())
        dpt[dpt < min_depth] = 0
        dpt[dpt > max_depth] = 0

        # figure = plt.figure()
        # ax = figure.add_subplot(1,4,1)
        # ax.imshow(dpt)
        # ax.set_title('input image')

        # calculate the com based on the contour    
        steps = 20
        dz = (self.max_depth - self.min_depth)/float(steps)
        for i in range(steps):
            part = dpt.copy()
            part[part < i*dz + self.min_depth] = 0
            part[part > (i+1)*dz + self.min_depth] = 0
            part[part != 0] = 10  # binaralize
            ret, thresh = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype(dtype=np.uint8)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) > 200:
                    coms = []
                    # centroid according to the contour
                    M = cv2.moments(contours[c])
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    # crop
                    xstart = int(max(cx-100, 0))
                    xend = int(min(cx+100, dpt.shape[1]))
                    ystart = int(max(cy-100, 0))
                    yend = int(min(cy+100, dpt.shape[0]))

                    cropped = dpt[ystart:yend, xstart:xend].copy()
                    cropped[cropped < i*dz + self.min_depth] = 0.
                    cropped[cropped > (i+1)*dz + self.min_depth] = 0.
                    com = self.CoM(cropped)
                    if np.allclose(com, 0.):
                        com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                    com[0] += xstart
                    com[1] += ystart
                    coms.append(com)


                    # refine iteratively
                    for k in range(5):
                        # calculate boundaries
                        zstart = com[2] - self.size3[2] / 2.
                        zend = com[2] + self.size3[2] / 2.
                        xstart  = int(max(math.floor((com[0]*com[2]/Camera.current.focal_x-self.size3[0]/2.)/com[2]*Camera.current.focal_x), 0.))
                        xend    = int(min(math.floor((com[0]*com[2]/Camera.current.focal_x+self.size3[0]/2.)/com[2]*Camera.current.focal_x), 
                                          dpt.shape[1]))
                        ystart  = int(max(math.floor((com[1]*com[2]/Camera.current.focal_y-self.size3[1]/2.)/com[2]*Camera.current.focal_y), 0.))
                        yend    = int(min(math.floor((com[1]*com[2]/Camera.current.focal_y+self.size3[1]/2.)/com[2]*Camera.current.focal_y), 
                                          dpt.shape[0]))

                        # crop
                        cropped = dpt[ystart:yend, xstart:xend].copy()
                        cropped[cropped < zstart] = 0.
                        cropped[cropped > zend] = 0.

                        com = self.CoM(cropped)
                        if np.allclose(com, 0.):
                            com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                        com[0] += xstart
                        com[1] += ystart
                        coms.append(com)

                    # ax = figure.add_subplot(1,4,2)
                    # ax.imshow(cropped)
                    # ax.scatter(coms[:][0], coms[:][1], c='r')
                    # ax.set_title('cropped image')
                    # plt.show()

                    return com

        # if not find, use the camera center as the arbitrary center
        com = np.zeros((3), np.float)
        com[0], com[1], com[2] = Camera.current.center_x, Camera.current.center_y, 500.0
        return com
