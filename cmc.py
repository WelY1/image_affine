'''
计算当前帧与参考帧（初始帧）的位姿变换
'''

import cv2
import numpy as np
import time
import copy
from tools import metric
import sys
import matplotlib.pyplot as plt

class GMC:
    def __init__(self, method='orb', downscale=1.):
        self.method = method
        if self.method == 'orb':
            self.disth = 0.7
            self.detector = cv2.FastFeatureDetector_create(20)   # FAST特征检测
            self.extractor = cv2.ORB_create()                    # ORB特征检测 用于生成ORB描述子
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)       # 汉明距离，适用于二进制描述子，如ORB描述子。

        elif self.method == 'sift':
            self.disth = 0.7
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            self.maxLevel = 4
            number_of_iterations = 1000
            termination_eps = 1e-5
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

            self.disth = 0.7
            self.detector = cv2.FastFeatureDetector_create(20)  # FAST特征检测
            self.extractor = cv2.ORB_create()  # ORB特征检测 用于生成ORB描述子
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # 汉明距离，适用于二进制描述子，如ORB描述子。
        
        elif self.method == 'OptFlow':
            '''
            # 定义 Shi-Tomasi 角点检测器的参数
            maxCorners：表示要检测的角点数量的最大值。默认为100
            qualityLevel：表示角点检测的质量水平。较高的值会得到较好的角点，但数量会减少。范围为0到1，默认值为0.01
            minDistance：表示检测到的角点之间的最小距离。默认为10个像素点
            blockSize：表示在角点检测中使用的窗口大小。较大的值可以检测到较大的角点，但是对于较小的角点则无法检测到。默认值为3
            useHarrisDetector：是否选择Harris角点检测算法，若为False则使用Shi-Tomasi角点检测算法，一般来说效果更好
            k：计算 Harris 角点响应函数时使用的自由参数，较小的值检测到的特征点数量少，质量高。默认值为0.04
            '''
            self.feature_params = dict(maxCorners=2000, qualityLevel=0.3, minDistance=10, blockSize=5,
                                       useHarrisDetector=False, k=0.06)
            
            '''
            # 控制光流计算过程的参数
            winSize：表示窗口的大小，它是一个二元组 (width, height)，用于指定光流算法在每一层金字塔图像上的搜索范围，通常情况下取小于图片尺寸的奇数值
            maxLevel：表示金字塔的最大层数，它是一个整数值，用于指定在多分辨率金字塔中计算光流时的最大层数，通常情况下取值在 2-4 之间
            criteria：表示迭代算法的停止准则，(type, maxCount, epsilon)，其中 type 表示停止准则类型，可以为 cv2.TERM_CRITERIA_EPS（表示通过迭代误差进行停止）
                        maxCount 表示迭代的最大次数，epsilon 表示迭代的误差容限
            '''
            self.lk_params = dict(winSize=(20,20), maxLevel=4, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        
        self.downscale = downscale
        self.preFrame = None
        self.preKeyPoints = None
        self.preDescriptors = None
        self.prePyramid = None
        self.FirstFrame = True
        
    def apply(self, src):
        if self.method == 'orb':
            return self.applyFeaures(src)
        elif self.method == 'sift':
            return self.applyFeaures(src)
        elif self.method == 'ecc':
            return self.applyEcc_v3(src)
            # return self.applyEcc(src)
        elif self.method == 'OptFlow':
            return self.applySparaseOptFlow(src)

    def applyEcc_v3(self, src):
        """Compute the warp matrix from src to dst.
            利用高斯金字塔加速ecc计算,初始值用orb求解
        """
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity

        H = np.eye(2, 3, dtype=np.float32)
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        height, width = src.shape[0], src.shape[1]
        # make the imgs smaller to speed up
        if self.downscale > 1.0:
            # src = cv2.GaussianBlur(src, (3, 3), 1.5)
            src = cv2.resize(src, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        img_pyr = [src]
        for i in range(self.maxLevel - 1):
            img_pyr.append(cv2.pyrDown(img_pyr[-1]))  # 降采样，宽高默认为原图的1/2

        # handle first frame
        if self.FirstFrame:
            self.preFrame = src.copy()
            self.prePyramid = copy.copy(img_pyr)
            self.FirstFrame = False
            return H

        H = self.applyFeaures(src).astype('float32')
        # print(H)
        for i in reversed(range(self.maxLevel)):
            # print(i)
            H_level = H.copy()
            src = img_pyr[i]
            preframe = self.prePyramid[i]
            # Run the ECC algorithm. The results are stored in H.
            (cc, H_level) = cv2.findTransformECC(src, preframe, H_level, self.warp_mode, self.criteria, None, 1)

            H_level[:2, 2] *= 2
            H = H_level.copy()

        H[:2, 2] *= self.downscale / 2
        warp_matrix = H
        return warp_matrix

    def applyEcc_v2(self, src):
        """Compute the warp matrix from src to dst.
        利用高斯金字塔加速ecc计算
        """
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity

        H = np.eye(2, 3, dtype=np.float32)
            
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        height, width = src.shape[0], src.shape[1]
        # make the imgs smaller to speed up
        if self.downscale > 1.0:
            # src = cv2.GaussianBlur(src, (3, 3), 1.5)
            src = cv2.resize(src, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
            
        img_pyr = [src]
        for i in range(self.maxLevel-1):
            img_pyr.append(cv2.pyrDown(img_pyr[-1]))       # 降采样，宽高默认为原图的1/2
                  
        # handle first frame
        if self.FirstFrame:
            self.preFrame = src.copy()
            self.prePyramid = copy.copy(img_pyr)
            self.FirstFrame = False
            return H
          
        for i in reversed(range(self.maxLevel)):
            # print(i)
            H_level = H.copy()
            
            src = img_pyr[i]
            preframe = self.prePyramid[i]
            # Run the ECC algorithm. The results are stored in H.
            (cc, H_level) = cv2.findTransformECC(src, preframe, H_level, self.warp_mode, self.criteria, None, 1)
            
            H_level[:2, 2] *= 2
            H = H_level.copy()
            
        H[:2, 2] *= self.downscale / 2
        warp_matrix = H     
        return warp_matrix
    
    def applyEcc(self, src):
        """Compute the warp matrix from src to dst.
        """
        height, width = src.shape[0], src.shape[1]

        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
     
        # make the imgs smaller to speed up
        if self.downscale > 1.0:
            # src = cv2.GaussianBlur(src, (3, 3), 1.5)
            src = cv2.resize(src, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
     
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # handle first frame
        if self.FirstFrame:
            self.preFrame = src.copy()
            self.FirstFrame = False
            return warp_matrix
          
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(src, self.preFrame, warp_matrix, self.warp_mode, self.criteria, None, 1)
     
        if self.downscale > 1.0:
            warp_matrix[0, 2] = warp_matrix[0, 2] * self.downscale
            warp_matrix[1, 2] = warp_matrix[1, 2] * self.downscale
        
        return warp_matrix
            
    def applyFeaures(self,src, plot=False):
        height, width = src.shape[0], src.shape[1]
        warp_matrix = np.eye(2, 3)

        # Convert images to grayscale
        if len(src.shape)>2 and src.shape[-1]>1:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            
        if self.downscale > 1.0:
            # src = cv2.GaussianBlur(src, (3, 3), 1.5)
            src = cv2.resize(src, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
        
        # find the keypoints
        mask = np.zeros_like(src)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        # if detections is not None:
        #     for det in detections:
        #         tlbr = (det[:4] / self.self.downscale).astype(np.int_)
        #         mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0
                
        keypoints = self.detector.detect(src, mask)               # FAST检测生成关键点
        keypoints, descriptors = self.extractor.compute(src, keypoints)   # ORB检测生成关键点和描述子
        '''
        # keypoints: [<class KeyPoint>]
        KeyPoint.pt  特征点在图像中的坐标
        KeyPoint.size  特征点的直径大小
        KeyPoint.angle  特征点的方向
        KeyPoint.response  特征点的响应强度
        KeyPoint.octave  特征点所在金字塔组和层数
        KeyPoint.class_id  特征点的类别标识
        
        # descriptors: (N, 32)
        '''
        
        if self.FirstFrame:
            self.preFrame = src.copy()
            self.preKeyPoints = copy.copy(keypoints)
            self.preDescriptors = copy.copy(descriptors)
            self.FirstFrame = False
            
            return warp_matrix
        
        knnMatches = self.matcher.knnMatch(self.preDescriptors, descriptors, 2)  # 选择k个最近的特征点，或者用radiusMatch，选择距离小于r的最近的一个点
        # print(knnMatches)
        '''
        # [<class DMatch>]
        DMatch.queryIdx  第一幅图像中特征点的索引
        DMatch.trainIdx  第二幅图像中特征点的索引
        DMatch.distance  两个特征点之间的距离
        '''
        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])
        
        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            # self.prevFrame = src.copy()
            # self.prevKeyPoints = copy.copy(keypoints)
            # self.prevDescriptors = copy.copy(descriptors)
            return warp_matrix
        
        for m, n in knnMatches:
            # m代表距离最近的特征点，n代表距离次近的特征点
            if m.distance < self.disth * n.distance:
                prevKeyPointLocation = self.preKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        
        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances
        # print(inliesrs)
        
        goodMatches = []
        prevPoints = []
        currPoints = []

        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.preKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        
        '''
        # Draw the keypoint matches on the output image
        if plot:
            matches_img = np.hstack((self.preFrame, src))       # 水平拼接
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.preFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.preKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()
        '''   

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4):
            # cur_frame align to pre_frame
            warp_matrix, _ = cv2.estimateAffinePartial2D(currPoints, prevPoints, cv2.RANSAC)  

            # Handle self.downscale
            if self.downscale > 1.0:
                warp_matrix[0, 2] *= self.downscale
                warp_matrix[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')    
            
        return warp_matrix
    
    def applySparaseOptFlow(self, src):
        height, width = src.shape[0], src.shape[1]
        warp_matrix = np.eye(2, 3)
        
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        if self.downscale > 1.0:
            # src = cv2.GaussianBlur(src, (3, 3), 1.5)
            src = cv2.resize(src, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
        
        keypoints = cv2.goodFeaturesToTrack(src, mask=None, **self.feature_params)
        
        if self.FirstFrame:
            self.FirstFrame = False
            self.preFrame = src.copy()
            self.preKeyPoints = copy.copy(keypoints)           
            return warp_matrix
        
        # 计算稀疏光流
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.preFrame, src, self.preKeyPoints, None, **self.lk_params)
        
        prevPoints = self.preKeyPoints[status==1]
        currPoints = matchedKeypoints[status==1]
        
        if (np.size(prevPoints, 0) > 4):
            # warp_matrix, _ = cv2.estimateAffinePartial2D(currPoints, prevPoints, cv2.RANSAC)
            warp_matrix, _ = cv2.estimateAffinePartial2D(currPoints, prevPoints)
            # Handle downscale
            if self.downscale > 1.0:
                warp_matrix[0, 2] *= self.downscale
                warp_matrix[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')
        
        return warp_matrix
    
       
    
def main(method,downscale):
    # video_path = '/home/zxc/catkin_ws/src/video/1.mp4'
    video_path = '.\pitch.mp4'
    # origin_path = './results/origin.avi'
    # result_path = './results/align.avi'
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # videoWriter_align = cv2.VideoWriter(result_path,
    #     cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
    # videoWriter_origin = cv2.VideoWriter(origin_path,
    #     cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
        
    cv2.namedWindow('origin',0)
    cv2.resizeWindow('origin', 900,900)
    cv2.namedWindow('align',0)
    cv2.resizeWindow('align', 900,900)
    
    frame_id = 0
    align = GMC(method,downscale)
    
    avgtime = 0

    pts = np.array([[325,152],[943,162],[1040,507],[182,485]],np.int32) 
    pts = pts.reshape((-1, 1, 2)) 
    

    while True:
        ret,frame = cap.read()
        # w,h,c = frame.shape
        # R = cv2.getRotationMatrix2D((h*0.5, w*0.5), 180, 1)
        # frame = cv2.warpAffine(frame, R, (h,w))
        # cv2.imwrite('img_'+str(frame_id)+'.png', frame)
        if frame is None:
            break
        
        # 目标检测
        frame_id += 1

        tic = time.time()
        H = align.apply(frame)
        toc = time.time()
        print('*' * 100)
        print(H)
        align_image = cv2.warpAffine(frame, H, size, flags=cv2.INTER_LINEAR)
        
        avgtime = 0.95 * avgtime + 0.05 * (toc-tic)
        print(f'{avgtime*1000}ms')
        # print(H, align_image.shpae)
        
        # 结果可视化
        # cv2.polylines(align_image,[pts],isClosed=True,color=(0,0,255),thickness=2) 
        cv2.imshow('align', align_image)
        # videoWriter_align.write(align_image)     

        # cv2.polylines(frame,[pts],isClosed=True,color=(0,0,255),thickness=2) 
        cv2.imshow('origin',frame)    
        # videoWriter_origin.write(frame)
        
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # align_image = cv2.cvtColor(align_image, cv2.COLOR_BGR2RGB)
        # axes[0].imshow(frame)
        # axes[0].set_title('origin')  
        # axes[1].imshow(align_image)
        # axes[1].set_title('ecc_align')
        # plt.show()
        
        a = cv2.waitKey(1)
        # a = cv2.waitKey(int(1000/fps))
        if ord('q') == a:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
        
if __name__ == '__main__':
    '''
    method: orb   ecc  OptFlow sift
    downscale: (should > 1) default=1
    '''
    main('ecc',downscale=1)
    
