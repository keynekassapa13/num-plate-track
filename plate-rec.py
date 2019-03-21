import cv2
import numpy as np
import os
import imutils
import collections
from matplotlib import pyplot as plt
from conf import *

'''
Some of the codes below has been adapted from

    OpenCV Documentation
    > docs.opencv.org

    OpenCV Python Tutorials
    > opencv-python-tutroals.readthedocs.io

I have used their reference to make my own function.
'''

class Plates:
    def __init__(self, ori_img, img_name, temp_num, color, pm_thresh, option=None):
        self.img_name = img_name
        self.ori = ori_img.copy()
        self.result = ori_img
        self.img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        self.temp_num = temp_num
        self.color = color
        self.pm_thresh = pm_thresh
        self.option = option

        # Do the work
        self.img_process()
        self.contour()
        self.pattern_matching()
        self.show_plt()

        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def img_process(self):
        '''
        Using standard Threshold function and inRange function to filter
        Get the high threshold and filter it again with inRange color extraction
        Some of images are categorized as 'special' where they have their own upper and lower threshold color range
        Did canny edge to detect edges after the images feature are highlighted
        Thus, the edges are identified from the highlighted area only.
        '''
        height = self.img.shape[0]
        width = self.img.shape[1]
        kernel = np.ones((2, 2),np.uint8)

        # Threshold
        _, mask = cv2.threshold(self.img, thresh=200, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if self.option is not None and self.option['no_bitwise']:
            self.img_mask = self.img
        else:
            self.img_mask = cv2.bitwise_and(self.img, mask)

        # inRange Threshold function
        if self.color in ('white', 'yellow'):
            hsv = cv2.cvtColor(self.ori, cv2.COLOR_BGR2HSV)
            h, s, v1 = cv2.split(hsv)
            if self.color == 'white':
                lower_white = np.array([0,0,160], dtype=np.uint8)
                upper_white = np.array([255,40,255], dtype=np.uint8)
            elif self.color == 'yellow':
                lower_white = np.array([20, 100, 100], dtype=np.uint8)
                upper_white = np.array([30, 255, 255], dtype=np.uint8)
            res_mask = cv2.inRange(hsv, lower_white, upper_white)
            self.res_img = cv2.bitwise_and(v1, self.img, mask=res_mask)
        else:
            if self.color == 'special' and self.option is not None and self.option['type'] == 'color':
                print(":::::Special - color")
                hsv = cv2.cvtColor(self.ori, cv2.COLOR_BGR2HSV)
                h, s, v1 = cv2.split(hsv)
                upper_white = self.option['upper_white']
                lower_white = self.option['lower_white']
                res_mask = cv2.inRange(hsv, lower_white, upper_white)
                self.res_img = cv2.bitwise_and(v1, self.img, mask=res_mask)
            else:
                self.res_img = self.img_mask

        # Edge Detection
        self.edges = cv2.Canny(self.res_img, height, width)
        # self.edges = cv2.Laplacian(self.res_img, cv2.CV_64F)
        # self.edges = cv2.Sobel(self.res_img, cv2.CV_64F, 0, 1, ksize=5)
        return

    def contour(self):
        '''
        Contour area of highlighted images
        Get some of the most contour areas
        Calculate the polygonal curve, pick with 4 curve (rect) OR
            Draw convex hull on the biggest contour area
        Crop of the picked contour/convex area
        '''

        # Contours
        _, contours, _ = cv2.findContours(
                self.res_img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        # cv2.drawContours(self.result, contours, -1, (150, 150, 255), 2)
        NumberPlateCnt = None
        found = False
        lt, rb = [10000, 10000], [0, 0]

        if self.color == 'white_bg':
            # Calculate polygonal curve, see if it has 4 curve
            for c in contours:
                 peri = cv2.arcLength(c, True)
                 approx = cv2.approxPolyDP(c, 0.06 * peri, True)
                 if len(approx) == 4:
                     found = True
                     NumberPlateCnt = approx
                     break
            if found:
                cv2.drawContours(self.result, [NumberPlateCnt], -1, (255, 0, 255), 2)

                for point in NumberPlateCnt:
                    cur_cx, cur_cy = point[0][0], point[0][1]
                    if cur_cx < lt[0]: lt[0] = cur_cx
                    if cur_cx > rb[0]: rb[0] = cur_cx
                    if cur_cy < lt[1]: lt[1] = cur_cy
                    if cur_cy > rb[1]: rb[1] = cur_cy

                cv2.circle(self.result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
                cv2.circle(self.result, (rb[0], rb[1]), 2, (150, 200, 255), 2)

                self.crop = self.res_img[lt[1]:rb[1], lt[0]:rb[0]]
                self.crop_res = self.ori[lt[1]:rb[1], lt[0]:rb[0]]
            else:
                self.crop = self.res_img.copy()
                self.crop_res = self.ori.copy()
        elif len(contours) > 0:
            # Convex Hull
            hull = cv2.convexHull(contours[0])
            # cv2.drawContours(ori_img, [hull], -1, (255, 0, 255),  2, 8)
            approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
            cv2.drawContours(self.result, [approx2], -1, (255, 0, 255), 2, lineType=8)

            for point in approx2:
                cur_cx, cur_cy = point[0][0], point[0][1]
                if cur_cx < lt[0]: lt[0] = cur_cx
                if cur_cx > rb[0]: rb[0] = cur_cx
                if cur_cy < lt[1]: lt[1] = cur_cy
                if cur_cy > rb[1]: rb[1] = cur_cy

            cv2.circle(self.result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
            cv2.circle(self.result, (rb[0], rb[1]), 2, (150, 200, 255), 2)

            self.crop = self.res_img[lt[1]:rb[1], lt[0]:rb[0]]
            self.crop_res = self.ori[lt[1]:rb[1], lt[0]:rb[0]]
        else:
            self.crop = self.res_img.copy()
            self.crop_res = self.ori.copy()

        return

    def pattern_matching(self):
        '''
        Pattern Matching is used to identify numbers in the cropped image
        CV_TM_CCOEFF is used in this case
        * Result is still not accurate
        '''
        self.pm = {}

        method = cv2.TM_CCOEFF_NORMED
        threshold = self.pm_thresh
        cw, ch = self.crop.shape[::-1]

        # cv2.imshow("crop", self.crop)

        for temp in self.temp_num:
            highest = 0
            highest_pt = []
            for i in range(1, 4):
                temp_result = []
                t_img = cv2.imread("./temp-num/{}-0{}.png".format(temp, str(i)), 0)
                t_img = imutils.resize(t_img, height = ch-2)
                w, h = t_img.shape[::-1]

                res = cv2.matchTemplate(self.crop, t_img, method)
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    temp_result.append(pt)
                    cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
                if len(temp_result) > highest:
                    highest = len(temp_result)
                    highest_pt = temp_result

            for pt in highest_pt:
                cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
                self.pm[pt[0]] = temp

        self.pm = collections.OrderedDict(sorted(self.pm.items()))
        self.pm_result = ''
        for _, pm in self.pm.items():
            self.pm_result += pm

        print("::::::RESULT = {}\n".format(self.pm_result))
        return

    def show_plt(self):
        '''
        Showing 6 main step of the process for turning original frame to the result frame
        '''
        title = [
            'Black and White',
            'Threshold',
            'Canny',
            'Num Plate Detected',
            'Num Plate Cropped',
            'Predicted Num :\n'+ self.pm_result
        ]
        result = [self.img, self.res_img, self.edges, self.result[:, :, ::-1], self.crop, self.crop_res]
        num = [231, 232, 233, 234, 235, 236]

        for i in range(len(result)):
            plt.subplot(num[i]),plt.imshow(result[i], cmap = 'gray')
            plt.title(title[i]), plt.xticks([]), plt.yticks([])

        plt.suptitle(self.img_name)
        plt.show()

if __name__ == '__main__':
    temp_num = [f for f in os.listdir('./temp-num') if os.path.isfile(os.path.join('./temp-num', f))]
    plt.rcParams["figure.figsize"] = (15,10)
    for f in files:
        print("-----Numberplates >> {}".format(f[0]['name']))
        ori_img = cv2.imread('./numberplates/' + f[0]['name'])
        option = None
        if f[0]['type'] == 'special':
            option = f[0]['option']
        Plates(ori_img, f[0]['name'], temp_files, f[0]['type'], f[0]['pm_thresh'], option)
