import sys
sys.path.append('../../')
import SimpleITK as sitk
import numpy as np

from skimage.transform import resize
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon
import cv2
import re

from converter.utils import nii_reader,trunc_gray,normalize



class Nii_Reader(object):
    def __init__(self,
                 image_path,
                 target_format=None,
                 label_path=None,
                 annotation_list=None,
                 trunc_flag=False,
                 normalize_flag=False):
        self.image_path = image_path
        self.target_format = target_format
        self.label_path = label_path
        self.annotation_list = annotation_list
        self.num_class = len(annotation_list)
        self.meta_data, self.images = nii_reader(self.image_path)
        self.trunc_flag = trunc_flag
        self.normalize_flag = normalize_flag

    def get_raw_images(self):
        if self.trunc_flag and self.target_format is not None:
            images = trunc_gray(self.images, self.target_format['scale'])
            if self.normalize_flag:
                return normalize(images)
            else:
                return images
        else:
            if self.normalize_flag:
                return normalize(self.images)
            else:
                return self.images

    def get_denoising_images(self):
        normal_image = trunc_gray(self.images, in_range=(-1000, 600))  #(-1000,600)
        normal_image = normalize(normal_image)
        tmp_images = self.get_raw_images()
        new_images = np.zeros_like(self.images, dtype=np.float32)
        for i in range(self.images.shape[0]):
            body = self.get_body(normal_image[i])
            new_images[i] = body * tmp_images[i]

        return new_images

    def get_resample_info(self):
        info = {}
        info['ori_shape'] = self.images.shape
        info['inplane_size'] = self.meta_data.GetSize()[0]
        info['z_scale'] = self.meta_data.GetSpacing()[-1]/ self.target_format['thickness']
        info['z_size'] = int(np.rint(info['z_scale'] * self.images.shape[0]))

        return info
    
    # resample on depth 
    def get_resample_images(self, in_raw=True):
        info = self.get_resample_info()
        if in_raw:
            if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
                return self.get_raw_images()
            else:
                images = self.get_raw_images()
                images = resize(images,
                                (info['z_size'], ) + tuple(self.target_format['size']),
                                mode='constant')
                return images


        else:
            if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
                return self.get_denoising_images()
            else:
                images = self.get_denoising_images()
                images = resize(images,
                                (info['z_size'], ) + tuple(self.target_format['size']),
                                mode='constant')
                return images


    def get_raw_labels(self):
        if self.label_path == None:
            raise ValueError("Need a Label data path!!")
        else:
            label_data = sitk.ReadImage(self.label_path)
            labels = sitk.GetArrayFromImage(label_data).astype(np.float32)
        return labels


    def get_resample_labels(self):
        info = self.get_resample_info()
        if info['inplane_size'] == self.target_format['size'][0] and info['z_scale'] == 1:
            return self.get_raw_labels()
        else:
            raw_label = self.get_raw_labels()
            labels = np.zeros((info['z_size'], ) + tuple(self.target_format['size']),dtype=np.float32)
            for i in range(self.num_class):
                roi = resize((raw_label == i + 1).astype(np.float32),
                             (info['z_size'], ) +tuple(self.target_format['size']),
                             mode='constant')
                labels[roi >= 0.5] = i + 1
            return labels


    def cropping(self, array, crop):
        return array[:, crop:-crop, crop:-crop]

    def padding(self, array, pad):
        return np.pad(array, ((0, 0), (pad, pad), (pad, pad)), 'constant')


    def get_body(self,image):
        img = rescale_intensity(image, out_range=(0, 255))
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body = cv2.erode(img, kernel, iterations=1)
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blur = cv2.GaussianBlur(body, (5, 5), 0)
        _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel_1, iterations=3)
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [[i, cv2.contourArea(contours[i])] for i in range(len(contours))]
        area.sort(key=lambda x: x[1], reverse=True)
        body = np.zeros_like(body, dtype=np.uint8)
        for i in range(min(len(area),3)):
            if area[i][1] > area[0][1] / 20:
                contour = contours[area[i][0]]
                r = contour[:, 0, 1]
                c = contour[:, 0, 0]
                rr, cc = polygon(r, c)
                body[rr, cc] = 1
        body = cv2.medianBlur(body, 5)

        return body

