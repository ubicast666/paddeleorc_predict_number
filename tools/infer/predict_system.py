# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import configparser
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt
from math import *

logger = get_logger()


class TextSystem(object):
    def __init__(self, det_algorithm, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
                 use_dilation, det_east_score_thresh, det_east_cover_thresh,
                 det_east_nms_thresh, det_limit_side_len, det_sast_score_thresh,
                 det_sast_nms_thresh, det_sast_polygon, rec_algorithm, rec_model_dir, rec_image_shape,
                 rec_char_type, rec_batch_num, max_text_length, rec_char_dict_path, use_space_char,
                 use_angle_cls, cls_model_dir, cls_image_shape, label_list, cls_batch_num, cls_thresh,
                 enable_mkldnn, use_pdserving, det_model_dir, use_fp16, use_gpu, ir_optim, use_tensorrt, gpu_mem):
        self.text_detector = predict_det.TextDetector(det_algorithm, det_db_thresh, det_db_box_thresh,
                                                      det_db_unclip_ratio,
                                                      use_dilation, det_east_score_thresh, det_east_cover_thresh,
                                                      det_east_nms_thresh, det_limit_side_len, det_sast_score_thresh,
                                                      det_sast_nms_thresh, det_sast_polygon,
                                                      det_model_dir, use_gpu, gpu_mem, use_tensorrt, use_fp16,
                                                      max_batch_size, enable_mkldnn)
        self.text_recognizer = predict_rec.TextRecognizer(rec_algorithm, rec_model_dir, rec_image_shape,
                                                          rec_char_type, rec_batch_num, max_text_length,
                                                          rec_char_dict_path, use_space_char,
                                                          use_gpu, gpu_mem, use_tensorrt, use_fp16, max_batch_size,
                                                          enable_mkldnn)
        self.use_angle_cls = use_angle_cls
        self.drop_score = drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(cls_image_shape, cls_batch_num, cls_thresh, label_list,
                                                              cls_model_dir, use_gpu, gpu_mem, use_tensorrt, use_fp16,
                                                              max_batch_size, enable_mkldnn)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # print("img_crop_list",img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def remote(img, degree, color):
    height, width = img.shape[:2]
    radians = float(degree / 180 * pi)
    heightNew = int(width * fabs(sin((radians))) + height * fabs(cos((radians))))
    widthNew = int(height * fabs(sin((radians))) + width * fabs(cos((radians))))
    # 得到二维旋转的仿射矩阵
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    # print("matRotation:",matRotation)
    """
    matRotation: [[ 8.66025404e-01  5.00000000e-01 -3.34066963e+02]
                  [-5.00000000e-01  8.66025404e-01  8.69245123e+02]]

    matRotation: [[ 8.66025404e-01  5.00000000e-01 -4.82050808e+02]
                  [-5.00000000e-01  8.66025404e-01  1.20096189e+03]]

    """
    # 中心位置的平移
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    if color == "white":
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))  # 白底
    else:
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))  # 黑底
    return imgRotation

def main(image_dir, vis_font_path, drop_score, det_algorithm, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
         use_dilation, det_east_score_thresh, det_east_cover_thresh,
         det_east_nms_thresh, det_limit_side_len, det_sast_score_thresh,
         det_sast_nms_thresh, det_sast_polygon, rec_algorithm, rec_model_dir, rec_image_shape,
         rec_char_type, rec_batch_num, max_text_length, rec_char_dict_path, use_space_char,
         use_angle_cls, cls_model_dir, cls_image_shape, label_list, cls_batch_num, cls_thresh,
         enable_mkldnn, use_pdserving, det_model_dir, use_fp16, use_gpu, ir_optim, use_tensorrt, gpu_mem):
    image_file_list = get_image_file_list(image_dir)
    text_sys = TextSystem(det_algorithm, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
                          use_dilation, det_east_score_thresh, det_east_cover_thresh,
                          det_east_nms_thresh, det_limit_side_len, det_sast_score_thresh,
                          det_sast_nms_thresh, det_sast_polygon, rec_algorithm, rec_model_dir, rec_image_shape,
                          rec_char_type, rec_batch_num, max_text_length, rec_char_dict_path, use_space_char,
                          use_angle_cls, cls_model_dir, cls_image_shape, label_list, cls_batch_num, cls_thresh,
                          enable_mkldnn, use_pdserving, det_model_dir, use_fp16, use_gpu, ir_optim, use_tensorrt,
                          gpu_mem)
    is_visualize = True
    font_path = vis_font_path
    drop_score = drop_score
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
            print(img.shape)
            height = img.shape[0]
            width = img.shape[1]
            if height > 2000 and width > 3000 and img.shape[0] < img.shape[1]:
                imgRotation = remote(img, -90, "white")
                img = imgRotation
                # img = imgRotation[760:1800, 400:1760]
            # elif height > 3000 and width > 2000:
            #     img = img[760:1800, 400:1760]




        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        # print("starttime",starttime)
        dt_boxes, rec_res = text_sys(img)
        # print("rec_res:",rec_res)
        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        for text, score in rec_res:
            logger.info("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == "__main__":
    conf = configparser.ConfigParser()
    conf.read('../../config.conf')
    image_dir = conf.get("info", "image_dir")
    use_fp16 = conf.getboolean("info", "use_fp16")
    use_gpu = conf.getboolean("info", "use_gpu")
    ir_optim = conf.getboolean("info", "ir_optim")
    use_tensorrt = conf.getboolean("info", "use_tensorrt")
    gpu_mem = conf.getint("info", "gpu_mem")

    drop_score = conf.getfloat("info", "drop_score")

    vis_font_path = conf.get("info", "vis_font_path")
    det_algorithm = conf.get("info", "det_algorithm")

    det_model_dir = conf.get("info", "det_model_dir")
    det_limit_side_len = conf.getint("info", "det_limit_side_len")
    det_limit_type = conf.get("info", "det_limit_type")

    # DB parmas
    det_db_thresh = conf.getfloat("info", "det_db_thresh")
    det_db_box_thresh = conf.getfloat("info", "det_db_box_thresh")
    det_db_unclip_ratio = conf.getfloat("info", "det_db_unclip_ratio")
    max_batch_size = conf.getint("info", "max_batch_size")
    use_dilation = conf.getboolean("info", "use_dilation")

    # EAST parmas
    det_east_score_thresh = conf.getfloat("info", "det_east_score_thresh")
    det_east_cover_thresh = conf.getfloat("info", "det_east_cover_thresh")
    det_east_nms_thresh = conf.getfloat("info", "det_east_nms_thresh")

    # SAST parmas
    det_sast_score_thresh = conf.getfloat("info", "det_sast_score_thresh")
    det_sast_nms_thresh = conf.getfloat("info", "det_sast_nms_thresh")
    det_sast_polygon = conf.getboolean("info", "det_sast_polygon")

    rec_algorithm = conf.get("info", "rec_algorithm")
    rec_model_dir = conf.get("info", "rec_model_dir")
    rec_image_shape = conf.get("info", "rec_image_shape")
    rec_char_type = conf.get("info", "rec_char_type")
    rec_batch_num = conf.getint("info", "rec_batch_num")
    max_text_length = conf.getint("info", "max_text_length")
    rec_char_dict_path = conf.get("info", "rec_char_dict_path")
    use_space_char = conf.getboolean("info", "use_space_char")

    # params for text classifier
    use_angle_cls = conf.getboolean("info", "use_angle_cls")
    cls_model_dir = conf.get("info", "cls_model_dir")
    cls_image_shape = conf.get("info", "cls_image_shape")
    label_list = conf.get("info", "label_list")
    cls_batch_num = conf.getint("info", "cls_batch_num")
    cls_thresh = conf.getfloat("info", "cls_thresh")

    enable_mkldnn = conf.getboolean("info", "enable_mkldnn")
    use_pdserving = conf.getboolean("info", "use_pdserving")

    main(image_dir, vis_font_path, drop_score, det_algorithm, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
         use_dilation, det_east_score_thresh, det_east_cover_thresh,
         det_east_nms_thresh, det_limit_side_len, det_sast_score_thresh,
         det_sast_nms_thresh, det_sast_polygon, rec_algorithm, rec_model_dir, rec_image_shape,
         rec_char_type, rec_batch_num, max_text_length, rec_char_dict_path, use_space_char,
         use_angle_cls, cls_model_dir, cls_image_shape, label_list, cls_batch_num, cls_thresh,
         enable_mkldnn, use_pdserving, det_model_dir, use_fp16, use_gpu, ir_optim, use_tensorrt, gpu_mem)
