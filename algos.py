import numpy as np
import cv2


def threshold(img, thr_value=64):
    thr_img = cv2.threshold(img, thr_value, 255, cv2.THRESH_OTSU)[1]
    thr_img = np.bitwise_not(thr_img)
    return thr_img


def erosion(img, kernel=(3, 1), num_iters=5):
    new_kernel = np.ones(kernel)
    ero_img = cv2.erode(img, new_kernel, iterations=num_iters)
    return ero_img


def dilate(img, kernel=(32, 2), num_iters=5):
    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    dil_img = cv2.dilate(img, new_kernel, iterations=num_iters)
    return dil_img


def contour(img, add_offset=False):
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cont_img = cv2.drawContours(np.zeros(img.shape), contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    bboxes = []
    for contour in contours:
        bbox = list(cv2.boundingRect(contour))
        if add_offset:
            bbox[1] = bbox[1] - 10 if bbox[1] > 10 else 0
            bbox[3] = bbox[3] + 10 if bbox[1]+bbox[3] + 10 <= img.shape[0] else img.shape[0]

        bboxes.append(bbox)

    clean_bboxes = []
    for bbox in bboxes:
        if np.sum([b[0] <= bbox[0] <= b[0] + b[2] and
                       b[1] <= bbox[1] <= b[1] + b[3] and
                       b[0] <= bbox[0] + bbox[2] <= b[0] + b[2] and
                       b[1] <= bbox[1] + bbox[3] <= b[1] + b[3]
                       for b in bboxes]) <= 1:
            clean_bboxes.append(bbox)

    return cont_img, clean_bboxes
