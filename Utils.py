import re
import os
import json
import cv2
import math
import random
import logging
import tensorflow as tf
import numpy as np
import statistics
import matplotlib.pyplot as plt
import xml.etree.ElementTree as etree

from datetime import datetime
from xml.dom import minidom
from scipy.signal import find_peaks

from DataClasses import Bbox, LineData, LayoutData



def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


def read_label(file_path: str) -> str:
    f = open(file_path, "r", encoding="utf-8")
    return f.readline()


def get_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read()
        json_content = json.loads(lines)

    return json_content

def read_lines_from_file(file_path: str) -> list[str]:
    with open (file_path, "r", encoding="utf-8") as f:
        content = f.readlines()
    
    return content


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def show_image(image: np.array, cmap: str = "", axis="off") -> None:
    plt.figure(figsize=(24, 13))
    plt.axis(axis)
    
    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

def show_overlay(image: np.array, mask: np.array, alpha=0.4, axis="off"):
    plt.figure(figsize=(24, 13))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


def batch_data(images: list[str], batch_size: int = 8) -> list[str]:
    if len(images) % batch_size == 0:
        num_batches = len(images) // batch_size

    else:
        num_batches = (len(images) // batch_size) + 1

    img_batches = np.array_split(images, num_batches)

    return img_batches


def get_lines_from_page(source_file: str) -> list[str]:
    """
    Returns a list of lines collected from a custom line-based annotation file, which
    uses <line> tagging for line separation, and occasionally <gap> tagging to mark wider gaps in pages.
    """

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()
        content = content.split("<line>")
        content = [x.replace("<gap>", "") for x in content]
        content = [x.replace("\n", "") for x in content]
        content = [x for x in content if x != ""]
    
    return content


def resize_image(
    orig_img: np.array, target_width: int = 2048
) -> tuple[np.array, float]:
    if orig_img.shape[1] > orig_img.shape[0]:
        resize_factor = round(target_width / orig_img.shape[1], 2)
        target_height = int(orig_img.shape[0] * resize_factor)

        resized_img = cv2.resize(orig_img, (target_width, target_height))

    else:
        target_height = target_width
        resize_factor = round(target_width / orig_img.shape[0], 2)
        target_width = int(orig_img.shape[1] * resize_factor)
        resized_img = cv2.resize(orig_img, (target_width, target_height))

    return resized_img, resize_factor


def get_random_color():
    rand_r = random.randint(40, 255)
    rand_g = random.randint(40, 200)
    rand_b = random.randint(40, 220)
    color = (rand_r, rand_g, rand_b)

    return color

def pad_image(img: np.array, patch_size: int = 64, 
              is_mask=False, pad_value: int = 255) -> tuple[np.array, tuple[int, int]]:
    
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= pad_value
        pad_x *= pad_value

    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def resize_to_height(image, target_height: int):
    width_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * width_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def resize_to_width(image, target_width: int):
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * width_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def binarize_line(img: np.array, adaptive: bool = True, block_size: int = 11, c: int = 13) -> np.array:
    line_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def pad_image2(
    img: np.array, target_width: int, target_height: int, padding: str
) -> np.array:
    width_ratio = target_width / img.shape[1]
    height_ratio = target_height / img.shape[0]

    if width_ratio < height_ratio:  # maybe handle equality separately
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])

    elif width_ratio > height_ratio:
        tmp_img = resize_to_height(img, target_height)

        if padding == "white":
            h_stack = np.ones(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            h_stack = np.zeros(
                (tmp_img.shape[0], target_width - tmp_img.shape[1]), dtype=np.uint8
            )
        h_stack *= 255

        out_img = np.hstack([tmp_img, h_stack])
    else:
        tmp_img = resize_to_width(img, target_width)

        if padding == "white":
            v_stack = np.ones(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        else:
            v_stack = np.zeros(
                (target_height - tmp_img.shape[0], tmp_img.shape[1]), dtype=np.uint8
            )
        v_stack *= 255

        out_img = np.vstack([v_stack, tmp_img])
        # print(f"Info -> equal ratio: {img.shape}, w_ratio: {width_ratio}, h_ratio: {height_ratio}")

    return cv2.resize(
        out_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )

def prepare_img_patches(image: np.array) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.dstack([image, image, image])
    return image

def patch_image(
    img: np.array, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> list:
    """
    A simple slicing function.
    Expects input_image.shape[0] and image.shape[1] % patch_size = 0
    """

    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> np.array:
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        final_out = np.vstack((final_out, x_stack))

    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def unpatch_prediction(prediction: np.array, y_splits: int) -> np.array:
    prediction *= 255
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def rotate_from_angle(image: np.array, angle: float) -> np.array:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    rotated_img = cv2.warpAffine(
        image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img


def rotate_from_angle_around_center(image: np.array, angle: float, center: tuple[int, int]) -> np.array:
    rows, cols = image.shape[:2]    
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(
        image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img


def rotate_page(
    original_image: np.array,
    line_mask: np.array,
    max_angle: float = 3.0,
    debug_angles: bool = False,
) -> float:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    # angles = [x for x in angles if abs(x) != 0.0 and x != 90.0]
    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if debug_angles:
        print(angles)

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = np.mean(low_angles)

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = -(90 - np.mean(high_angles))

    else:
        mean_angle = 0


    rows, cols = original_image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_angle, 1)
    rotated_img = cv2.warpAffine(
        original_image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    rotated_prediction = cv2.warpAffine(
        line_mask, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img, rotated_prediction, mean_angle


def rotate_from_hough(image: np.array) -> tuple[np.array, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11)

    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        threshold=130,
        minLineLength=40,
        maxLineGap=8
    )

    if lines is None or len(lines) == 0:
        logging.warning(f"No lines found in image, skipping...")

        return image, 0
    
    #prev_img = image.copy()
    angles = []
    zero_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2-y1, x2-x1) * 180 / np.pi

        if abs(angle) < 5 and abs(angle) > 0:
            angles.append(angle)

        elif int(angle) == 0:
            zero_angles.append(angle)

        #cv2.line(prev_img, (x1, y1), (x2, y2), (100, 100, 0), 3)

    if len(angles) != 0:     
        avg_angle = statistics.median(angles)
        ratio = (len(zero_angles) / len(angles))

        if ratio < 0.5:
            rot_angle = avg_angle
        elif ratio > 0.5 and ratio < 0.9:
            rot_angle = avg_angle / 2
        else:
            rot_angle = 0.0
    else:
        logging.warning("No angle data found in image.")
        rot_angle = 0

    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), (rot_angle), 1)
    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img, rot_angle

def get_line_image_from_bbox(image: np.array, line: LineData, x_offset: int = 15, y_offset: float = 0.7):    
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    min_rect = cv2.minAreaRect(line.contour)
    center, width_height, angle = min_rect

    w, h = width_height
    if angle > 87 and w < h:
        angle = (90 -angle)

        min_rect = (center, (h, w), angle)
    else:
        angle = -angle
        
    box = cv2.boxPoints(min_rect)
    box = np.intp(box)
    box = np.expand_dims(box, axis=1)
    rotated_box = rotate_contour(box, center, angle)
    more_rot_img = rotate_from_angle_around_center(image, angle, center=(int(center[0]), int(center[1])))
    rotated_b_rect = cv2.boundingRect(rotated_box)
    x, y, w, h = rotated_b_rect
    y_off = int(h * y_offset)
    x_off = x_offset

    cv2.rectangle(
                image_mask,
                (x-x_off, y-y_off),
                (x + w + x_off, y + h + y_off),
                (255, 255, 255),
                -1,
            )
    
    image_masked = cv2.bitwise_and(more_rot_img, more_rot_img, mask=image_mask)
    image_masked = np.delete(image_masked, np.where(~image_masked.any(axis=1))[0], axis=0)
    image_masked = np.delete(image_masked, np.where(~image_masked.any(axis=0))[0], axis=1)

    return image_masked


def get_kernel_line(
    image: np.array,
    line: LineData,
    dilate_kernel: int = 24, # TODO: dynamic kernel calculation..
    dilate_iterations: int = 1,
    binarize: bool = False,
    calculate_iterations: bool = True
):
    # enforce a default value in case the somebody passes 0 as value
    if dilate_kernel == 0:
        dilate_kernel = 24
   
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    _, _, angle = cv2.minAreaRect(line.contour)


    cv2.drawContours(
        image_mask, [line.contour], contourIdx=0, color=(255, 255, 255), thickness=-1
    )

    if angle > 87:
        angle = angle - 90
    
    image_mask = rotate_from_angle(image_mask, angle)
    image = rotate_from_angle(image, angle)

    # kernel iteration calculation based on the actual contour by using multiple slices of the contour instead of the bbox since the bbox can be too large in case of bended lines

    dilate_iterations = 1

    slice_width = 20
    steps = 8
    x, y, w, h = cv2.boundingRect(image_mask)
    bbox_width = w - x
    x_steps = (bbox_width // steps) - 1 - slice_width

    slices = []
    heights = []

    for step in range(1, steps+1):
        x_offset = x_steps * step
        x_start = x+x_offset
        x_end = x_start+slice_width

        slice = image_mask[y:y+h, x_start:x_end]
        slices.append(slice)

        contours, _ = cv2.findContours(slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0]) # or use the max contour here?
            heights.append(h)

    mean_height = np.mean(heights)
    #print(f"mean height: {mean_height}")

    if mean_height <= 10:
        dilate_iterations = int(dilate_kernel // 1.5)
    elif mean_height > 10 and mean_height < 30:
        dilate_iterations = int(dilate_kernel * 0.8)
    elif mean_height >= 30 and mean_height < 50:
        dilate_iterations = dilate_kernel * 2
    elif mean_height > 50:
        dilate_iterations = (
            dilate_kernel * 6
        )  # maybe go with 4-5? kernel sizes can go up to 126
    else:
        #print(f"Fallback on kernel calculation, height: {mean_height}: dilate_kernel = {dilate_kernel}, iterations: {dilate_iterations}")
        dilate_iterations = 1

    #print(f"Calculated kernel iterations: {dilate_iterations}")

    """
    old calcuation using the bbox of the line

    if calculate_iterations:
        dilate_iterations = calculate_dilate_iterations(line, dilate_kernel)
    """

    dilated1 = cv2.dilate(
        image_mask,
        kernel=(dilate_kernel, dilate_kernel),
        iterations=dilate_iterations,
        borderValue=0,
        anchor=(-1, 0),
        borderType=cv2.BORDER_DEFAULT,
    )
    dilated2 = cv2.dilate(
        image_mask,
        kernel=(dilate_kernel, dilate_kernel),
        iterations=dilate_iterations,
        borderValue=0,
        anchor=(0, 1),
        borderType=cv2.BORDER_DEFAULT,
    )
    combined = cv2.add(dilated1, dilated2)
    image_masked = cv2.bitwise_and(image, image, mask=combined)

    cropped_img = np.delete(image_masked, np.where(~image_masked.any(axis=1))[0], axis=0)
    cropped_img = np.delete(cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1)
    #cropped_img = cv2.dilate(cropped_img, kernel=(1, 8), iterations=2)
    
    if binarize:
        # indices = np.where(cropped_img[:, :, 1] == 0)
        # clear = cropped_img.copy()
        # clear[indices[0], indices[1], :] = [255, 255, 255]
        # clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)
        # line_images.append(thresh)
        cropped_img = binarize_line(cropped_img)
    
    return cropped_img, combined, image_mask



def get_line_images(
    image: np.array,
    sorted_line_contours: dict,
    dilate_kernel: int = 24,
    binarize: bool = False,
):
    line_images = []

    # enforce a default value in case the somebody passes 0 as value
    if dilate_kernel == 0:
        dilate_kernel = 24

    dilate_iterations = 1

    for _, contour in sorted_line_contours.items():
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(
            image_mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1
        )

        _, _, _, height = cv2.boundingRect(contour)

        if height <= 30:
            dilate_iterations = dilate_kernel // 2
        elif height > 30 and height < 50:
            dilate_iterations = dilate_kernel * 1
        elif height >= 50 and height < 90:
            dilate_iterations = dilate_kernel * 2
        elif height > 90:
            dilate_iterations = (
                dilate_kernel * 6
            )  # maybe go with 4-5? kernel sizes can go up to 126
        else:
            print(
                f"Fallback on kernel calculation, height: {height}: dilate_kernel = {dilate_kernel}, iterations: {dilate_iterations}"
            )
        # print(f"BBox Height: {height} => dilate iterations: {dilate_iterations}")

        dilated1 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(-1, 0),
            borderType=cv2.BORDER_DEFAULT,
        )
        dilated2 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(0, 1),
            borderType=cv2.BORDER_DEFAULT,
        )
        combined = cv2.add(dilated1, dilated2)
        image_masked = cv2.bitwise_and(image, image, mask=combined)

        cropped_img = np.delete(
            image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
        )
        cropped_img = np.delete(
            cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
        )

        # print(cropped_img.shape)

        if binarize:
            # indices = np.where(cropped_img[:, :, 1] == 0)
            # clear = cropped_img.copy()
            # clear[indices[0], indices[1], :] = [255, 255, 255]
            # clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)
            # line_images.append(thresh)
            cropped_img = binarize_line(cropped_img)
            line_images.append(cropped_img)
        else:
            line_images.append(cropped_img)

    return line_images

def calculate_dilate_iterations(line: LineData, kernel: int):
    bbox_height = line.bbox.h

    if bbox_height <= 30:
        dilate_iterations = kernel // 2
    elif bbox_height > 30 and bbox_height < 50:
        dilate_iterations = kernel * 1
    elif bbox_height >= 50 and bbox_height < 90:
        dilate_iterations = kernel * 2
    elif bbox_height > 90:
        dilate_iterations = (
            kernel * 6
        )  # maybe go with 4-5? kernel sizes can go up to 126
    else:
        print(
            f"Fallback on kernel calculation, height: {bbox_height}: dilate_kernel = {kernel}, iterations: {dilate_iterations}"
        )
        dilate_iterations = 1
    # print(f"BBox Height: {height} => dilate iterations: {dilate_iterations}")


    return dilate_iterations

def get_line_images_from_line_data(
    image: np.array,
    line_data: list[LineData],
    dilate_kernel: int = 24, # TODO: dynamic kernel calculation..
    dilate_iterations: int = 1,
    binarize: bool = False,
    calculate_iterations: bool = True
):
    line_images = []

    # enforce a default value in case the somebody passes 0 as value
    if dilate_kernel == 0:
        dilate_kernel = 24


    for line in line_data:
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(
            image_mask, [line.contour], contourIdx=0, color=(255, 255, 255), thickness=-1
        )
        if calculate_iterations:
            dilate_iterations = calculate_dilate_iterations(line, dilate_kernel)

        dilated1 = cv2.dilate(
            image_mask,
            kernel=(int(1.5*dilate_kernel), dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(-1, 0),
            borderType=cv2.BORDER_DEFAULT,
        )
        dilated2 = cv2.dilate(
            image_mask,
            kernel=(int(1.5*dilate_kernel), dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(0, 1),
            borderType=cv2.BORDER_DEFAULT,
        )
        combined = cv2.add(dilated1, dilated2)
        image_masked = cv2.bitwise_and(image, image, mask=combined)

        cropped_img = np.delete(
            image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
        )
        cropped_img = np.delete(
            cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
        )

        # print(cropped_img.shape)

        if binarize:
            # indices = np.where(cropped_img[:, :, 1] == 0)
            # clear = cropped_img.copy()
            # clear[indices[0], indices[1], :] = [255, 255, 255]
            # clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)
            # line_images.append(thresh)
            cropped_img = binarize_line(cropped_img)
            line_images.append(cropped_img)
        else:
            line_images.append(cropped_img)

    return line_images


def group_bbox_centers(bbox_centers: list[tuple[int, int]], line_threshold: int = 20):
  
  sorted_bbox_centers = []
  tmp_line = []

  for i in range(0, len(bbox_centers)):    
      if len(tmp_line) > 0:
        for s in range(0, len(tmp_line)):
          ys = [y[1] for y in tmp_line]
          mean_y = np.mean(ys)

          #y_diff = abs(tmp_line[s][1] - bbox_centers[i][1])
          """
          I use the mean of the hitertho present line chunks in tmp_line since
          the precalculated fixed threshold can break the sorting if
          there is some slight bending in the line. This part may need some tweaking after
          some further practical review
          """
          y_diff = abs(mean_y - bbox_centers[i][1])

          if y_diff > line_threshold:
            tmp_line.sort(key=lambda x:x[0])
            sorted_bbox_centers.append(tmp_line.copy())
            tmp_line.clear()

            tmp_line.append(bbox_centers[i])
            break
          else:
            tmp_line.append(bbox_centers[i])
            break
      else:
        tmp_line.append(bbox_centers[i])

  sorted_bbox_centers.append(tmp_line)

  for y in sorted_bbox_centers:
    y.sort(key=lambda x:x[0])

  sorted_bbox_centers = list(reversed(sorted_bbox_centers))

  return sorted_bbox_centers


def get_line_treshold(line_prediction: np.array, slice_width: int = 20, steps: int = 8):
    """
    This function generates n slices (of n = steps) width the width of slice_width across the bbox of the detected lines.
    The slice with the max. number of contained contours is taken to be the canditate to calculate the bbox center of each contour and
    take the median distance between each bbox center as estimated line cut-off threshold to sort each line segment across the horizontal

    Note: This approach might turn out to be problematic in case of sparsely spread line segments across a page
    """
    
    slice_width = 20
    steps = 8
    x, y, w, h = cv2.boundingRect(line_prediction)
    bbox_width = w - x
    x_steps = (bbox_width // steps) - 1 - slice_width

    bbox_numbers = []

    for step in range(1, steps+1):
        x_offset = x_steps * step
        x_start = x+x_offset
        x_end = x_start+slice_width
        #print(f"{x_start}:{x_end}")

        slice = line_prediction[y:y+h, x_start:x_end]
        contours, _ = cv2.findContours(
                    slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
        #bbox = cv2.boundingRect(contour)
        bbox_numbers.append((len(contours), contours))

    sorted_list = sorted(bbox_numbers, key = lambda x : x[0], reverse=True)
    reference_slice = sorted_list[0]

    y_points = []
    n_contours, contours = reference_slice

    for _, contour in enumerate(contours):
        x, y, w, h  = cv2.boundingRect(contour)
        #x_center = x + (w // 2)
        y_center = y + (h // 2)
        y_points.append(y_center)
        
    
    line_threshold = np.median(y_points) // n_contours

    return line_threshold

def get_line_bbox_centers(layout_data: LayoutData) -> list[tuple[int, int]]:
    # TODO: change input to a list of LineData, so that this function can also run within the line detection-only workflow
    centers = []

    for line in layout_data.lines:
        x_center = line.bbox.x + (line.bbox.w // 2)
        y_center = line.bbox.y + (line.bbox.h // 2)
       
        centers.append((x_center, y_center))

    return centers



def get_lines(image: np.array, prediction: np.array, line_kernel: int, binarize: bool):
    line_contours, _ = cv2.findContours(
        prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    x, y, w, h = cv2.boundingRect(prediction)

    if len(line_contours) == 0:
        return [], None, None, None

    elif len(line_contours) == 1:
        bbox_center = (x + w // 2, y + h // 2)
        peaks = [bbox_center]
        sorted_contours = {bbox_center: line_contours[0]}
        line_images = get_line_images(
            image, sorted_contours, dilate_kernel=line_kernel, binarize=binarize
        )

        return line_images, sorted_contours, (x, y, w, h), peaks
    else:
        sorted_contours, peaks = sort_lines(prediction, line_contours)
        line_images = get_line_images(
            image, sorted_contours, dilate_kernel=line_kernel, binarize=binarize
        )

        return line_images, sorted_contours, (x, y, w, h), peaks


def sort_lines(line_prediction: np.array, contours: tuple):
    """
    A preliminary approach to sort the found contours and sort them by reading lines. The relative distance between the lines is currently taken as roughly constant,
    wherefore mean // 2 is taken as threshold for line breaks. This might not work in scenarios in which the line distances are less constant.

    Args:
        - tuple of contours returned by cv2.findContours()
    Returns:
        - dictionary of {(bboxcenter_x, bbox_center_y) : [contour]}
        - peaks returned by find_peaks() marking the line breaks
    """

    horizontal_projection = np.sum(line_prediction, axis=1)
    horizontal_projection = horizontal_projection / 255
    mean = int(np.mean(horizontal_projection))
    peaks, _ = find_peaks(horizontal_projection, height=mean, width=4)

    # calculate the line distances
    line_distances = []
    for idx in range(len(peaks)):
        if idx < len(peaks) - 1:
            line_distances.append(
                peaks[(len(peaks) - 1) - idx] - (peaks[(len(peaks) - 1) - (idx + 1)])
            )

    if len(line_distances) == 0:
        line_distance = 0
    else:
        line_distance = int(
            np.mean(line_distances)
        )  # that might not work great if the line distances are varying a lot

    # get the bbox centers of each contour and keep a reference to the contour in contour_dict
    centers = []
    contour_dict = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_center = y + (h // 2)
        x_center = x + (w // 2)
        centers.append((x_center, y_center))
        contour_dict[(x_center, y_center)] = contour

    centers = sorted(centers, key=lambda x: x[1])

    # associate bbox centers with the peaks (i.e. line breaks)
    cnt_dict = {}

    for center in centers:
        if center == centers[-1]:
            cnt_dict[center[1]] = [center]
            continue

        for peak in peaks:
            diff = abs(center[1] - peak)
            if diff <= line_distance // 2:
                if peak in cnt_dict.keys():
                    cnt_dict[peak].append(center)
                else:
                    cnt_dict[peak] = [center]

    # sort bbox centers for x value to get proper reading order
    for k, v in cnt_dict.items():
        if len(v) > 1:
            v = sorted(v)
            cnt_dict[k] = v

    # build final dictionary with correctly sorted bbox_centers by y and x -> contour
    sorted_contour_dict = {}
    for k, v in cnt_dict.items():
        for l in v:
            sorted_contour_dict[l] = contour_dict[l]

    return sorted_contour_dict, peaks


def group_line_chunks(sorted_bbox_centers, layout_data: LayoutData, debug: bool = False):
    new_line_data = []
    for bbox_centers in sorted_bbox_centers:

      if len(bbox_centers) > 1: # i.e. more than 1 bbox center in a group
          contour_stack = []
          
          for box_center in bbox_centers:
            for line_data in layout_data.lines:
              if box_center == line_data.center:
                  contour_stack.append(line_data.contour)
                  break
        
          stacked_contour = np.vstack(contour_stack)
          stacked_contour = cv2.convexHull(stacked_contour)
          x, y, w, h = cv2.boundingRect(stacked_contour)
          _bbox = Bbox(x, y, w, h)
          x_center = _bbox.x + (_bbox.w // 2)
          y_center = _bbox.y + (_bbox.h // 2)

          new_line = LineData(
             contour=stacked_contour,
             bbox=_bbox,
             center = (x_center, y_center))
          
          new_line_data.append(new_line)
         

      else:
        for _bcenter in bbox_centers:
            for line_data in layout_data.lines:
              if _bcenter == line_data.center:      
                new_line_data.append(line_data)
                break
        """
        for line_data in layout_data.lines:
              if line_data.center == line_data.center:
                 print(line_data.center)
                 new_line_data.append(line_data)
                 break
        """       
    return new_line_data


def generate_line_image(image, contour, kernel: tuple = (10, 14), iterations: int = 4):
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(image_mask, [contour], contourIdx=-1, color=(255, 255, 255), thickness=-1)
    dilate_k = np.ones(kernel, dtype=np.uint8)
    kernel_iterations = iterations

    image_mask = cv2.dilate(image_mask, dilate_k, iterations=kernel_iterations)
    image_masked = cv2.bitwise_and(image, image, mask=image_mask)

    cropped_img = np.delete(image_masked, np.where(~image_masked.any(axis=1))[0], axis=0)
    cropped_img = np.delete(cropped_img,np.where(~cropped_img.any(axis=0))[0], axis=1)   
    #display(Image.fromarray(cropped_img))

    #indices = np.where(cropped_img[:, :]==0)
    #clear = cropped_img.copy()
    #clear[indices[0], indices[1]] = [255, 255, 255]
    #clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
    #_ ,thresh = cv2.threshold(clear_bw, 170, 255,cv2.THRESH_BINARY)

    return cropped_img


def generate_line_images(
    image: np.array, prediction: np.array, line_kernel: int, binarize: bool
):
    """
    Applies some rotation correction to the original image and creates the line images based on the predicted lines.
    """
    logging.warn("This function will be removed soon.")
    rotated_img, rotated_prediction, angle = rotate_page(
        original_image=image, line_mask=prediction
    )
    line_images, sorted_contours, bbox, peaks = get_lines(
        rotated_img, rotated_prediction, line_kernel, binarize
    )

    return rotated_img, line_images, sorted_contours, bbox, peaks, angle


"""
The code for the contour rotation is taken from: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
For this use case the contour is rotated 'back' using the center of the image, instead of the centroid of the contour
"""


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def rotate_contour(cnt, center, angle):
    cx = center[0]
    cy = center[1]

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def back_rotate_lines(image: np.array, sorted_contours: dict, angle: float):
    rows, cols = image.shape[:2]
    line_contours = list(sorted_contours.values())
    rotated_cnts = [rotate_contour(x, (cols / 2, rows / 2), angle) for x in line_contours]
    rotated_cnts = [optimize_countour(x, e=0.001) for x in rotated_cnts]

    return rotated_cnts

def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_ocr_image(
line_image: np.array, target_width: int = 2000, target_height: int = 80
):
    if len(line_image.shape) > 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    image = pad_image2(
        line_image,
        target_width=target_width,
        target_height=target_height,
        padding="black",
    )

    image = image.reshape((1, target_height, target_width))
    #image = (image / 127.5) - 1.0
    iamge = image / 255.0
    image = image.astype(np.float32)
    return image


def prepare_tf_ocr_image(image: np.array, target_height: int = 80, target_width: int = 2000):
    tf_img = tf.image.resize_with_pad(image, target_height, target_width)
    tf_img = tf_img / 255.0
    
    tf_img = tf.transpose(tf_img, perm=[1, 0, 2])
    tf_img = tf.squeeze(tf_img)

    return tf_img


def get_utc_time():
    t = datetime.now()
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    s = s.split(" ")

    return f"{s[0]}T{s[1]}"


def get_text_points(contour):
    points = ""
    for box in contour:
        point = f"{box[0][0]},{box[0][1]} "
        points += point
    return points


def get_text_line_block(coordinate, index, unicode_text):
    text_line = etree.Element(
        "Textline", id="", custom=f"readingOrder {{index:{index};}}"
    )
    text_line = etree.Element("TextLine")
    text_line_coords = coordinate

    text_line.attrib["id"] = f"line_9874_{str(index)}"
    text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

    coords_points = etree.SubElement(text_line, "Coords")
    coords_points.attrib["points"] = text_line_coords
    text_equiv = etree.SubElement(text_line, "TextEquiv")
    unicode_field = etree.SubElement(text_equiv, "Unicode")
    unicode_field.text = unicode_text

    return text_line


def process_wylie_label(label: str) -> str:
    label = label.replace("\\u0f85", "&")
    label = label.replace("\\u0f09", "ä")
    label = label.replace("\\u0f13", "ö")
    label = label.replace("\\u0f12", "ü")
    label = label.replace("\\u0fd3", "@")
    label = label.replace("\\u0fd4", "#")
    label = label.replace("\\u0f00", "oM")
    label = label.replace("\\u0f7f", "}")
    label = label.replace(":", "aH ")
    label = label.replace("*", " ")
    label = label.replace("  ", " ")
    label = label.replace("_", "")
    label = label.replace(" ", "§") # specific encoding for the tsheg

    label = re.sub("[\[(].*?[\])]", "", label)

    return label



def prepare_ocr_image2(
    line_image: np.array, target_width: int = 2000, target_height: int = 80
    ):
        if len(line_image.shape) > 2:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        image = pad_image2(
            line_image,
            target_width=target_width,
            target_height=target_height,
            padding="black",
        )
        
        image = image.reshape((1, target_height, target_width))

        #image = (image / 127.5) - 1.0
        image = image / 255.0
        image = image.astype(np.float32)
        return image


def adaptive_binarize(image: np.array, block_size: int = 13, c: float = 11, invert: bool = False):
    c = round(c, 2)
    
    bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    if invert:
        bw = cv2.bitwise_not(bw)

    return bw

def tweak_image(line_img: np.array):
    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
    #line_img = cv2.GaussianBlur(line_img, (3, 3), 0)
    line_img = cv2.erode(line_img, kernel=(3, 3), iterations=1)
    line_img = adaptive_binarize(line_img)

    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    return line_img



# old xml export
def build_xml_document(
    image: np.array,
    image_name: str,
    text_region_bbox,
    coordinates: list,
    text_lines: list,
):
    #logging.warn("This function will be deprecated soon, don't use it anymore.")
    root = etree.Element("PcGts")
    root.attrib[
        "xmlns"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
    root.attrib[
        "xsi:schemaLocation"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Transkribus"
    created = etree.SubElement(metadata, "Created")
    created.text = get_utc_time()

    page = etree.SubElement(root, "Page")
    page.attrib["imageFilename"] = image_name
    page.attrib["imageWidth"] = f"{image.shape[1]}"
    page.attrib["imageHeight"] = f"{image.shape[0]}"

    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup")
    ordered_group.attrib["id"] = f"1234_{0}"
    ordered_group.attrib["caption"] = "Regions reading order"

    region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
    region_ref_indexed.attrib["index"] = "0"
    region_ref = "region_main"
    region_ref_indexed.attrib["regionRef"] = region_ref

    text_region = etree.SubElement(page, "TextRegion")
    text_region.attrib["id"] = region_ref
    text_region.attrib["custom"] = "readingOrder {index:0;}"

    text_region_coords = etree.SubElement(text_region, "Coords")
    text_region_coords.attrib["points"] = text_region_bbox

    # print(f"Adding Text to Document: {len(coordinates)}, {len(text_lines)}")
    for i in range(0, len(coordinates)):
        text_coords = get_text_points(coordinates[i])
        # print(f"Unicode: {text_lines[i]}")
        if text_lines != None and len(text_lines) > 0:
            text_region.append(
                get_text_line_block(text_coords, i, unicode_text=text_lines[i])
            )
        else:
            text_region.append(get_text_line_block(text_coords, i, unicode_text=""))

    xmlparse = minidom.parseString(etree.tostring(root))
    prettyxml = xmlparse.toprettyxml()

    return prettyxml
