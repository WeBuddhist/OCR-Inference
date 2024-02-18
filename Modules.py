import os
import cv2
import logging
import json
import shutil
import requests
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from pathlib import Path
from scipy.special import expit
from scipy.signal import find_peaks
from scipy.special import softmax
from fast_ctc_decode import viterbi_search
from Utils import (
    pad_image,
    patch_image,
    unpatch_image,
    resize_image,
    unpatch_prediction,
    generate_line_images,
    prepare_ocr_image,
    prepare_img_patches,
    prepare_tf_ocr_image,
    binarize_line,
    resize_to_width,
    rotate_from_hough,
    back_rotate_lines,
    get_line_bbox_centers,
    get_line_treshold,
    group_bbox_centers,
    group_line_chunks,
)
from Config import COLOR_DICT
from DataClasses import Bbox, LineData, LayoutData

# from KerasModels import Easter2
# from keras.models import Model


class classModernLayoutDetection:
        
        def __init__(self, config_file: str):
            self.config_file = config_file
            self.patch_size = 512
            self.execution_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
            self.inference_session = None

            self.init(self.config_file)


        def init(self, config_file: str):
            if not os.path.isfile(config_file):
                logging.error(f"{config_file} does not exist, exiting.")
            else:
                try:
                    model_dir = os.path.dirname(config_file)
                    _file = open(config_file)
                    json_content = json.loads(_file.read())
                    self.patch_size = int(json_content["input_width"])
                    onnx_model_file = json_content["model"]
                    self.class_dict = json_content["classes"]
                    model_file_path = f"{model_dir}/{onnx_model_file}"
                    self.inference_session = ort.InferenceSession(
                        model_file_path, providers=self.execution_providers
                    )

                    logging.info(f"Loaded model from: {onnx_model_file}")
                
                except BaseException as e:
                    logging.error(f"Failed to initialize Layout Detection: {e}")


class PatchedPerigLayoutDetection:
    """
    Layout Detection specifically optimized for the traditional 'long' Tibetan book format 'perig'.
    The inference runs on the provided onnx model file.
    """

    def __init__(self, config_file: str, mode: str = "cuda") -> None:
        self.config_file = config_file
        self.patch_size = 512
        self.mode = mode
        self.execution_providers = None
        self.layout_inference = None
        self.class_dict = {}

        self._can_run = False
        self.current_layout_data = None

        self.init(self.config_file)

    def init(self, config_file: str):
        if not os.path.isfile(config_file):
            logging.error(f"{config_file} does not exist, exiting.")
        else:
            try:
                model_dir = os.path.dirname(config_file)
                _file = open(config_file)
                json_content = json.loads(_file.read())
                self.patch_size = int(json_content["input_width"])
                onnx_model_file = json_content["model"]
                self.class_dict = json_content["classes"]
                model_file_path = f"{model_dir}/{onnx_model_file}"

                if self.mode == "cuda":
                    execution_providers = ["CUDAExecutionProvider"]
                else:
                    execution_providers = ["CPUExecutionProvider"]

                self.execution_providers = execution_providers

                self.layout_inference = ort.InferenceSession(
                    model_file_path, providers=self.execution_providers
                )

                logging.info(f"Loaded model from: {onnx_model_file}")
                self._can_run = True

            except BaseException as e:
                logging.error(f"Failed to initialize Layout Detection: {e}")

    def predict_image(self, image: np.array):
        padded_img, (pad_x, pad_y) = pad_image(image, self.patch_size, pad_value=0)
        img_patches, y_steps = patch_image(padded_img, patch_size=self.patch_size)

        img_batch = [prepare_img_patches(x) for x in img_patches]
        img_batch = np.array(img_batch)
        img_batch = np.transpose(img_batch, axes=[0, 3, 1, 2])
        img_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        pred_batch = self.layout_inference.run(None, {"input": img_batch})
        pred_batch = pred_batch[0]
        pred_batch = np.transpose(pred_batch, axes=[0, 2, 3, 1])
        pred_batch = softmax(pred_batch, axis=-1)

        unpatched_predition = unpatch_prediction(pred_batch, y_splits=y_steps)
        unpatched_predition = unpatched_predition[
            : unpatched_predition.shape[0] - pad_y,
            : unpatched_predition.shape[1] - pad_x,
        ]
        back_sized_image = cv2.resize(
            unpatched_predition, (image.shape[1], image.shape[0])
        )

        combined_output = np.argmax(back_sized_image, axis=-1)
        combined_output = combined_output.astype(np.uint8)
        combined_output = np.expand_dims(combined_output, axis=-1)

        return back_sized_image, combined_output

    def generate_page_data(
        self, predictions: np.array, angle: float, area_threshold: int = 20, filter_for_border: bool = True
    ) -> np.array:
        # TODO: move the calculation of the text bounding box after the filtering of the lines so that small outlier contours don't extend the box too much
        # TODO: add filtering for the other classes
        # TODO: maybe use cv2.minAreaRect instead of bounding rect?

        pred_images = predictions[:, :, 1]
        pred_lines = predictions[:, :, 2]
        pred_margin = predictions[:, :, 3]
        pred_caption = predictions[:, :, 4]

        pred_images = np.where(pred_images > 200, 255, 0)
        pred_lines = np.where(pred_lines > 200, 255, 0)
        pred_margin = np.where(pred_margin > 200, 255, 0)
        pred_caption = np.where(pred_caption > 200, 255, 0)

        pred_images = pred_images.astype(np.uint8)
        pred_lines = pred_lines.astype(np.uint8)
        pred_margin = pred_margin.astype(np.uint8)
        pred_caption = pred_caption.astype(np.uint8)

        images = []
        text_bboxes = []
        lines = []
        captions = []
        margins = []

        prediction_images = {}

        prediction_height, prediction_width = predictions.shape[:2]

        if np.sum(pred_lines) > 0:
            x, y, w, h = cv2.boundingRect(np.array(pred_lines))
            bbox = Bbox(x, y, w, h)
            text_bboxes.append(bbox)

            contours, _ = cv2.findContours(
                pred_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            """for contour in contours:
                min_rect = cv2.minAreaRect(contour[0])
                box = cv2.boxPoints(min_rect)
                box = np.intp(box)
                lines.append(box)
            """
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)

                if area > area_threshold:
                    filtered_contours.append(contour)

            for _, contour in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(contour)
                x_center = x + (w // 2)
                y_center = y + (h // 2)

                # filter contours that touch the image border
                if filter_for_border:
                    
                    if x == 0 or y == 0 :
                        continue

                if x+w == prediction_width or y+h == prediction_height:
                    continue

                line_data = LineData(
                    contour=contour, bbox=Bbox(x, y, w, h), center=(x_center, y_center)
                )

                lines.append(line_data)

            prediction_images["lines"] = pred_lines
        else:
            prediction_images["lines"] = None

        if np.sum(pred_images) > 0:
            contours, _ = cv2.findContours(
                pred_images, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = Bbox(x, y, w, h)
                images.append(bbox)

            prediction_images["images"] = pred_lines # TODO -> why pred_lines=!?
        else:
            prediction_images["images"] = None

        if np.sum(pred_margin) > 0:
            contours, _ = cv2.findContours(
                pred_margin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = Bbox(x, y, w, h)
                margins.append(bbox)

            prediction_images["margins"] = pred_margin
        else:
            prediction_images["margins"] = None

        if np.sum(pred_caption) > 0:
            contours, _ = cv2.findContours(
                pred_caption, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = Bbox(x, y, w, h)
                captions.append(bbox)

            prediction_images["captions"] = pred_caption
        else:
            prediction_images["captions"] = None

        layout_data = LayoutData(angle,
            images, text_bboxes, lines, captions, margins, predictions=prediction_images
        )

        return layout_data

    def generate_preview(
        self, image: np.array, layout_data: LayoutData, alpha: float = 0.4
    ) -> np.array:
        preview_image = image.copy()
        preview_mask = np.zeros(image.shape, dtype=np.uint8)

        if len(layout_data.lines) > 0:
            color = tuple([int(x) for x in COLOR_DICT["line"].split(",")])

            for line_data in layout_data.lines:
                cv2.drawContours(
                    preview_mask,
                    [line_data.contour],
                    contourIdx=-1,
                    color=color,
                    thickness=-1,
                )

        if len(layout_data.images) > 0:
            color = tuple([int(x) for x in COLOR_DICT["image"].split(",")])

            for bbox in layout_data.images:
                cv2.rectangle(
                    preview_mask,
                    (bbox.x, bbox.y),
                    (bbox.x + bbox.w, bbox.y + bbox.h),
                    color,
                    2,
                )

        if len(layout_data.captions) > 0:
            color = tuple([int(x) for x in COLOR_DICT["caption"].split(",")])

            for bbox in layout_data.captions:
                cv2.rectangle(
                    preview_mask,
                    (bbox.x, bbox.y),
                    (bbox.x + bbox.w, bbox.y + bbox.h),
                    color,
                    2,
                )

        if len(layout_data.margins) > 0:
            color = tuple([int(x) for x in COLOR_DICT["margin"].split(",")])

            for bbox in layout_data.margins:
                cv2.rectangle(
                    preview_mask,
                    (bbox.x, bbox.y),
                    (bbox.x + bbox.w, bbox.y + bbox.h),
                    color,
                    2,
                )

        cv2.addWeighted(preview_mask, alpha, preview_image, 1 - alpha, 0, preview_image)

        return preview_image

    def sort_lines(self, layout_data: LayoutData, threshold: int = 20, calculate: bool = True, group_lines: bool = False):
        # TODO: change input to a list of LineData, so that this function can also run within the line detection-only workflow
        # So return here only the list of LineData and update the layout data outside of this function, maybe keeping the unmerged lines stored temporarily
        if len(layout_data.lines) > 0:
            bbox_centers = get_line_bbox_centers(layout_data)
            
            if calculate:
                line_treshold = get_line_treshold(layout_data.predictions["lines"])
            else:
                line_treshold = threshold
            
            sorted_bbox_centers = group_bbox_centers(
                bbox_centers, line_threshold=line_treshold
            )
            if group_lines:
                new_lines = group_line_chunks(sorted_bbox_centers, layout_data)
                layout_data.lines = new_lines
            self.current_layout_data = layout_data
            return layout_data
        else:
            logging.warning(
                f"Provided layout data does not contain any lines, skipping."
            )

        return layout_data

    def predict(self, image: np.array, area_threshold: int = 20):
        p_img = resize_to_width(image, target_width=2048)

        p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        p_img, angle = rotate_from_hough(p_img)

        _prediction, class_image = self.predict_image(p_img)
        _prediction = cv2.resize(_prediction, (image.shape[1], image.shape[0]))
        class_image = cv2.resize(class_image, (image.shape[1], image.shape[0]))
        p_img = cv2.resize(p_img, (image.shape[1], image.shape[0]))

        _prediction = _prediction.astype(np.uint8)
        layout_data = self.generate_page_data(_prediction, angle, area_threshold=area_threshold)
        self.current_layout_data = layout_data

        preview_img = self.generate_preview(p_img, layout_data)

        return layout_data, preview_img


class PatchedLineDetection:
    """
    Handles layout detection
    Args:
        - config_file: json file with the following parameters:
            - onnx model file
            - image input width and height
            - classes

    """

    def __init__(
        self,
        config_file: str,
        binarize_output: bool = False,
        class_threshold: int = 0.8,
        mode: str = "cpu",
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._patch_size = 512
        self._binarize_output = binarize_output
        self._class_threshold = class_threshold
        self._inference = None
        # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self.mode = mode

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.execution_providers = execution_providers

        self._init()

    def _init(self) -> None:
        _file = open(self._config_file)
        json_content = json.loads(_file.read())
        model_path = Path(self._config_file).parent
        print(f"Assuming model path: {model_path}")

        # TODO: do some exception handling here if the files are not found
        if self.mode == "cuda":
            self._onnx_model_file = os.path.join(model_path, json_content["gpu-model"])
        else:
            self._onnx_model_file = os.path.join(model_path, json_content["cpu-model"])

        self._patch_size = int(json_content["input_width"])  #  that's a little unclean

        if self._onnx_model_file is not None:
            try:
                self._inference = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
            except Exception as error:
                print.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        print(f"Line Inference -> Init(): {self.can_run}")

    def predict(
        self,
        original_image: np.array,
        unpatch_type: int = 0,
        class_threshold: float = 0.7,
        line_kernel: int = 20,
    ) -> np.array:
        image, _ = resize_image(original_image)
        padded_img, (pad_x, pad_y) = pad_image(image, self._patch_size)
        image_patches, y_splits = patch_image(padded_img, self._patch_size)
        image_batch = np.array(image_patches)
        image_batch = image_batch.astype(np.float32)
        image_batch /= 255.0

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])  # make B x C x H xW

        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()
        prediction = np.squeeze(prediction, axis=1)
        prediction = expit(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

        if unpatch_type == 0:
            unpatched_image = unpatch_image(image, pred_list)
        else:
            unpatched_image = unpatch_prediction(image, y_splits)

        cropped_image = unpatched_image[
            : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
        ]
        # back_sized = cv2.resize(cropped_image, (int(cropped_image.shape[1] / resize_factor), int(cropped_image.shape[0] / resize_factor)))
        back_sized_image = cv2.resize(
            cropped_image, (original_image.shape[1], original_image.shape[0])
        )
        back_sized_image = back_sized_image.astype(np.uint8)

        # TODO: remove this into a post-processing module
        (
            rotated_img,
            line_images,
            sorted_contours,
            bbox,
            peaks,
            angle,
        ) = generate_line_images(
            original_image, back_sized_image, line_kernel, self._binarize_output
        )
        if sorted_contours is not None and len(sorted_contours) > 0:
            sorted_contours = back_rotate_lines(original_image, sorted_contours, angle)
        return back_sized_image, rotated_img, line_images, sorted_contours, bbox, peaks


class OnnxInference:
    def __init__(self, config_file, mode: str = "cuda") -> None:
        self.config = config_file
        self._onnx_model_file = None
        self._input_width = 2000
        self._input_height = 80
        self._characters = []
        self._can_run = False
        self.ocr_session = None
        self.mode = mode
        self._init()

    def _init(self) -> None:
        _model_dir = os.path.dirname(self.config)
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())
        self._onnx_model_file = f"{_model_dir}/{json_content['onnx-model']}"

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._squeeze_channel_dim = (
            True if json_content["squeeze_channel_dim"] == "yes" else False
        )
        self._swap_hw = True if json_content["swap_hw"] == "yes" else False
        self._characters = self.get_charset(json_content["charset"])

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=execution_providers
        )

    def get_charset(self, charset: str) -> str:
        return f"ß{charset} "

        # print(f"Squeezing channel dim: {self._squeeze_channel_dim}")

    def run(self, line_images: np.array, replace_tsheg: str = "", convert_to_unicode: bool = False):

        img_batch = np.array(line_images, np.float32)

        if self._squeeze_channel_dim:
            img_batch = np.squeeze(img_batch, axis=1)

        if self._swap_hw:
            img_batch = np.transpose(img_batch, axes=[0, 2, 1])

        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        prediction = ocr_results[0].numpy()
        # print(f"Prediction: {prediction.shape}")

        # this is a little unclear, it basically takes into account, the different
        # ordering of the channels for CRNN and EASTER models. Maybe rename this variable or make this process
        # more explicit

        if not self._swap_hw:
            prediction = np.transpose(prediction, axes=[1, 0, 2])

        predicted_text = []
        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            text, _ = viterbi_search(pred_line, self._characters)

            if len(text) > 0:
                text = text.replace(" ", "")
                text = text.replace("§", replace_tsheg)
                predicted_text.append(text)
            else:
                predicted_text.append("")

        return predicted_text, prediction


import json
import pyewts
import tensorflow as tf
from keras.models import Model
from KerasModels import Easter2, Easter2b
import keras.backend as K
from Utils import prepare_tf_ocr_image


class KerasInference:
    def __init__(self, config_file) -> None:
        self.config_file = config_file
        self._input_width = 2000
        self._input_height = 80
        self._time_steps = 500
        self._characters = []
        self._model = None
        self._converter = pyewts.pyewts()

        self._init()

    def get_model_characters(self, characters: str) -> list[str]:
        characters = [c for c in characters]
        characters.append("[BLK]")
        characters.insert(0, "[UNK]")

        return characters

    def _init(self) -> None:
        _file = open(self.config_file, encoding="utf-8")
        _file_path = os.path.dirname(self.config_file)
        json_content = json.loads(_file.read())

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._architecture = json_content["architecture"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._time_steps = json_content["time_steps"]
        self._characters = self.get_model_characters(json_content["charset"])

        if self._architecture == "easter2b":
            print(f"Loading Easter2b")
            model = Easter2b(
                input_width=self._input_width,
                input_height=self._input_height,
                classes=len(self._characters),
            )
        else:
            print(f"Loading Easter2")
            model = Easter2(
                input_width=self._input_width,
                input_height=self._input_height,
                classes=len(self._characters),
            )

        weights = json_content["h5"]
        model.load_weights(f"{_file_path}/{weights}")
        self._model = Model(
            model.get_layer(self._input_layer).input, model.get_layer(self._output_layer).output
        )

    def run(
        self, line_images: np.array, replace_blank: str = "", wylie_output: bool = False
    ):
        ### todo: make sure each image in line_images has the correct shape 
    
        line_batch = np.array(line_images)
        #print(line_batch.shape[-1])

        if line_batch.shape[-1] == 3:
            line_batch = line_batch[:, :, :, 0]

        print(line_batch.shape)
        prediction = self._model.predict_on_batch(line_images)

        predicted_text = []

        for idx in range(prediction.shape[0]):
            pred_line = prediction[idx, :, :]
            pred_line = np.expand_dims(pred_line, axis=0)

            out = K.get_value(
                K.ctc_decode(
                    pred_line,
                    input_length=np.ones(pred_line.shape[0]) * pred_line.shape[1],
                    greedy=False, beam_width=10
                )[0][0]
            )[0]

            filtered_out = np.delete(out, np.where(out == -1))
            filtered_list = filtered_out.tolist()
            decoded = "".join(self._characters[x] for x in filtered_list)
            decoded = decoded.replace("§", " ")

            if not wylie_output:
                decoded = self._converter.toUnicode(decoded)

            predicted_text.append(decoded)

        return predicted_text


class IIIFDownloader:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._current_download_dir = None

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def get_download_dir(self) -> str:
        return self._current_download_dir

    def get_json(self, link: str):
        response = requests.get(link)
        return response.json()

    def download_data(
        self,
        manifest_data: str,
        work_id: str,
        file_limit: int = 20,
        last_pages: bool = True,
    ):
        if "sequences" in manifest_data:
            seq = manifest_data["sequences"]
            volume_id = seq[0]["@id"].split("bdr:")[1].split("/")[0]

            volume_out = os.path.join(self._output_dir, work_id, volume_id)
            self._current_download_dir = volume_out

            if not os.path.exists(volume_out):
                os.makedirs(volume_out)

            max_images = len(seq[0]["canvases"])

            if max_images > 0:
                if file_limit == 0 or file_limit > max_images or file_limit < 0:
                    file_limit = max_images

                for idx in tqdm(range(file_limit)):
                    if "images" in seq[0]["canvases"][idx]:
                        img_url = seq[0]["canvases"][idx]["images"][0]["resource"][
                            "@id"
                        ]
                        img_name = img_url.split("::")[1].split(".")[0]
                        out_file = f"{volume_out}/{img_name}.jpg"

                        if not os.path.isfile(out_file):
                            res = requests.get(img_url, stream=True)

                            if res.status_code == 200:
                                with open(out_file, "wb") as f:
                                    shutil.copyfileobj(res.raw, f)

                # download also the last images within the range of the specified file-limit
                if last_pages:
                    for idx in tqdm(range(max_images - file_limit, max_images)):
                        if "images" in seq[0]["canvases"][idx]:
                            img_url = seq[0]["canvases"][idx]["images"][0]["resource"][
                                "@id"
                            ]
                            img_name = img_url.split("::")[1].split(".")[0]
                            out_file = f"{volume_out}/{img_name}.jpg"

                            if not os.path.isfile(out_file):
                                res = requests.get(img_url, stream=True)

                                if res.status_code == 200:
                                    with open(out_file, "wb") as f:
                                        shutil.copyfileobj(res.raw, f)

    def download(self, manifest_link: str, file_limit: int = 50):
        file_limit = int(file_limit)
        work_id = manifest_link.split(":")[-1]
        data = self.get_json(manifest_link)

        if not "sequences" in data:
            print("No direkt manifest found, assuming this is a collection manifest")
            manifests = data["manifests"]

            for manifest in manifests:
                manifest_data = self.get_json(manifest["@id"])
                self.download_data(
                    manifest_data,
                    work_id,
                    file_limit=file_limit,
                )
        else:
            self.download_data(data, work_id, file_limit=file_limit)
