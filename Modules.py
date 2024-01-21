import os
import cv2
import json
import shutil
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
import onnxruntime as ort
from scipy.special import expit
from scipy.signal import find_peaks
from fast_ctc_decode import viterbi_search
from Utils import (
    pad_image,
    patch_image,
    unpatch_image,
    resize_image,
    unpatch_prediction,
    generate_line_images,
    prepare_ocr_image,
    prepare_tf_ocr_image,
    binarize_line,
)

#from KerasModels import Easter2
#from keras.models import Model


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
        rotated_image, line_images, sorted_contours, bbox, peaks = generate_line_images(
            original_image, back_sized_image, line_kernel, self._binarize_output
        )
        return back_sized_image, rotated_image, line_images, sorted_contours, bbox, peaks


class OCRInference:
    def __init__(self, config_file, mode: str = "cpu") -> None:
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
        _file = open(self.config, encoding="utf-8")
        json_content = json.loads(_file.read())

        if self.mode == "cuda":
            self._onnx_model_file = json_content["gpu-model"]
        else:
            self._onnx_model_file = json_content["cpu-model"]

        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._input_layer = json_content["input_layer"]
        self._output_layer = json_content["output_layer"]
        self._squeeze_channel_dim = (
            True if json_content["squeeze_channel_dim"] == "yes" else False
        )
        self._swap_hw = True if json_content["swap_hw"] == "yes" else False
        self._characters = json_content["charset"]

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=execution_providers
        )

        # print(f"Squeezing channel dim: {self._squeeze_channel_dim}")

    def run(self, line_images: list, replace_blank: str = ""):
        line_tresh = 10  # temporary value, TODO: turn into width and height check
        line_images = [x for x in line_images if x.shape[0] > line_tresh]

        img_batch = [
            prepare_ocr_image(
                x, target_width=self._input_width, target_height=self._input_height
            )
            for x in line_images
        ]
        img_batch = np.array(img_batch, np.float32)

        if self._squeeze_channel_dim:
            img_batch = np.squeeze(img_batch, axis=1)

        if self._swap_hw:
            img_batch = np.transpose(img_batch, axes=[0, 2, 1])
        

        ort_batch = ort.OrtValue.ortvalue_from_numpy(img_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        prediction = ocr_results[0].numpy()
        #print(f"Prediction: {prediction.shape}")
        
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
                text = text.replace("§", replace_blank)
                predicted_text.append(text)
            else:
                predicted_text.append("")

        return predicted_text, prediction


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

    def download_data(self, manifest_data: str, work_id: str, file_limit: int = 20, last_pages: bool = True):
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
                    for idx in tqdm(range(max_images-file_limit, max_images)):
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