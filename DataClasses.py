import numpy as np
from dataclasses import dataclass

@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class LineData:
    contour: list
    bbox: Bbox
    center: tuple[int, int]

@dataclass
class PerigPrediction:
    images: np.array
    lines: np.array
    captions: np.array
    margins: np.array

@dataclass
class LayoutData:
    images: list[Bbox]
    text_bboxes: list[Bbox]
    lines: list[LineData]
    captions: list[Bbox]
    margins: list[Bbox]
    predictions: dict