import cv2
import extcolors
import numpy as np

from PIL import Image

from typing import Tuple, List, Any
from cv2.typing import MatLike



def kpt2part_array(kpt_data) -> Tuple[List[List[Any]], list[Any]]:
    hands = [[kpt_data[i][9], kpt_data[i][10]] for i in range(len(kpt_data))]
    body = [np.array([kpt_data[i][5], kpt_data[i][6], kpt_data[i][12], kpt_data[i][11]]) for i in range(len(kpt_data))]
    return hands, body


def part_array2color(img: np.ndarray, part: List[int]) -> np.ndarray[Any, Any]:
    print(part)
    color = img[part[1]][part[0]]
    return color


def masked_thorax(img: MatLike, body: np.ndarray) -> MatLike:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [body], (255, 255, 255))
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def masked_person(img: MatLike, body_array: np.ndarray) -> MatLike:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    print(mask.shape)
    for body in body_array:
        mask[body[1]][body[0]] = 255
    return mask


def extract(img: MatLike) -> tuple[Any, Any]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pill_img = Image.fromarray(img)
    colors, pixel_count = extcolors.extract_from_image(pill_img, 50)
    return colors, pixel_count
