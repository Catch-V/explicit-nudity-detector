import cv2
import extcolors
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

from typing import Tuple, List, Any
from cv2.typing import MatLike



def kpt2part_array(kpt_data) -> Tuple[List[List[Any]], List[Any]]:
    '''
    Converts keypoints data into arrays representing body parts.

    :param kpt_data: Keypoints data.
    :return: Tuple containing lists of hands and body parts.
    '''
    hands = [[kpt_data[i][9], kpt_data[i][10]] for i in range(len(kpt_data))]
    body = [np.array([kpt_data[i][5], kpt_data[i][6], kpt_data[i][12], kpt_data[i][11]]) for i in range(len(kpt_data))]
    return hands, body


def part_array2color(img: np.ndarray, part: List[int]) -> np.ndarray:
    '''
    Converts a part array to the corresponding color in the image.

    :param img: Input image (np.ndarray), the image from which color is extracted.
    :param part: Part array containing coordinates (List[int]) representing a point in the image.
    :return: Color (np.ndarray) corresponding to the given part array.
    '''
    print(part)
    color = img[part[1]][part[0]]
    return color


def masked_thorax(img: np.ndarray, body: np.ndarray) -> np.ndarray:
    '''
    Generates a masked image focusing on the thorax region based on the provided body polygon.

    :param img: Input image (MatLike), the image to be masked.
    :param body: Array representing the polygonal thorax region (np.ndarray).
    :return: Masked image (MatLike) focusing on the thorax region.
    '''
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [body], (255, 255, 255))
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img



def masked_person(img: np.ndarray, body_array: np.ndarray) -> np.ndarray:
    '''
    Generates a mask for the person in the image based on the provided body coordinates.

    :param img: Input image (MatLike), the image for which the mask is generated.
    :param body_array: Array of body coordinates (np.ndarray), specifying the body parts of the person.
    :return: Generated mask (MatLike) representing the person in the image.
    '''
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    print(mask.shape)
    for body in body_array:
        mask[body[1]][body[0]] = 255
    return mask


def extract(img: Any, color_tolerance: int = 50, visualize: bool = False) -> Tuple[Any, Any]:
    '''
    Extracts colors and their pixel counts from the image.

    :param img: Input image (MatLike), the image from which colors are extracted.
    :param color_tolerance: Tolerance for color extraction (int), defaults to 50.
    :param visualize: Whether to visualize the color distribution (bool), defaults to False.
    :return: Tuple containing extracted colors and their pixel counts.
    '''
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pill_img = Image.fromarray(img)
    else:
        pill_img = img

    colors, pixel_count = extcolors.extract_from_image(pill_img, color_tolerance)
    
    if visualize:
        pill_img.show()
        # Visualize the color distribution
        rgb_values, counts = zip(*colors)
        hex_colors = ['#%02x%02x%02x' % rgb for rgb in rgb_values]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(colors)), counts, color=hex_colors)
        plt.xlabel('Color Index')
        plt.ylabel('Counts')
        plt.title('RGB Color Distribution')
        plt.legend(hex_colors, title='RGB Values')
        plt.show()
        
    return colors, pixel_count

def all_img_collecter(file_path: str) -> List[str]:
    '''
    Collects all image files with specified extensions in the given directory path.

    :param file_path: Path to the directory containing image files (str).
    :return: List of sorted paths to the collected image files (List[str]).
    '''
    file_types = ['jpg', 'png', 'jpeg', 'webp']
    files = sorted([file for file_type in file_types for file in glob.glob(os.path.join(file_path, f"*.{file_type}"))])
    return files

def show_image(image, axis_label='off'):
    '''
    Displays the given image.

    :param image: Image to be displayed, either as a NumPy array or a PIL Image.
    :return: None.
    '''
    if isinstance(image, np.ndarray):
        # Display the image using Matplotlib
        plt.imshow(image)
        plt.axis(axis_label)  # Turn off axis labels
        plt.show()
    elif isinstance(image, Image.Image):
        # Display the image using PIL's display function
        image.show()
    else:
        print("Unsupported image type. Please provide either a NumPy array or a PIL image.")

