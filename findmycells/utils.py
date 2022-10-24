import numpy as np
import os
from pathlib import Path
from skimage.io import imread
from skimage import measure
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


def listdir_nohidden(path: Path) -> List:
    return [f for f in os.listdir(path) if f.startswith('.') == False]


def crop_stitching_artefacts(rgb_image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image == 0, axis = -1))
    lower_row_idx, upper_row_idx = get_cropping_indices(rows_with_black_px)
    lower_col_idx, upper_col_idx = get_cropping_indices(columns_with_black_px)
    cropped_rgb_image = rgb_image[lower_row_idx:upper_row_idx, 
                                  lower_col_idx:upper_col_idx]
    return cropped_rgb_image, lower_row_idx, lower_col_idx


def get_cropping_indices(a: np.ndarray) -> Tuple[int, int]:
    unique, counts = np.unique(a, return_counts=True)
    indices_with_black_pixels = unique[np.where(counts > 100)]
    lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
    upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]
    return lower_cropping_index, upper_cropping_index


def convert_12_to_8_bit_rgb_image(rgb_image: np.ndarray) -> np.ndarray:
    converted_image = (rgb_image / 4095 * 255).round(0).astype('uint8')
    return converted_image


def load_zstack_as_array_from_single_planes(path: Path, file_id: str, 
                                            minx: Optional[int]=None, maxx: Optional[int]=None, 
                                            miny: Optional[int]=None, maxy: Optional[int]=None) -> np.ndarray:
    types = list(set([type(minx), type(maxx), type(miny), type(maxy)]))    
    if any([minx, maxx, miny, maxy]):
        if (len(types) == 1) & (types[0] == int):
            cropping = True
        else:
            raise TypeError("'minx', 'maxx', 'miny', and 'maxy' all have to be integers - or None if no cropping has to be done")
    else:
        cropping = False
    filenames = [filename for filename in listdir_nohidden(path) if filename.startswith(file_id)]
    cropped_zstack = list()
    for single_plane_filename in filenames:
        tmp_image = imread(path.joinpath(single_plane_filename))
        if cropping:
            tmp_image = tmp_image[minx:maxx, miny:maxy]
        cropped_zstack.append(tmp_image.copy())
        del tmp_image
    return np.asarray(cropped_zstack) 


def unpad_x_y_dims_in_2d_array(padded_2d_array: np.ndarray, pad_width: int) -> np.ndarray:
    return padded_2d_array[pad_width:padded_2d_array.shape[0]-pad_width, pad_width:padded_2d_array.shape[1]-pad_width]
    
    
def unpad_x_y_dims_in_3d_array(padded_3d_array: np.ndarray, pad_width: int) -> np.ndarray:
    return padded_3d_array[:, pad_width:padded_3d_array.shape[1]-pad_width, pad_width:padded_3d_array.shape[2]-pad_width]


def get_polygon_from_instance_segmentation(single_plane: np.ndarray, label_id: int) -> Polygon:
    x_dim, y_dim = single_plane.shape
    tmp_array = np.zeros((x_dim, y_dim), dtype='uint8')
    tmp_array[np.where(single_plane == label_id)] = 1
    tmp_contours = measure.find_contours(tmp_array, level = 0)[0]
    roi = Polygon(tmp_contours)
    if roi.is_valid == False:
        roi = make_valid(roi)
    return roi


def get_cropping_box_arround_centroid(roi: Polygon, half_window_size: int) -> Tuple[int, int, int, int]:
    centroid_x, centroid_y = round(roi.centroid.x), round(roi.centroid.y)
    cminx, cmaxx = centroid_x - half_window_size, centroid_x + half_window_size
    cminy, cmaxy = centroid_y - half_window_size, centroid_y + half_window_size
    return cminx, cmaxx, cminy, cmaxy


def get_color_code(label_ids: List, for_rgb: bool=False) -> Dict:
    n_label_ids = len(label_ids)
    colormixer = plt.cm.rainbow(np.linspace(0, 1, n_label_ids))
    color_code = dict()
    for idx in range(n_label_ids):
        if for_rgb:
            color_code[label_ids[idx]] = {'red': colormixer[idx][0],
                                          'green': colormixer[idx][1],
                                          'blue': colormixer[idx][2]}
        else:
            color_code[label_ids[idx]] = colormixer[idx]
    return color_code


def get_rgb_color_code_for_3D(zstack: np.ndarray) -> Dict:
    label_ids = list(np.unique(zstack))
    if 0 in label_ids:
        label_ids.remove(0)
    if 0.0 in label_ids:
        label_ids.remove(0)
    color_code = get_color_code(label_ids, for_rgb=True)

    red_colors = np.zeros(zstack.shape)
    green_colors = np.zeros(zstack.shape)
    blue_colors = np.zeros(zstack.shape)

    for label_id in label_ids:
        red_colors[np.where(zstack == label_id)] = color_code[label_id]['red']
        green_colors[np.where(zstack == label_id)] = color_code[label_id]['green']
        blue_colors[np.where(zstack == label_id)] = color_code[label_id]['blue']

    rgb_color_code = np.zeros(zstack.shape + (3,))
    rgb_color_code[..., 0] = red_colors
    rgb_color_code[..., 1] = green_colors
    rgb_color_code[..., 2] = blue_colors

    return rgb_color_code