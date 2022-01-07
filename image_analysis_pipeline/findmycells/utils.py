import numpy as np
import os
from skimage.io import imread
from skimage import measure
from shapely.geometry import Polygon


def crop_stitching_artefacts(rgb_image):

    rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image == 0, axis = -1))
    lower_row_idx, upper_row_idx = get_cropping_indices(rows_with_black_px)
    lower_col_idx, upper_col_idx = get_cropping_indices(columns_with_black_px)

    cropped_rgb_image = rgb_image[lower_row_idx:upper_row_idx, 
                                  lower_col_idx:upper_col_idx]
    

    return cropped_rgb_image, lower_row_idx, lower_col_idx


def get_cropping_indices(a):
    unique, counts = np.unique(a, return_counts=True)
    indices_with_black_pixels = unique[np.where(counts > 100)]
    lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
    upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]

    return lower_cropping_index, upper_cropping_index


def convert_12_to_8_bit_rgb_image(rgb_image):
    converted_image = (rgb_image / 4095 * 255).round(0).astype('uint8')
    
    return converted_image


def load_zstack_as_array_from_single_planes(path, file_id, minx=None, maxx=None, miny=None, maxy=None):
    types = list(set([type(minx), type(maxx), type(miny), type(maxy)]))    
    if any([minx, maxx, miny, maxy]):
        if (len(types) == 1) & (types[0] == int):
            cropping = True
        else:
            raise TypeError("'minx', 'maxx', 'miny', and 'maxy' all have to be integers - or None if no cropping has to be done")
    else:
        cropping = False
    filenames = [filename for filename in os.listdir(path) if filename.startswith(file_id)]
    cropped_zstack = list()
    for single_plane_filename in filenames:
        tmp_image = imread(path + single_plane_filename)
        if cropping:
            tmp_image = tmp_image[minx:maxx, miny:maxy]
        cropped_zstack.append(tmp_image.copy())
        del tmp_image
    return np.asarray(cropped_zstack) 


def unpad_x_y_dims_in_2d_array(padded_2d_array, pad_width):
    return padded_2d_array[pad_width:padded_2d_array.shape[0]-pad_width, pad_width:padded_2d_array.shape[1]-pad_width]
    
    
def unpad_x_y_dims_in_3d_array(padded_3d_array, pad_width):
    return padded_3d_array[:, pad_width:padded_3d_array.shape[1]-pad_width, pad_width:padded_3d_array.shape[2]-pad_width]


def get_polygon_from_instance_segmentation(single_plane: np.ndarray, label_id: int) -> Polygon:
    x_dim, y_dim = single_plane.shape
    tmp_array = np.zeros((x_dim, y_dim), dtype='uint8')
    tmp_array[np.where(single_plane == label_id)] = 1
    tmp_contours = measure.find_contours(tmp_array, level = 0)[0]
    return Polygon(tmp_contours)