import numpy as np

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