from abc import ABC, abstractmethod
import os
import numpy as np

class PreprocessingStrategy(ABC):
    def __init__



for file_id in db.file_infos['file_id']:
    if file_id == '0010':
        file_infos = db.get_file_infos(file_id)
        group, subject = file_infos['group_id'], file_infos['subject_id']
        filename, image_filetype, roi_filetype = file_infos['original_file_id'], file_infos['microscopy_filetype'], file_infos['rois_filetype']
        image_filepath_in = f'{db.microscopy_image_dir}{group}/{subject}/{filename}{image_filetype}' 
        rois_filepath_in = f'{db.rois_to_analyze_dir}{group}/{subject}/{filename}{roi_filetype}'
        
        print('starting to load file_id: ' + file_id)
        microscopy_image = CZIZStack(image_filepath_in)
        rois = ImageJROIs(rois_filepath_in)
        # right now only one ROI that is used for all planes. Ultimately, however, it should also be possible to have plane-specific ROI(s)!
        cropping_strategy = CropStitchingArtefactsCroppingStrategy()
        
        for plane_idx in range(microscopy_image.total_planes):
            plane_id = str(plane_idx).zfill(3)
            image_plane = microscopy_image.as_array[plane_idx].copy() 
            cropped_image = cropping_strategy.crop_image(image_plane)
            cropped_image = convert_12_to_8_bit_rgb_image(cropped_image)
            cropped_image = Image.fromarray(cropped_image, 'RGB')
            image_filepath_out = f'{db.preprocessed_images_dir}{group}/{subject}/{file_id}-{plane_id}.png'
            cropped_image.save(image_filepath_out)
            del image_plane, cropped_image
            print(f'done with plane {plane_id}')
        
        cropping_strategy.adjust_rois(rois)
        rois.from_array_to_shapely_polygon()
        
        db.update_file_infos(file_id, 'cropping_row_indices', (cropping_strategy.lower_row_idx, cropping_strategy.upper_row_idx))
        db.update_file_infos(file_id, 'cropping_column_indices', (cropping_strategy.lower_col_idx, cropping_strategy.upper_col_idx))
        db.import_roi_polygons(rois)
        
        del microscopy_image
        print('done with processing of file_id: ' + file_id)












class CroppingStrategy(ABC):
      
    @abstractmethod
    def crop_image(self, image):
        pass
    
    @abstractmethod
    def adjust_rois(self, roi_file):
        pass
    
    
class CropStitchingArtefactsCroppingStrategy(CroppingStrategy):
    
    def crop_image(self, rgb_image):
        rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image == 0, axis = -1))
        self.lower_row_idx, self.upper_row_idx = self.get_cropping_indices(rows_with_black_px)
        self.lower_col_idx, self.upper_col_idx = self.get_cropping_indices(columns_with_black_px)
        cropped_rgb_image = rgb_image[self.lower_row_idx:self.upper_row_idx, 
                                      self.lower_col_idx:self.upper_col_idx].copy()
        return cropped_rgb_image
           
        
    def adjust_rois(self, roi_object: ROIs):
        for roi_id in roi_object.roi_coordinates.keys():
            roi_object.roi_coordinates[roi_id][0] -= self.lower_row_idx
            roi_object.roi_coordinates[roi_id][1] -= self.lower_col_idx
        
        
    def get_cropping_indices(self, a):
        unique, counts = np.unique(a, return_counts=True)
        indices_with_black_pixels = unique[np.where(counts > 100)]
        lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
        upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]
        return lower_cropping_index, upper_cropping_index