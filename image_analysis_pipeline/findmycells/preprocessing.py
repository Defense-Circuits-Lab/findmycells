# Probably has to become an entire module with several submodules, e.g. for cropping, channel splitting, max projections, ...

from abc import ABC, abstractmethod
import os
import numpy as np
from skimage.io import imsave
from shapely.geometry import Polygon
from typing import List, Tuple, Dict
from skimage.io import imsave

from .database import Database
from .microscopyimages import MicroscopyImageLoader
from .rois import ROILoader
from .utils import convert_12_to_8_bit_rgb_image


"""
Next steps:
 - load ROIs (or create one for with shape of the image)
 - make adaptations to database:
   - use Path objects to enable continous integration
   - create sorted list of all preprocessing steps (i.e. preprocessing strategies)
 - adapt main
 - function that saves preprocessed images

 - Minimal preprocessing steps are:
    - save the "unprocessed" microscopy images to the preprocessed_dir
    - load the unprocessed ROIs into the database (create ROI with shape of image if whole image is to be analyzed)
"""

class PreprocessingObject:
    
    def __init__(self, database: Database, file_id: str):
        self.database = database
        self.file_id = file_id
        self.file_info = self.database.get_file_infos(identifier = self.file_id)
        self.preprocessed_image = self.load_microscopy_image()
        self.total_planes = self.preprocessed_image.shape[0]
        if self.preprocessed_image.shape[3] == 3:
            self.is_rgb = True
        else:
            self.is_rgb = False
        self.preprocessed_rois = self.load_rois()

        
    def load_microscopy_image(self) -> np.ndarray:
        microscopy_image_loader = MicroscopyImageLoader(filepath = self.file_info['microscopy_filepath'],
                                                        filetype = self.file_info['microscopy_filetype'])
        return microscopy_image_loader.as_array()
    
    
    def load_rois(self) -> Dict:
        if self.file_info['rois_present'] ==  False:
            message_line0 = 'As of now, it is not supported to not provide a ROI file for each image.\n'
            message_line1 = 'If you would like to quantify the image features in the entire image, please provide a ROI that covers the entire image.\n'
            message_line2 = 'Warning: However, please be aware that this feature was not fully tested yet and will probably cause problems,\n'
            message_line3 = 'specifically if any cropping is used as preprocessing method.'
            error_message = message_line0 + message_line1 + message_line2 + message_line3
            raise NotImplementedError(error_message)
        elif self.file_info['rois_present']:
            roi_loader = ROILoader(filepath = self.file_info['rois_filepath'],
                                   filetype = self.file_info['rois_filetype'])
        return roi_loader.as_dict()
    
    
    def run_all_preprocessing_steps(self):
        for preprocessing_strategy in self.database.preprocessing_strategies:
            self = preprocessing_strategy().run(preprocessing_object = self, step = self.database.preprocessing_strategies.index(preprocessing_strategy))
    
    
    def save_preprocessed_images_on_disk(self):
        for plane_index in range(self.total_planes):
            image = self.preprocessed_image[plane_index].astype('uint8')
            filepath_out = self.database.preprocessed_images_dir.joinpath(f'{self.file_id}-{str(plane_index).zfill(3)}.png')
            imsave(filepath_out, image)

    
    def save_preprocessed_rois_in_database(self):
        self.database.import_rois_dict(file_id = self.file_id, rois_dict = self.preprocessed_rois)
        
    
    def update_database(self):
        updates = dict()
        updates['RGB'] = self.is_rgb
        updates['total_planes'] = self.total_planes
        updates['preprocessing_completed'] = True
        self.database.update_file_infos(file_id = self.file_id, updates = updates)

            

class PreprocessingStrategy(ABC):
    
    @abstractmethod
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        # do preprocessing
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    @abstractmethod
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'PreprocessingStrategyName'
        # Add additional information if neccessary
        database.update_file_infos(file_id = file_id, updates = updates)
        return database


class CropStitchingArtefactsRGB(PreprocessingStrategy):
    
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        self.cropping_indices = self.determine_cropping_indices_for_entire_zstack(preprocessing_object = preprocessing_object)
        preprocessing_object.preprocessed_image = self.crop_rgb_zstack(zstack = preprocessing_object.preprocessed_image)
        preprocessing_object.preprocessed_rois = self.adjust_rois(rois_dict = preprocessing_object.preprocessed_rois)
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
        
    
    def get_cropping_indices(self, a, min_black_px_stretch: int=100):
        unique, counts = np.unique(a, return_counts=True)
        indices_with_black_pixels = unique[np.where(counts >= min_black_px_stretch)]
        if indices_with_black_pixels.shape[0] > 0: 
            if np.where(np.diff(indices_with_black_pixels) > 1)[0].shape[0] > 0:
                lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
                upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]
            else:
                if indices_with_black_pixels[0] == 0:
                    lower_cropping_index = indices_with_black_pixels[-1]
                    upper_cropping_index = a.shape[0] - 1
                else:
                    lower_cropping_index = 0
                    upper_cropping_index = indices_with_black_pixels[0]
        else:
            lower_cropping_index = 0
            upper_cropping_index = a.shape[0] - 1
        return lower_cropping_index, upper_cropping_index 
                                                  
    
    def determine_cropping_indices_for_entire_zstack(self, preprocessing_object: PreprocessingObject) -> Dict:
        for plane_index in range(preprocessing_object.total_planes):
            rgb_image_plane = preprocessing_object.preprocessed_image[plane_index]
            rows_with_black_px, columns_with_black_px = np.where(np.all(rgb_image_plane == 0, axis = -1))
            lower_row_idx, upper_row_idx = self.get_cropping_indices(rows_with_black_px)
            lower_col_idx, upper_col_idx = self.get_cropping_indices(columns_with_black_px)  
            if plane_index == 0:
                min_lower_row_cropping_idx, max_upper_row_cropping_idx = lower_row_idx, upper_row_idx
                min_lower_col_cropping_idx, max_upper_col_cropping_idx = lower_col_idx, upper_col_idx
            else:
                if lower_row_idx > min_lower_row_cropping_idx:
                    min_lower_row_cropping_idx = lower_row_idx
                if upper_row_idx < max_upper_row_cropping_idx:
                    max_upper_row_cropping_idx = upper_row_idx
                if lower_col_idx > min_lower_col_cropping_idx:
                    min_lower_col_cropping_idx = lower_col_idx
                if upper_col_idx < max_upper_col_cropping_idx:
                    max_upper_col_cropping_idx = upper_col_idx  
        cropping_indices = {'lower_row_cropping_idx': min_lower_row_cropping_idx,
                            'upper_row_cropping_idx': max_upper_row_cropping_idx,
                            'lower_col_cropping_idx': min_lower_col_cropping_idx,
                            'upper_col_cropping_idx': max_upper_col_cropping_idx}
        return cropping_indices
    
    
    def crop_rgb_zstack(self, zstack: np.ndarray) -> np.ndarray:
        min_row_idx = self.cropping_indices['lower_row_cropping_idx']
        max_row_idx = self.cropping_indices['upper_row_cropping_idx']
        min_col_idx = self.cropping_indices['lower_col_cropping_idx']
        max_col_idx = self.cropping_indices['upper_col_cropping_idx']
        return zstack[:, min_row_idx:max_row_idx, min_col_idx:max_col_idx, :]
                                                           
    
    def adjust_rois(self, rois_dict: Dict) -> Dict:
        lower_row_idx = self.cropping_indices['lower_row_cropping_idx']
        lower_col_idx = self.cropping_indices['lower_col_cropping_idx']
        for plane_identifier in rois_dict.keys():
            for roi_id in rois_dict[plane_identifier].keys():
                adjusted_row_coords = [coordinates[0] - lower_row_idx for coordinates in rois_dict[plane_identifier][roi_id].boundary.coords[:]]
                adjusted_col_coords = [coordinates[1] - lower_col_idx for coordinates in rois_dict[plane_identifier][roi_id].boundary.coords[:]]
                rois_dict[plane_identifier][roi_id] = Polygon(np.asarray(list(zip(adjusted_row_coords, adjusted_col_coords))))
        return rois_dict
    
    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'CropStitchingArtefactsRGB'
        updates['cropping_row_indices'] = (self.cropping_indices['lower_row_cropping_idx'], self.cropping_indices['upper_row_cropping_idx'])
        updates['cropping_column_indices'] = (self.cropping_indices['lower_col_cropping_idx'], self.cropping_indices['upper_col_cropping_idx'])       
        database.update_file_infos(file_id = file_id, updates = updates)
        return database
        
        
                                                           
                                                           
class ConvertFrom12To8BitRGB(PreprocessingStrategy):
    
    def run(self, preprocessing_object: PreprocessingObject, step: int) -> PreprocessingObject:
        for plane_index in range(preprocessing_object.total_planes):
            preprocessing_object.preprocessed_image[plane_index] = self.convert_rgb_image(rgb_image = preprocessing_object.preprocessed_image[plane_index])
        preprocessing_object.database = self.update_database(database = preprocessing_object.database, file_id = preprocessing_object.file_id, step = step)
        return preprocessing_object
    
    def convert_rgb_image(self, rgb_image: np.ndarray) -> np.ndarray:
        converted_image = (rgb_image / 4095 * 255).round(0).astype('uint8')
        return converted_image

    def update_database(self, database: Database, file_id: str, step: int) -> Database:
        updates = dict()
        updates[f'preprocessing_step_{str(step).zfill(2)}'] = 'ConvertFrom12To8BitRGB'    
        database.update_file_infos(file_id = file_id, updates = updates)
        return database
        





# Old version starts here:

"""

How this is structured:

- PreprocessingObject:
    - image-roi-pair that will be preprocessed
    
- PreprocessingMethods:
    - where the processing is actually done
    - e.g. CropStitchingArtefacts()
    
- PreprocessingStrategies:
    - interface between Preprocessor, PreprocessingObject, and PreprocessingMethods
    - e.g. CroppingAsPreprocessingStrategy() [which would actually works with any PreprocessingMethod of type CroppingMethod]

- Preprocessor:
    - prepares and initializes the preprocessing (via PreprocessingStrategies)
    - serves as interface for main

"""
"""
class PreprocessingObject:
    
    def __init__(self, file_info: dict):
        self.file_info = file_info
        self.microscopy_image = None
        self.preprocessed_image = None
        self.roi_file = None
        self.preprocessed_roi_file = None


class PreprocessingMethod(ABC):
    
    @property
    @abstractmethod
    def processsing_strategy(self):
        pass
    
    @property
    @abstractmethod
    def method_category(self):
        pass
    
    @property
    @abstractmethod
    def method_info(self):
        pass

    
class PreprocessingStrategy(ABC):
    
    @abstractmethod
    def run(self, preproobj: PreprocessingObject, prepro_method_obj: PreprocessingMethod, database: Database) -> Tuple[PreprocessingObject, Database]:
        pass

    

class CroppingAsPreprocessingStrategy(PreprocessingStrategy):
    
    def run(self, preproobj: PreprocessingObject, prepro_method_obj: PreprocessingMethod, database: Database) -> Tuple[PreprocessingObject, Database]:
        file_id = preproobj.file_info['file_id']
        updates = dict()
        
        if preproobj.preprocessed_image == None:
            if preproobj.microscopy_image == None:
                # here we should actually determine the exact type and initiate a corresponding object - but for now, we stick with czi files:
                preproobj.microscopy_image = CZIZStack(preproobj.file_info['microscopy_filepath'])
                updates['RGB'] = preproobj.microscopy_image.isrgb
                updates['total_image_planes'] = preproobj.microscopy_image.total_planes
        
        if preproobj.preprocessed_roi_file == None:
            if preproobj.roi_file == None:
                preproobj.roi_file = ImageJROIs(preproobj.file_info['rois_filepath'])
        
        cropping_strategy = prepro_method_obj
        updates['cropping_method'] = cropping_strategy.method_info
        
        for plane_idx in range(preproobj.microscopy_image.total_planes):
            plane_id = str(plane_idx).zfill(3)
            image_plane = preproobj.microscopy_image.as_array[plane_idx].copy()
            cropped_image = cropping_strategy.crop_image(image_plane)        
            if plane_idx == 0:
                min_lower_row_cropping_idx, max_upper_row_cropping_idx = cropping_strategy.lower_row_idx, cropping_strategy.upper_row_idx
                min_lower_col_cropping_idx, max_upper_col_cropping_idx = cropping_strategy.lower_col_idx, cropping_strategy.upper_col_idx
            else:
                if cropping_strategy.lower_row_idx > min_lower_row_cropping_idx:
                    min_lower_row_cropping_idx = cropping_strategy.lower_row_idx
                if cropping_strategy.upper_row_idx < max_upper_row_cropping_idx:
                    max_upper_row_cropping_idx = cropping_strategy.upper_row_idx
                if cropping_strategy.lower_col_idx > min_lower_col_cropping_idx:
                    min_lower_col_cropping_idx = cropping_strategy.lower_col_idx
                if cropping_strategy.upper_col_idx < max_upper_col_cropping_idx:
                    max_upper_col_cropping_idx = cropping_strategy.upper_col_idx
        
        preprocessed_image_planes = list()
        for plane_idx in range(preproobj.microscopy_image.total_planes):
            plane_id = str(plane_idx).zfill(3)
            image_plane = preproobj.microscopy_image.as_array[plane_idx].copy() 
            cropped_image = image_plane[min_lower_row_cropping_idx:max_upper_row_cropping_idx, min_lower_col_cropping_idx:max_upper_col_cropping_idx]
            # The following steps [1: type conversion (X-bit to 8-bit) and 2: saving] should become own PreprocessingStrategies
            # cropped images should be stored in preproobj as "preprocessed_image" attribute
            # This would enable further PreprocessingStrategies to continue from their
            cropped_image = convert_12_to_8_bit_rgb_image(cropped_image)
            cropped_image = Image.fromarray(cropped_image, 'RGB')
            image_filepath_out = f'{database.preprocessed_images_dir}{file_id}-{plane_id}.png'
            cropped_image.save(image_filepath_out)
            preprocessed_image_planes.append(image_plane.copy())
            del image_plane, cropped_image
            print(f'done with plane {plane_id}')

        # make sure cropping indices were identical for each image plane:
        cropping_strategy.lower_row_idx = min_lower_row_cropping_idx
        cropping_strategy.upper_row_idx = max_upper_row_cropping_idx
        cropping_strategy.lower_col_idx = min_lower_col_cropping_idx
        cropping_strategy.upper_col_idx = max_upper_col_cropping_idx
        cropping_strategy.adjust_rois(preproobj.roi_file)
        preproobj.roi_file.from_array_to_shapely_polygon()
        updates['cropping_row_indices'] = (cropping_strategy.lower_row_idx, cropping_strategy.upper_row_idx)
        updates['cropping_column_indices'] = (cropping_strategy.lower_col_idx, cropping_strategy.upper_col_idx)
        database.import_roi_polygons(preproobj.roi_file)
        
        # update preprocessingobject and database when everything is finished:
        preproobj.preprocessed_image = np.asarray(preprocessed_image_planes)
        preproobj.preprocessed_roi_file = preproobj.roi_file.as_polygons

        if len(updates.keys()) > 0:
            for key, value in updates.items():
                if key not in database.file_infos.keys():
                    database.add_new_key_to_file_infos(key)
                database.update_file_infos(file_id, key, value)
                
        del updates, preprocessed_image_planes

        return preproobj, database   

    

class CroppingMethod(PreprocessingMethod):
    processsing_strategy = CroppingAsPreprocessingStrategy()
    method_category = 'Cropping'
         
    @abstractmethod
    def crop_image(self, image):
        pass
    
    @abstractmethod
    def adjust_rois(self, roi_file):
        pass    

    
class CropStitchingArtefacts(CroppingMethod):
    method_info = 'CropStitchingArtefacts'
    
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
        if indices_with_black_pixels.shape[0] > 0: #changed
            if np.where(np.diff(indices_with_black_pixels) > 1)[0].shape[0] > 0:
                lower_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0]][0] + 1
                upper_cropping_index = indices_with_black_pixels[np.where(np.diff(indices_with_black_pixels) > 1)[0] + 1][0]
            else:
                if indices_with_black_pixels[0] == 0:
                    lower_cropping_index = indices_with_black_pixels[-1]
                    upper_cropping_index = a.shape[0] - 1
                else:
                    lower_cropping_index = 0
                    upper_cropping_index = indices_with_black_pixels[0]
        else:
            lower_cropping_index = 0
            upper_cropping_index = a.shape[0] - 1
        return lower_cropping_index, upper_cropping_index     

    
class Preprocessor:


    def __init__(self, file_ids, database: Database):
        self.database = database
        self.file_ids = file_ids
        self.file_info_dicts = self.get_file_info_dicts()
        self.preprocessing_steps = self.get_preprocessing_steps()        
    
    def get_file_info_dicts(self) -> List[dict]:
        file_info_dicts = list() 
        for file_id in self.file_ids:
            file_info_dicts.append(self.database.get_file_infos(file_id))
        
        return file_info_dicts
    
    
    def get_preprocessing_steps(self) -> List:
        prepro_configs = self.database.preprocessing_configs
        preprocessing_steps = [(elem, prepro_configs[elem]['ProcessingStrategy'], prepro_configs[elem]['ProcessingMethod']) for elem in prepro_configs.keys()]
        preprocessing_steps.sort(key=lambda elem: elem[0])
        
        return preprocessing_steps
        
     

    def run_individually(self) -> Database:

        for file_info in self.file_info_dicts:
            if file_info['preprocessing_completed'] != True:
                prepro_obj = PreprocessingObject(file_info)
                for _, prepro_strategy_obj, prepro_method_obj in self.preprocessing_steps:
                    prepro_obj, self.database = prepro_strategy_obj.run(prepro_obj, prepro_method_obj, self.database)
                self.database.update_file_infos(file_info['file_id'], 'preprocessing_completed', True)
                del prepro_obj
        
        return self.database
"""