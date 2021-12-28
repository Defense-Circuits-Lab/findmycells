from abc import ABC, abstractmethod
import os
import numpy as np

class Database():
    def __init__(self, project_root_dir: str):
        self.root_dir = project_root_dir
        self.construct_database()
        
    def construct_database(self):
        # At first, ensure that all subdirectories are present - if not: create the missing ones:
        subdirectories = os.listdir(self.root_dir)
        
        self.microscopy_image_dir = self.root_dir + [elem for elem in subdirectories if 'microscopy' in elem][0] + '/'
        self.rois_to_analyze_dir = self.root_dir + [elem for elem in subdirectories if 'rois' in elem][0] + '/'
        
        try:
            self.preprocessed_images_dir = self.root_dir + [elem for elem in subdirectories if 'preprocessed' in elem][0] + '/' 
        except:
            self.preprocessed_images_dir = self.root_dir + 'preprocessed_images/'
            os.mkdir(self.preprocessed_images_dir)          
        
        # This should become an interactive hub at some point for multiple processing tools (e.g. Intellesis)
        try:
            self.deepflash2_dir = self.root_dir + [elem for elem in subdirectories if 'deepflash2' in elem][0] + '/' 
        except:
            self.deepflash2_dir = self.root_dir + 'deepflash2/'
            os.mkdir(self.deepflash2_dir)
            
        self.create_deepflash2_subdirectories()
        
        
        try:
            self.binary_segmentations_dir = self.root_dir + [elem for elem in subdirectories if 'binary' in elem][0] + '/' 
        except:
            self.binary_segmentations_dir = self.root_dir + 'binary_segmentations/'
            os.mkdir(self.binary_segmentations_dir)
            
        try:
            self.instance_segmentations_dir = self.root_dir + [elem for elem in subdirectories if 'instance' in elem][0] + '/'
        except:
            self.instance_segmentations_dir = self.root_dir + 'instance_segmentations/'
            os.mkdir(self.instance_segmentations_dir)
        
        # Next, construct list of experimental subjects and files - based on microscopy image directory:
        self.file_infos = {'file_id': list(),
                           'original_file_id': list(),
                           'group_id': list(),
                           'subject_id': list(),
                           'microscopy_filetype': list(),
                           'rois_filetype': list(),
                           'cropping_row_indices': list(),
                           'cropping_column_indices': list()}
        
        file_id = 0
        for group in os.listdir(self.microscopy_image_dir):
            for subject in os.listdir(self.microscopy_image_dir + group + '/'):
                for filename in os.listdir(self.microscopy_image_dir + group + '/' + subject + '/'):
                    self.file_infos['file_id'].append(str(file_id).zfill(4))
                    original_file_id = filename[:filename.find('.')]
                    self.file_infos['original_file_id'].append(original_file_id)
                    self.file_infos['group_id'].append(group)
                    self.file_infos['subject_id'].append(subject)
                    self.file_infos['microscopy_filetype'].append(filename[filename.find('.'):])
                    roi_filename = [elem for elem in os.listdir(self.rois_to_analyze_dir + group + '/' + subject + '/') if elem.startswith(original_file_id)][0]
                    self.file_infos['rois_filetype'].append(roi_filename[roi_filename.find('.'):])
                    file_id += 1
                    
        
    def create_subdirectory_structure(self, subdir):
        subjects_per_group = list(set(zip(self.file_infos['group_id'], self.file_infos['subject_id'])))
        for group in set(self.file_infos['group_id']):
            group_dir = subdir + group + '/'
            if os.path.isdir(group_dir) == False:
                os.mkdir(group_dir)
            for subject in [elem[1] for elem in subjects_per_group if elem[0] == group]:
                subject_dir = group_dir + subject + '/'
                if os.path.isdir(subject_dir) == False:
                    os.mkdir(subject_dir)
                    
    def get_file_infos(self, file_id: str):
        file_infos = dict()
        for key in self.file_infos.keys():
            if len(self.file_infos[key]) > 0:
                file_infos[key] = self.file_infos[key][int(file_id)]
                
        return file_infos
    
    
    def update_file_infos(self, file_id: str, key: str, value):
        if len(self.file_infos[key]) < int(file_id):
            for i in range(len(self.file_infos['file_id'])):
                self.file_infos[key].append('not_available')           

        self.file_infos[key][self.file_infos['file_id'].index(file_id)] = value

    
    def import_roi_polygons(self, rois_object):
        if hasattr(self, 'rois_as_shapely_polygons') == False:
            self.rois_as_shapely_polygons = dict()
            
        original_file_id = rois_object.filepath[rois_object.filepath.rfind('/') + 1 : rois_object.filepath.rfind('.')]
        file_id = self.file_infos['file_id'][self.file_infos['original_file_id'].index(original_file_id)]
        
        self.rois_as_shapely_polygons[file_id] = dict()
        for roi_id in rois_object.as_polygons.keys():
            # potential conflict when different rois are used for the individual planes. Update keys e.g. to roi_id_000 and/or plane_id_000 and/or all_planes
            self.rois_as_shapely_polygons[file_id][roi_id] = rois_object.as_polygons[roi_id]
        
        
    def create_deepflash2_subdirectories(self):      
        try:
            self.trained_models_dir = self.deepflash2_dir + [elem for elem in os.listdir(self.deepflash2_dir) if 'models' in elem][0] + '/'
        except:
            self.trained_models_dir = self.deepflash2_dir + 'trained_models/'
            os.mkdir(self.trained_models_dir)
            
        try:
            self.deepflash2_temp_dir = self.deepflash2_dir + [elem for elem in os.listdir(self.deepflash2_dir) if 'temp' in elem][0] + '/'
        except:
            self.deepflash2_temp_dir = self.deepflash2_dir + 'temp/'
            os.mkdir(self.deepflash2_temp_dir)         
        
        
class RGBZStack(ABC): #dependency inversion!
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.isrgb = True
        self.is3d = True
        self.load_zstack()
        self.total_planes = self.as_array.shape[0]
    
    @abstractmethod
    def load_zstack(self):
        """ create self.as_array with structure: [plane, rows, columns, rgb] """
        pass  
        
        

import czifile

class CZIZStack(RGBZStack):
    
    def load_zstack(self):
        self.as_array = czifile.imread(self.filepath)[0, 0, 0]
        
        
        
        
        
        
        
from shapely.geometry import Polygon

class ROIs(ABC): #dependency inversion!
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_roi_coordinates()
        #self.total_rois = 
        #self.as_polygons = 
    
    @abstractmethod
    def load_roi_coordinates(self):
        """ create self.roi_coordinates as dict with structure: {roi_id: [row_coords, column_coords]} """
        pass
            
    def from_array_to_shapely_polygon(self):
        self.as_polygons = dict()
        for roi_id in self.roi_coordinates.keys():
            self.as_polygons[roi_id] = Polygon(np.asarray(list(zip(self.roi_coordinates[roi_id][0], self.roi_coordinates[roi_id][1]))))
            
        
        
import roifile

class ImageJROIs(ROIs):
    
    def load_roi_coordinates(self):
        # implementation for multiple rois (zip file) to be done

        roi_file = roifile.ImagejRoi.fromfile(self.filepath)
        total_rois = 1     
        self.total_rois = total_rois
        self.roi_coordinates = dict()
        for roi_id in range(total_rois):
            row_coords, column_coords = roi_file.coordinates()[:, 1], roi_file.coordinates()[:, 0]
            self.roi_coordinates[str(roi_id).zfill(3)] = [row_coords, column_coords]

            
            
            
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