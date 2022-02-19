from abc import ABC, abstractmethod
import os
import numpy as np
from shapely.geometry import Polygon
import roifile
from pathlib import Path
from typing import Dict


class ROIReader(ABC):

    @abstractmethod
    def read(self, filepath: Path) -> Dict:
        """ 
        Return the roi(s) as shapely.geometry.Polygon(s) in a nested dictionary with structure: {plane_id: {roi_id: Polygon}}
        In case plane-specific ROIs are required / requested at some point, 
        having the additional level that enables the reference to plane_id(s) should foster the implementation.
        The current implementation, however, only supports the use of ROIs for all planes - the corresponding plane_id is hence: 'all_planes'
        Ultimately, this file_id specific dictionary can then be integrated into the 'rois_as_shapely_polygons' attribute of the database.
        
        Note: If multiple ROIs are used for one image, the individual ROIs must be named properly in the ROIManager-Tool in ImageJ.
              For instance, if images of the hippocampus are investigated & they can contain images of the DG, CA3 and CA1, 
              the corresponding ROIs that mark the respective area have to be named consistenly for all .zip files. This makes it possible, 
              that findmycells can handle the analysis even if not all ROIs are present for each image, e.g. for some files only DG and CA3.
        """
        pass
    
class ImageJROIReader(ROIReader):
    
    def read(self, filepath: Path) -> Dict:
        if filepath.name.endswith('.roi'):
            loaded_rois = [roifile.ImagejRoi.fromfile(filepath)]
        elif filepath.name.endswith('.zip'):
            loaded_rois = roifile.ImagejRoi.fromfile(filepath)
        else:
            filetype = filepath.name[filepath.name.find('.'):]
            raise ValueError(f'ImageJROIReader cannot handle files of type {filetype}')
        
        # In case plane-specific ROIs are required / requested at some point, 
        # having this additional level that enables the reference to image planes
        # should foster the implementation.
        rois_as_shapely_polygons = {'all_planes': dict()}
        roi_count = len(loaded_rois)
        for roi_index in range(roi_count):
            row_coords = loaded_rois[roi_index].coordinates()[:, 1]
            col_coords = loaded_rois[roi_index].coordinates()[:, 0]
            if roi_count > 1:
                rois_as_shapely_polygons['all_planes'][loaded_rois[roi_index].name] = Polygon(np.asarray(list(zip(row_coords, col_coords))))
            else:
                rois_as_shapely_polygons['all_planes'][str(roi_index).zfill(3)] = Polygon(np.asarray(list(zip(row_coords, col_coords))))     
        
        return rois_as_shapely_polygons


class ROILoader:
    
    def __init__(self, filepath: Path, filetype: str):
        self.filepath = filepath
        self.reader = self.determine_reader(filetype = filetype)
    
    def determine_reader(self, filetype: str) -> ROIReader:
        if filetype in ['.roi', '.zip']:
            reader = ImageJROIReader()
        else:
            message_part1 = 'The roi-file format you are trying to load is not implemented yet.'
            message_part2 = 'Please consider raising an issue in our GitHub repository!'
            full_message = message_part1 + message_part2
            raise ValueError(full_message)
        return reader
    
    def as_dict(self):
        return self.reader.read(filepath = self.filepath)