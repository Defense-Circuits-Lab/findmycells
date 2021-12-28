from abc import ABC, abstractmethod
import os
import numpy as np
from shapely.geometry import Polygon
import roifile

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