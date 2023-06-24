# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/04_readers_02_rois.ipynb.

# %% auto 0
__all__ = ['ROIReaders', 'ImageJROIReader']

# %% ../../nbs/api/04_readers_02_rois.ipynb 3
from typing import Dict, List, Any
from pathlib import PosixPath
import numpy as np
from shapely.geometry import Polygon
import roifile

from ..core import DataReader

# %% ../../nbs/api/04_readers_02_rois.ipynb 4
class ROIReaders(DataReader):
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
    
    def assert_correct_output_format(self, output: Dict[str, Dict[str, Polygon]]) -> None:
        assert type(output) == dict, 'The overall type of the returned data is not a dictionary!'
        for plane_id, nested_dict in output.items():
            assert type(plane_id) == str, 'Not all keys of the constructed ROI dictrionary are strings!'
            assert type(nested_dict) == dict, 'Not all elements in the constructed ROI dictionary are nested dictionaries!'
            for roi_id, polygon in output[plane_id].items():
                assert type(roi_id) == str, 'Not all assigned ROI-IDs are strings!'
                assert type(polygon) == Polygon, 'Not all loaded ROIs were successfully converted into Polygon objects!'

# %% ../../nbs/api/04_readers_02_rois.ipynb 5
class ImageJROIReader(ROIReaders):
    
    
    @property
    def readable_filetype_extensions(self) -> List[str]:
        return ['.roi', '.zip']
    
    
    def read(self,
             filepath: PosixPath, # filepath to the roi file
             reader_configs: Dict # the project database
            ) -> Dict[str, Dict[str, Polygon]]: # nested dictionaries of shapely polygons: {plane_id: {roi_id: Polygon}}
        if filepath.suffix == '.roi':
            loaded_rois = [roifile.ImagejRoi.fromfile(filepath)]
        else: # it´s a .zip file:
            loaded_rois = roifile.ImagejRoi.fromfile(filepath)
        rois_as_shapely_polygons = {'all_planes': {}} # plane specific ROIs are not yet supported, but this structure would allow it
        roi_count = len(loaded_rois)
        for idx in range(roi_count):
            row_coords = loaded_rois[idx].coordinates()[:, 1]
            col_coords = loaded_rois[idx].coordinates()[:, 0]
            assert (len(row_coords) > 2) & (len(col_coords) > 2), f"Can't draw a roi from file {loaded_rois[idx].name}, as it has less than 3 coordinates!"
            if reader_configs['load_roi_ids_from_file'] == True:
                rois_as_shapely_polygons['all_planes'][loaded_rois[idx].name] = Polygon(np.asarray(list(zip(row_coords, col_coords))))
            else:
                rois_as_shapely_polygons['all_planes'][str(idx).zfill(3)] = Polygon(np.asarray(list(zip(row_coords, col_coords))))            
        return rois_as_shapely_polygons
