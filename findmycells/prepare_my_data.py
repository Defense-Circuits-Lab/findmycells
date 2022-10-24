from pathlib import Path
import pandas as pd

from typing import List, Tuple, Dict, Optional


class CreateExcelFilesForImageLoader:
    
    def __init__(self, root: Path, destination: Path, batch_processing: bool) -> None:
        if batch_processing == True:
            self.batch_create_excel_files_for_image_loader(root_dir_path = root, destination_path = destination)
        else:
            expected_filename = f'{root.name}.xlsx'
            if destination.name != expected_filename:
                error_message_line1 = f'The destination filepath you provided ({destination.as_posix()}) \n'
                error_message_line2 = f'does not match the expected input, which has to included the following expected filename: {expected_filename}'
                error_message = error_message_line1 + error_message_line2
                raise ValueError(error_message)
            else:
                self.create_excel_file_for_image_loader(source_directory_path = root, destination_filepath = destination)
                
    
    def batch_create_excel_files_for_image_loader(self, root_dir_path: Path, destination_path: Path) -> None:
        all_subdirs = []
        for elem in root_dir_path.iterdir():
            if elem.is_dir():
                all_subdirs.append(elem)        
        for subdir_path in all_subdirs:
            destination_filepath = destination_path.joinpath(f'{subdir_path.name}.xlsx')
            self.create_excel_file_for_image_loader(source_directory_path = subdir_path, destination_filepath = destination_filepath)
            
    
    def create_excel_file_for_image_loader(self, source_directory_path: Path, destination_filepath: Path) -> None:
        data = {'plane_id': [],
                'plane_filename': [],
                'plane_filepath': []}
        for elem in source_directory_path.iterdir():
            if elem.name.startswith('.') == False:
                data['plane_id'].append(self.identify_plane_index_str(filename = elem.name, source_path = source_directory_path))
                data['plane_filename'].append(elem.name)
                data['plane_filepath'].append(elem.as_posix())
        self.check_for_consecutive_plane_ids(indices = data['plane_id'], source_path = source_directory_path)
        df = pd.DataFrame(data = data)
        df.to_excel(destination_filepath, index=False)
        
        
    def identify_plane_index_str(self, filename: str, source_path: Path) -> str:
        filename = filename[:filename.rindex('.')]
        numbers_in_filename = []
        for elem in filename:
            try: 
                numbers_in_filename.append(int(elem))
            except ValueError:
                continue
        if len(numbers_in_filename) == 0:
            error_message_line1 = 'The image filename must indicate the index of the corresponding plane of the image in the z-stack.\n'
            error_message_line2 = f'However, no numbers could be detected in: {source_path.as_posix()}\\{filename}. Please consider renaming!'
            error_message = error_message_line1 + error_message_line2
            raise ValueError(error_message)
        plane_id = ''
        for number in numbers_in_filename:
            plane_id += str(number)
        if len(plane_id) < 3:
            plane_id = plane_id.zfill(3)
        return plane_id   
    
    
    def check_for_consecutive_plane_ids(self, indices: List[str], source_path: Path) -> None:
        indices = [int(elem) for elem in indices]
        for i in range(len(indices) - 1):
            if indices[i] + 1 != indices[i + 1]:
                raise ValueError(f'The plane ids as inferred from the filenames in {source_path.as_posix()} are not consecutive!')