# Release notes

<!-- do not remove -->

## 0.1.3

### New Features

- Remove stitching artefacts only implemented for black pixels ([#31](https://github.com/Defense-Circuits-Lab/findmycells/issues/31))
  - ToDo: Option to specify whether artefact pixel color is black or white

- adding Windows compatibility ([#6](https://github.com/Defense-Circuits-Lab/findmycells/issues/6))
  - pathlib objects create a PosixPath object on Mac and Linux, but a WindowsPath object on Windows
assertion for valid input types has to allow not only PosixPath, but WindowsPath as well

### Bugs Squashed

- czi-file shape behaves differently across different imaging conditions ([#28](https://github.com/Defense-Circuits-Lab/findmycells/issues/28))
  - Depending on whether you acquired czi-files with more than one z-plane, more than one channel, more than one image version and so on, the czifile module returns image data in different shape. Taking care of the different possibilities, can be achieved by the following solution (in case there are more conditions, that are not handled here, more elif loops can be added):

Solution:
in readers/microscopy_images.py: 

<img width="860" alt="image" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/068b1c17-49da-4693-b812-86b4b4684e9f">

- make unzipped folders containing .roi files usable! ([#24](https://github.com/Defense-Circuits-Lab/findmycells/issues/24))
  - unzipping, edit files (e.g., renaming) and zipping again -> .zip file is not recognised as ImageJ-roi file

- Installation of sample dataset fails ([#22](https://github.com/Defense-Circuits-Lab/findmycells/issues/22))
  - <img width="743" alt="sample_dataset_download_error" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/9f97ec40-7ccd-413a-aee8-05476094b7cd">

- Setting a slider range in preprocessing step causes fail in segmentation step ([#21](https://github.com/Defense-Circuits-Lab/findmycells/issues/21))

- Migrate projects between computers ([#19](https://github.com/Defense-Circuits-Lab/findmycells/issues/19))
  - saving and loading projects does not work after migration of project to different computer

- Assert for validity of .roi files ([#8](https://github.com/Defense-Circuits-Lab/findmycells/issues/8))
  - if there are roi-files with less than 3 coordinates given, the roi can't be created and the shapely pckg will throw an error "ValueError: A LinearRing must have at least 3 coordinate tuples"

<img width="468" alt="image" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/4801ee6c-fdc1-4b14-a12c-96bb7b270dc0">
<img width="468" alt="image" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/faa6b441-1589-40f5-b7f9-5b38dbaf7916">

- set dependencie versions: imageio, scikit-image ([#5](https://github.com/Defense-Circuits-Lab/findmycells/issues/5))
  - I followed the installation instructions, downloaded the test data and went through the GUI tutorial, but somehow at this step, it fails... 
User system: MacOS

<img width="557" alt="image" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/c1923613-8ec4-4aa5-b407-cf725c2551f5">
<img width="482" alt="image" src="https://github.com/Defense-Circuits-Lab/findmycells/assets/104254966/d3feb394-27b0-4c9b-8250-807f66478be9">
