# Release notes

<!-- do not remove -->

## 0.1.4

### New Features

- Added quantification strategy to calculate the feature count relative to ROI areas
- Adapted export to work with multiple quantification strategies

## 0.1.3

### New Features

- Remove stitching artefacts only implemented for black pixels ([#31](https://github.com/Defense-Circuits-Lab/findmycells/issues/31))
  - added option to remove white pixels

- adding Windows compatibility ([#6](https://github.com/Defense-Circuits-Lab/findmycells/issues/6))
  - pathlib objects create a PosixPath object on Mac and Linux, but a WindowsPath object on Windows -> assertion for input types considers PosixPath and WindowsPath as valid

### Bugs Squashed

- czi-file shape behaves differently across different imaging conditions ([#28](https://github.com/Defense-Circuits-Lab/findmycells/issues/28))
  - Depending on whether czi-files were acquired with more than one z-plane, more than one channel, more than one image version and so on, the czifile module returns image data in different shape. Taking care of the different possibilities, is achieved by checking the czi-file metadata

- Installation of sample dataset fails ([#22](https://github.com/Defense-Circuits-Lab/findmycells/issues/22))
  - fixed target directory in shutil.move and set wget.download() out attribute to str instead of Path

- Setting a slider range in preprocessing step caused fail in segmentation step ([#21](https://github.com/Defense-Circuits-Lab/findmycells/issues/21))
  - set a filter for None values in expected file count

- Migrate projects between computers ([#19](https://github.com/Defense-Circuits-Lab/findmycells/issues/19)): saving and loading projects does not work after migration of project to different computer
  - project root dir updated

- Assert for validity of .roi files ([#8](https://github.com/Defense-Circuits-Lab/findmycells/issues/8)): roi-files with less than 3 coordinates given, can't be created and the shapely pckg will throw an error "ValueError: A LinearRing must have at least 3 coordinate tuples"
  - Error handled in findmycells

- set dependencie versions: imageio, scikit-image ([#5](https://github.com/Defense-Circuits-Lab/findmycells/issues/5))
  - imageio and scikit image versions were fixed in settings.ini
