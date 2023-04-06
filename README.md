# Development of an Automated Method of Cultural Depression Identification
Max Duso
dusomasimiliano@gmail.com

This script is the result of a project focused on automating the process of identifying historical remnant of indigenous shelters known as cultural depression. The script takes as an input a digital elevation model, and returns identified sites of cultural depressions as shapefiles.

## Data Configuration
As this script is set up, it works with a local folder "data" inside of which are two subfolders "tif_folder" and "shapes". The script expects the input DEM to be held in the fif folder and will output th  final result to the shapes folder. Additionally, there is an intermediary otuput tif called "depth_sink_int.tif" that will be output tot eh tiff folder. This file is not important but may be usefull to visually analyse to give a sense of what features are being filtered for  in the parameterization stage.

## Workflow
### Whitebox Tools (Depth Sink)
Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

It was found that hydological tools could be used for this purpose as cultural depressions can be though  of as sinks which are somewhat unique to a landscape as they have no outlet for water. This step in the analysis results in a raster output where pixel values represent the depth of that pixel below the top of a given pit. Though this depth information has potential to be usefull given a different methodology, the image was converted to binary where pixels were either 127(pits), or 0(non-pits).

### Skimage Segementation
scikit source: https://www.youtube.com/watch?v=qfUJHY3ku9k&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=57

Skimage contains an image segemntation fucntion within its measure library where it splits an image into "labels". Essentially, instead of all pixells found within pits in the image having the same value (127) they now have values according to which pit they are found within. Skimage uses pixel connectivity rules to do this.

Regionprops_table is then used to filter these labels based on their shape and size whic skimage also allow by caluclating label properties. 

`properties = ['label','area','eccentricity','perimeter']`

Each label and its associated parameters then get stored in a row of a dataframe. Rows of the datafame are filtered out based on the parameter values that are defined by the user.

### Conversion to Shapefile
In order to convert the filtered set of pits from the form of a numpy array to a shapfile, the final step is to re-apply the spatial profile of the raster extracted upon the import of the digital elevation raster. Finally rasterio is used to produce shape features from the georeferenced filtered labels. 

## User Defined Variables for Script

- **area_max** -> the maximum area filter for identified pits.
- **area_min** -> the minimum area filter for identified pits.
- **eccentricity_max** -> the minimum circularity filter for identified pits where eccentricity = 0 is a perfect circle (float 0-1).
- **data_dir** -> the path to the dirrectoory where dems are stored and also where output will be written (string).
- **in_file_name** -> the file name of the dem that is desired to scan for cultural depressions (string).
- **out_file_name** -> the desired name of the shapefile output (string)

## Limitations
whitebox_X_scikit is flawed in its ability to detect depressions with touching edges. Wthin the study area it was quite common that sites of past shelters be dirrectly adjacent to one another. This causes errosionto take place more  rapidly on the berm at the point where the sites touch due to increase angle of slope on either side of the berm. As a result, the depth sink tool merges the two pits as the middle point is lower that the rest of the perimeters resulting in a figure 8 shape. Becasue the script is not parameterized to detect this shape, nor the size of two merged pits, these pits are missed.

In order to adress this issue, the imlimentation of skimage watershed tool was explored. This tool fills depressions, or as the tool would define them watersheds, but does not merge two if they should meet at a col feature in the landscape. Though this appeards in theory to be the correct method for the job, it has not proven fruitful yet as it has done a poor job of identification. This approach warrants more exploration which will continue into the future.