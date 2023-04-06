# Development of an Automated Method of Pit Identification
Max Duso
dusomasimiliano@gmail.com

This script is the result of a project focused on automating the process of identifying pits of a given shape and size in the ground. Inputs of the script include size and shape parameters of desire to look for, as well as a digital elevation model in which to find the pits.

## Data Configuration
As this script is set up, it works with a local folder "data" inside of which are two subfolders "tif_folder" and "shapes". The script expects the input DEM to be held in the tif folder and will output final result to the shapes folder. Additionally, there is an intermediate output tif called "binary_image.tif" that will be output to the tiff folder. This file is not a main result but may be usefull for visual analysis to give a sense of what features are getting filtered out as a result of the filtration step.

## Workflow
### Whitebox Tools (Depth Sink)
Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

It was found that hydological toolsets have the functionality to find pits, which are defined as low lying pixels grouping which have no outlet for water. This step in the analysis results in a raster output where pixel values represent the depth of that pixel below the top of a given pit (the lowest outlet for water). The image was then converted to binary where pixels were either assigned a value of 127(pits), or 0(non-pits) for ease of segmentation. Worth exploring may be the depth profiles of the pits in future, more advanced studies.

### Skimage Segementation
scikit source: https://www.youtube.com/watch?v=qfUJHY3ku9k&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=57

Skimage contains an image segemntation function within its measure library where it splits an image into "labels". Essentially, instead of all pixels found within pits in the image having the same value (127) they now have values according to which pit they are found within. Skimage uses pixel connectivity rules to do this.

Skimage's Regionprops_table is then used to filter these labels based on their shape and size which skimage also allows by caluclating label properties. 

`properties = ['label','area','eccentricity','perimeter']`

Each label and its associated parameters then get stored in a row of a dataframe. Rows of the datafame are filtered out based on the parameter values that are defined by the user.

### Conversion to Shapefile
In order to convert the filtered set of pits from the form of a numpy array to a shapfile, the final step is to re-apply the spatial profile of the raster extracted upon the import of the digital elevation raster. Finally rasterio is used to produce shape features from the georeferenced filtered labels. 

## User Defined Variables for Script

- **area_max** -> The maximum area filter for identified pits (float).
- **area_min** -> The minimum area filter for identified pits (float).
- **eccentricity_max** -> The minimum circularity filter for identified pits where eccentricity = 0 is a perfect circle (float 0-1).
- **data_dir** -> The path to the dirrectoory where dems are stored and also where output will be written (string).
- **in_file_name** -> The file name of the dem that is desired to scan for cultural depressions (string). The file itself should be of .tif format. This script was made to work with DEMs up to 1m in resolution, finer resolution however should also work.
- **out_file_name** -> The desired name of the shapefile output (string). The file itself is a shapefile found in your shapes dirrectory.

## Limitations
whitebox_X_scikit is flawed in its ability to detect depressions with touching edges. Within the study area it was quite common that sites be dirrectly adjacent to one another. This causes errosion to take place more  rapidly on the berm at the point where the sites touch due to increased angle of slope on either side of the berm. As a result, the depth sink tool merges the two pits becasue the middle point is lower than the rest of the perimeters resulting in a figure 8 shape. Becasue the script is not parameterized to detect this shape, nor the size of two merged pits, these pits are missed.

In order to adress this issue, the implimentation of the skimage watershed tool was explored. This tool fills depressions, or as the tool would define them watersheds, but does not merge two if they should meet at a col feature in the landscape. Though this appeards in theory to be the correct method for the job, it has not proven fruitful yet as it has done a poor job of identification upon trial runs. This approach warrants more exploration which will continue into the future.

Additionally though it is convenient that skimage has built in to it segment parameter calculation, the exploration of calculating parameters of shape features later in the analysis is encouraged. Skimage parameterization resulted in some output shapes which are undesireable in shape while others were filtered as expected. Therefore omparisson of methods would be valuable.