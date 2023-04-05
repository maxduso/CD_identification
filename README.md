# Development of an Automated Method of Cultural Depression Identification
This script is the result of a project focused on automating the process of identifying historical remnant of indigenous shelters known as cultural depression. The script takes as an input a digital elevation model, and returns identified sites of cultural depressions as shapefiles.

## Workflow
### Whitebox Tools (Depth Sink)
Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html

It was found that hydological tools could be used for this purpose as cultural depressions can be though  of as sinks which are somewhat unique to a landscape as they have no outlet for water. This step in the analysis results in a raster output where pixel values represent the depth of that pixel below the top of a given pit. Though this depth information has potential to be usefull given a different methodology, the image was converted to binary where pixels were either 127(pits), or 0(non-pits).

### Skimage Segementation
scikit source: https://www.youtube.com/watch?v=qfUJHY3ku9k&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=57

Skimage contains an image segemntation fucntion within its measure library where it splits an image into "labels". Essentially, instead of all pixells found within pits in the image having the same value (127) they now have values according to which pit they are found within. Skimage uses pixel connectivity rules to do this.

Regionprops_table is then used to filter these labels based on their shape and size whic skimage also allow by caluclating label properties. 

`properties = ['label','area','eccentricity','perimeter']`

Each label and its associated parameters then get stored in a row of a dataframe 