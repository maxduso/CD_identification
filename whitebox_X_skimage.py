#!/usr/bin/env python
# coding: utf-8

# Pit detection script

#Credit to the following python libraries and documentation
# ## Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html
# ## skimage source: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html

from whitebox_tools import WhiteboxTools
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from skimage.segmentation import slic, clear_border
from skimage.color import label2rgb, rgb2gray
from skimage import measure, io, img_as_ubyte
from skimage.measure import label, regionprops
import rasterio as rio
from rasterio.plot import show
import rasterio.features
from rasterio.features import shapes
from rasterio import features
import pandas as pd
import numpy as np
from shapely.geometry import shape
import pprint
#must set the dirrectory to where whitebox tools package was stored on computer for some reason
wbt = WhiteboxTools()
wbt.set_whitebox_dir('C:\\Users\\maxduso.stu\\Anaconda3\\pkgs\\whitebox_tools-2.2.0-py311hc37eb10_2\\Library\\bin')

#SET PARAMS
area_max = 150
area_min = 15
eccentricity_max = 0.6

#SET WORKING DIRRECTORY / DATA FOLDERS
data_dir = 'C:\\Users\\maxduso.stu\\Desktop\\FCOR_599\\project_work\\data\\'
os.chdir(data_dir)

#SET INPUT AND OUTPUT PATH
in_file_name = "full_pa_1m.tif"
out_file_name = "identified_cds.shp"
input_path = data_dir + "tif_folder\\" + in_file_name
output_path = data_dir + "shapes\\" + out_file_name

#HYDROLOGY DEPTH SINK PIT DELINEATION
# ### Fills depressions and then uses that surface to sibtract from the original surface 
# so pretty much just gives the depressions in the end. But could be pretty innefficient due to the need to fill depressions and then subtract the two surfaces.

intermediate_path = data_dir + "tif_folder\\" + "depth_sink_int.tif"
sink_depth = wbt.depth_in_sink(
                        input_path,
                        intermediate_path,
                        zero_background= True
)


# ## READ IN SPATIAL IMAGE, EXTRACT ARRAY AND SPATIAL ATTRIBUTES

## import image with spatial attributes using rasterio
spatial_image = rio.open(intermediate_path)
spatial_profile = spatial_image.profile
input_crs = spatial_profile['crs']
print("the initial spatial profile is",spatial_profile)

# read the band of spatial image to a numpy array for processing
image = spatial_image.read(1)


# ### CREATE THRESHOLDED BINARY IMAGE

# Make solid black classified image, uint8 is plenty for 3 values
classified = np.zeros(image.shape, np.uint8)

# Set everything above a low threshold to 127
classified[image>0.001] = 127

# ## CREATE SEGEMENTS

#label segments
#this creates a numpy array where each pixel is categorized to a cluster or blob
labels = measure.label(classified, connectivity = image.ndim)

# ## FILTER THE LABELS BASE ON THEIR PROPERTIES

#specify the properties that I want to gather
#just a dictionary oft he properties that I would like to measure for each label
props = measure.regionprops_table(labels, image,
                                      properties = ['label',
                                                   'area',
                                                   'eccentricity',
                                                   'perimeter'])


#create a dataframe from these object properties
seg_props = pd.DataFrame(props)
#create an area column expressed in meters
#seg_props['area_m'] = seg_props['area'] / 3.3  #each pixel is about 0.3m x 0.3m so to get meters multiply by this factor

#print(seg_props.columns)
#filter on area parameter
#important to note that in this case the pixel size is 1m so I do not do any math, but depends on raster size
filtered_segs = seg_props[seg_props['area'] > area_min]
filtered_segs = filtered_segs[filtered_segs['area'] < area_max]
                              
#filter on eccentricity
#Eccentricity of the ellipse that has the same second-moments as the region. 
#The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. 
#The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
filtered_segs = filtered_segs[filtered_segs['eccentricity'] < eccentricity_max]

#extract the label incices of the labels which meet the physical criteria outlined above.
filtered_seg_inds = filtered_segs['label']
#conver to list
filtered_list = filtered_seg_inds.tolist()

print("The number of identified labels meeting criteria is:", len(filtered_segs))

#EXTRACT FILTERED INDICES FROM THE OG LABELS IMAGE

#create bool array where true values are where filtered labels are
filtered_labels = np.in1d(labels, filtered_list).reshape(labels.shape)
filtered_labels = filtered_labels.astype(int)

# ## CREATE GEOREFFERENCED RASTER ONCE AGAIN

# Create GeoTiff of NNIR Water Array
output_image = input_path = data_dir + "tif_folder\\" + "cds_identified.tif"

with rio.Env():
    #spatial_profile = spatial_image.profile # get profile of spatial image
    spatial_profile.update(dtype=rio.uint16, count=1, nodata=None) # update profile. count is number of bands
    with rio.open(output_image, "w", **spatial_profile) as dst: # create virtual file with nnir_profile
        dst.write(filtered_labels.astype(rio.uint16), 1) # write data to file with datatype


#Georefferenced Raster to GeoJson geometry Fefatures
#https://rasterio.readthedocs.io/en/latest/topics/features.html

with rasterio.open(output_image) as src:
    print("the src profile is",src.profile)
    image = src.read(1) # first band

# mask to only include the pits and not surrounding area
mask = image == 1

#create shape features
shapes = features.shapes(image, mask = mask, transform=src.transform)
pprint.pprint(next(shapes))

# read the shapes as separate lists
raster_val = []
geometry = []
for shapedict, value in shapes:
    raster_val.append(value)
    geometry.append(shape(shapedict))

# build the gdf object over the two lists
gdf = gpd.GeoDataFrame(
    {'raster_val': raster_val, 'geometry': geometry },
    crs="EPSG:3005"
)

print(gdf['geometry'])
# ## POLYGONIZE WITH GEOPANDAS

#RUN ONE MORE FILTER TO ENSURE THAT THE PIT SEGMENTS ARE INDEED BEING WRITTEN TO FILE
#Make sure that only the actual pit segments are returned
gdf = gdf.loc[gdf['raster_val']==1]

#Make sure that groupings of only a couple of pixels are not returned
gdf = gdf.loc[gdf['geometry'].area > 4]

#use this to find out what the distribution of the features areas are so we know which
#to eliminate
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(gdf['geometry'].area, bins = [0, 2, 5, 10, 25])

# ## EXPORT TO SHAPEFILE
print(len(gdf['geometry']))
gdf.to_file(output_path)


#create the binary sinks image just so that you can look at it and see if it makes sense what has been identified
dpi = 80
height, width = np.array(classified.shape, dtype=float) / dpi

fig = plt.figure(figsize=(width, height), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

binary_path = data_dir + "tif_folder\\" + "binary_image.tif"
ax.imshow(classified, interpolation='none')
fig.savefig(binary_path, dpi=dpi)
