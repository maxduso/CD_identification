#!/usr/bin/env python
# coding: utf-8

# # Cultural depression identification using whitebox hydrological tools X scikit image segmentation
# 
# ## Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html
# ## scikit source: https://www.youtube.com/watch?v=qfUJHY3ku9k&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=57

# In[58]:


area_max = 300
area_min = 50
eccentricity_max = 0.6


# In[59]:


from whitebox_tools import WhiteboxTools
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio import plot as rasterplot
from skimage.segmentation import slic, clear_border
from skimage.color import label2rgb, rgb2gray
from skimage import measure, io, img_as_ubyte
from skimage.measure import label, regionprops
import rasterio as rio
from rasterio.plot import show
import rasterio.features
from rasterio.features import shapes
import pandas as pd
import skimage.io
import cv2
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from osgeo import osr, gdal, ogr
import fiona
from shapely.geometry import Polygon,shape
import shapely
import rioxarray as rxr
import earthpy as et
#must set the dirrectory to where whitebox tools package was stored on computer for some reason
wbt = WhiteboxTools()
wbt.set_whitebox_dir('C:\\Users\\maxduso.stu\\Anaconda3\\pkgs\\whitebox_tools-2.2.0-py311hc37eb10_2\\Library\\bin')


# In[60]:


#set up working dirrectory / data folder
data_dir = 'C:\\Users\\maxduso.stu\\Desktop\\FCOR_599\\project_work\\data\\'
os.chdir(data_dir)

#Set path to input image
in_file_name = "full_pa.tif"
out_file_name = "sink_depth_rast.tif"
input_path = data_dir + "tif_folder\\" + in_file_name
output_path = data_dir + "tif_folder\\" + out_file_name


# ## Hydrology "Depth Sink" pit delineation
# 
# ### Fills depressions and then uses that surface to sibtract from the original surface so pretty much just gives the depressions in the end. But could be pretty innefficient due to the need to fill depressions and then subtract the two surfaces.

# In[61]:


sink_depth = wbt.depth_in_sink(
                        input_path,
                        output_path,
                        zero_background= True
)


# ## Ensure image type and get spatial attributes

# In[62]:


## import image with spatial attributes using rasterio
spatial_image = rio.open(output_path)

# read the band of spatial image to a numpy array for processing
image = spatial_image.read(1)

spatial_profile = spatial_image.profile

#old method of doing shit
#image = cv2.imread(output_path, -1)
#spatial_image = gdal.Open(output_path)

#prj = spatial_image.GetProjection()
#srs = osr.SpatialReference(wkt=prj)
#input_crs = srs.GetAttrValue('projcs')
#input_crs
#print(type(image))
print(spatial_profile['crs'])


# ### Create Threshold and Binarize Image

# In[63]:


# Make solid black classified image, uint8 is plenty for 3 values
classified = np.zeros(image.shape, np.uint8)

# Set everything above a low threshold to 127
classified[image>0.001] = 127


# ## Create the Segments

# In[64]:


#label segments
#this creates a numpy array where each pixel is categorized to a cluster or blob
labels = measure.label(classified, connectivity = image.ndim)


# ## Filter labels

# In[65]:


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
seg_props['area_m'] = seg_props['area'] / 3.3  #each pixel is about 0.3m x 0.3m so to get meters multiply by this factor


# In[66]:


#print(seg_props.columns)
#filter on area parameter
filtered_segs = seg_props[seg_props['area_m'] > area_min]
filtered_segs = filtered_segs[filtered_segs['area_m'] < area_max]
                              
#filter on eccentricity
#Eccentricity of the ellipse that has the same second-moments as the region. 
#The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. 
#The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
filtered_segs = filtered_segs[filtered_segs['eccentricity'] < eccentricity_max]

#extract the label incices of the labels which meet the physical criteria outlined above.
filtered_seg_inds = filtered_segs['label']
#conver to list
filtered_list = filtered_seg_inds.tolist()

len(filtered_segs)


# ## Extract Filtered Indices from the OG Labels Image

# In[67]:


#create bool array where true values are where filtered labels are
filtered_labels = np.in1d(labels, filtered_list).reshape(labels.shape)
filtered_labels = filtered_labels.astype(int)
print(type(spatial_image))
type(filtered_labels)


# ## Create georeferenced raster again (From Github)
# 
# ### https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8?fbclid=IwAR2hi51gLAJ8GluR3hyjUawtu0V7iqLWKITzakr2pvjpTCuQKUUqFyn1ezs

# In[68]:


# Create GeoTiff of NNIR Water Array
output_image = input_path = data_dir + "tif_folder\\" + "cds_identified.tif"

with rio.Env():
    spatial_profile = spatial_image.profile # get profile of spatial image
    spatial_profile.update(dtype=rio.uint16, count=1, nodata=None) # update profile. count is number of bands
    with rio.open(output_image, "w", **spatial_profile) as dst: # create virtual file with nnir_profile
        dst.write(filtered_labels.astype(rio.uint16), 1) # write data to file with datatype


# ## Georefferenced Raster to GeoJson geometry Fefatures
# 
# ### https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons

# In[69]:


# mask to only include the pits and not surrounding area
mask = image == 1
input_crs = spatial_profile['crs']
    
with rasterio.Env():
    with rasterio.open(output_image, crs = input_crs) as src:
        image = src.read(1) # first band
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=src.transform)))
        
geoms = list(results)
len(geoms)


# ## Polygonize With Geopandas

# In[70]:


gdf  = gpd.GeoDataFrame.from_features(geoms)
gdf.set_crs(input_crs, inplace = True)


# ## Export to shapefile

# In[72]:


outShapefile = input_path = data_dir + "shapes\\" + "identified_sites.shp"
gdf.to_file(outShapefile)

