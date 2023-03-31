#!/usr/bin/env python
# coding: utf-8

# # Cultural depression identification using whitebox hydrological tools X scikit image segmentation
# 
# ## Whitebox docs: https://www.whiteboxgeo.com/manual/wbt_book/preface.html
# ## scikit source: https://www.youtube.com/watch?v=qfUJHY3ku9k&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=57

# In[1]:


area_max = 300
area_min = 50
eccentricity_max = 0.6


# In[2]:


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


# In[3]:


#set up working dirrectory / data folder
data_dir = 'C:\\Users\\maxduso.stu\\Desktop\\FCOR_599\\project_work\\data\\'
os.chdir(data_dir)

#Set path to input image
in_file_name = "Clip_depth_sink.tif"
out_file_name = "sink_depth_rast.tif"
input_path = data_dir + "tif_folder\\" + in_file_name
output_path = data_dir + "tif_folder\\" + out_file_name


# ## Hydrology "Depth Sink" pit delineation
# 
# ### Fills depressions and then uses that surface to sibtract from the original surface so pretty much just gives the depressions in the end. But could be pretty innefficient due to the need to fill depressions and then subtract the two surfaces.

# In[4]:


sink_depth = wbt.depth_in_sink(
                        input_path,
                        output_path,
                        zero_background= True
)


# ## Ensure image type and get spatial attributes

# In[5]:


## import image
image = cv2.imread(output_path, -1)
spatial_image = gdal.Open(output_path)

prj = spatial_image.GetProjection()

srs = osr.SpatialReference(wkt=prj)
input_crs = srs.GetAttrValue('projcs')
input_crs
print(type(image))
print(type(spatial_image))


# ### Create Threshold and Binarize Image

# In[6]:


# Make solid black classified image, uint8 is plenty for 3 values
classified = np.zeros(image.shape, np.uint8)

# Set everything above a low threshold to 127
classified[image>0.001] = 127


# ## Create the Segments

# In[7]:


#label segments
#this creates a numpy array where each pixel is categorized to a cluster or blob
labels = measure.label(classified, connectivity = image.ndim)


# ## Filter labels

# In[8]:


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


# In[9]:


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

len(filtered_list)


# ## Extract Filtered Indices from the OG Labels Image

# In[10]:


#create bool array where true values are where filtered labels are
filtered_labels = np.in1d(labels, filtered_list).reshape(labels.shape)
filtered_labels = filtered_labels.astype(int)
print(type(spatial_image))
type(filtered_labels)


# ## Create georeferenced raster again (From Github)
# 
# ### https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8?fbclid=IwAR2hi51gLAJ8GluR3hyjUawtu0V7iqLWKITzakr2pvjpTCuQKUUqFyn1ezs

# In[23]:


def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file with spatial information
        array : numpy.array 
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    print(cols)
    print(rows)
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 
    print(dataset.GetGeoTransform())

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    print(newRasterfn)
    print(cols)
    print(rows)
    print(band_num)
    print(GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    print(prj)


# In[24]:


array2raster("tif_folder/out_raster.tif", spatial_image, filtered_labels, "Byte")


# ## Georefferenced Raster to GeoJson geometry Fefatures
# 
# ### https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons

# In[ ]:


# mask to only include the pits and not surrounding area
mask = image == 1

with rasterio.Env():
    with rasterio.open('tif_folder/out_raster.tif', crs = input_crs) as src:
        image = src.read(1) # first band
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=src.transform)))
        
geoms = list(results)
len(geoms)


# ## Polygonize With Geopandas

# In[ ]:


gdf  = gpd.GeoDataFrame.from_features(geoms)
gdf.set_crs(input_crs, inplace = True)


# ## Export to shapefile

# In[ ]:


outShapefile = 'shapes/full_pa_50_300_05.shp'
gdf.to_file(outShapefile)

