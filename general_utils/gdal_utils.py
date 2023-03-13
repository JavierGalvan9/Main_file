# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:03:36 2022

@author: UX325
"""

import os
# Import the required libraries
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numba import njit
from osgeo import gdal, osr

#import pickle
#import csv
parentDir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management


# Method that opens a raster using Gdal
def openRaster(raster_file, access=0):
    ds = gdal.Open(raster_file, access)
    if ds is None:
        print("Error opening raster dataset")
    return ds

# Method to remove from the dataset the no data values
def clean_nodata_values(raster_array, nodatavalue):
    mask = raster_array!=nodatavalue
    clean_raster = raster_array[mask]
    return clean_raster, mask

def getPointCoordinates(ds):
    # https://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    # get the existing coordinate system
    old_cs= osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs .ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs,new_cs) 

    #get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3] 

    #get the coordinates in lat long
    lat1, long1, _ = transform.TransformPoint(minx,miny) 
    lat2, long2, _ = transform.TransformPoint(maxx,maxy)
    
    lats = np.linspace(lat2, lat1, height)
    longs = np.linspace(long1, long2, width)
    
    xs, ys = np.meshgrid(longs, lats)
    
    return xs, ys

# Method to extract a clean digitized dataset where we have 
# positives (1) and negatives (2) almond trees
def getDigitizedBand(dfn, band=1, access=0, save_figures=False):    
    ds = openRaster(dfn, access)
    band = ds.GetRasterBand(band)  
    crude_band = band.ReadAsArray()
    #clean_band, mask = clean_nodata_values(crude_band, nodatavalue)
    if save_figures:
        #We want to visualize the band so we set the "no data" values to 0
        z = np.copy(crude_band)
        nodatavalue = band.GetNoDataValue()
        z[z == nodatavalue] = 0
        fig = plt.figure()
        plt.imshow(z, interpolation='none')  
        legend_elements = [Line2D([0], [0], marker='s', color='y', linestyle='None', label='Negatives', markerfacecolor='y', markersize=10),
                           Line2D([0], [0], marker='s', color='c', linestyle='None', label='Positives', markerfacecolor='c', markersize=10)]
        plt.legend(handles=legend_elements, loc='upper left')
        plt.axis('off')
        path = 'Bands_images'
        os.makedirs(path, exist_ok=True)
        plt.tight_layout()
        fig.savefig(os.path.join(path, 'Digitized_band.png'), dpi=300, transparent=True) 
        plt.close()
    return crude_band

# Method that save plots of the different bands
def band_plot(df, band_name, original_shape=(253, 303)):
    band_array = df[band_name]
    vmin = np.min(band_array)
    vmax = np.max(band_array)
    band_array = band_array.to_numpy().reshape(original_shape)
    im = plt.imshow(band_array, vmin=vmin, vmax=vmax)  
    plt.colorbar(im)
    plt.axis('off')
    path = 'Bands_images'
    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join('Bands_images', band_name)+'.png', dpi=300, transparent=True) 
    plt.close()
    
# Method that extract the processed data arrays for the different bands
def getMultiRasterBands(raster_file, 
                        metadata_file, 
                        TOA=True,
                        vegetation_indices=True,
                        plot_band_images=True,
                        save_dataframe=True,
                        save_file='',
                        manual_sample_dir=None, 
                        access=0):
    ds = openRaster(raster_file, access)
    # Create dictionaries which will contain all the dataset
    df = {}
    metadata_df = {}
    with open(metadata_file) as f:
        band_counter = 0
        for line in f:
            if 'BEGIN_GROUP' in line and 'BAND' in line:
                band_counter += 1
                band_name = line[19:-1]
                metadata_df[band_name] = {}
                df[band_name] = {}
                actual_band = ds.GetRasterBand(band_counter)  
                nodatavalue = actual_band.GetNoDataValue()
                crude_band = actual_band.ReadAsArray()
                real_image_shape = crude_band.shape
                clean_band, _ = clean_nodata_values(crude_band, nodatavalue)
                df[band_name] = clean_band
            elif 'absCalFactor' in line:
                metadata_df[band_name]['absCalFactor'] = float(line[16:-2])
            elif 'effectiveBandwidth' in line:
                metadata_df[band_name]['effectiveBandwidth'] = float(line[22:-2])   
            elif 'TDILevel' in line:
                metadata_df[band_name]['TDILevel'] = float(line[12:-2])    
    
    metadata_df['original_shape'] = real_image_shape
    df = pd.DataFrame(df, columns= list(df.keys()))
    
#  " Digital numbers of the WorldView-2 images were converted to top-of-atmosphere reflectance using the absolute 
#  radiometric calibration factors and effective bandwidths for each band, according to the specifications provided
#  by the data provider (Digital Globe 2010a). This step was mainly done to enable calculation of vegetation 
#  indices (Glenn et al. 2008) and to transform the data into a common scale (i.e., percent reflectance)."
#  Remote Sensing of Woodland Structure and Composition in the Sudano-Sahelian zone 

# Here is an issue that does not apply to the NDVI, but that is critically important for other vegetation indices 
# (VI). Some VI incorporate numerical constants, typically determined using reflectance data. For example, 
# the Soil Adjusted Vegetation Index includes the value 0.5 as a factor in the denominator of the equation, 
# and the factor 1.5 as a multiplier. These values are scaled assuming that the red and NIR spectral data are 
# measured in reflectance units, scaled 0-1. If DN values (e.g. 0-255) are used instead, the soil adjustment 
# will be totally ineffective. The same applies to various other VI.
    if TOA:
        for band_name in df.columns:
            DN_band = df[band_name].astype('float64')
            K = metadata_df[band_name]['absCalFactor']
            D = metadata_df[band_name]['effectiveBandwidth']
            TOA_band = DN_band*K/D
            df[band_name] = TOA_band
        
    if vegetation_indices:
        # Normalized Difference Vegetation Index (NDVI)
        df['NDVI_1'] = (df['N'] - df['R'])/(df['N'] + df['R'])
        df['NDVI_2'] = (df['N2'] - df['R'])/(df['N2'] + df['R'])
        
        # Enhanced Normalized Difference Vegetation Index (ENDVI)
        df['ENDVI_1'] = (df['N'] + df['G'] - 2*df['B'])/(df['N'] + df['G'] + 2*df['B'])
        df['ENDVI_2'] = (df['N2'] + df['G'] - 2*df['B'])/(df['N2'] + df['G'] + 2*df['B'])

        # Normalized ratio of NIR and red:
        df['ratio_nir_red_1'] = df['N']/df['R']
        df['ratio_nir_red_2'] = df['N2']/df['R']
        
        # Normalized ratio of red edge and NDVI:
        df['ratio_red_NDVI1'] = df['RE']/df['NDVI_1']
        df['ratio_red_NDVI2'] = df['RE']/df['NDVI_2']
        
        # Average intensity of red, green and blue (BR3):
        df['BR3'] = (df['R'] + df['G'] + df['B'])/3
        
        # Average intensity of nir, red, green and blue (BR4):
        df['BR4'] = (df['N'] + df['R'] + df['G'] + df['B'])/4
        
        # Green Normalized Difference Vegetation Index (GNDVI)
        df['GNDVI_1'] = (df['N'] - df['G'])/(df['N'] + df['G'])
        df['GNDVI_2'] = (df['N2'] - df['G'])/(df['N2'] + df['G'])

        # Soil Adjusted Vegetation Index (SAVI)
        df['SAVI_1'] = (df['N'] - df['R'])*(1 + 0.25)/(df['N'] + df['R'] + 0.25)
        df['SAVI_2'] = (df['N'] - df['R'])*(1 + 0.5)/(df['N'] + df['R'] + 0.5)
        df['SAVI_3'] = (df['N'] - df['R'])*(1 + 1)/(df['N'] + df['R'] + 1)
        df['SAVI_4'] = (df['N2'] - df['R'])*(1 + 0.25)/(df['N2'] + df['R'] + 0.25)
        df['SAVI_5'] = (df['N2'] - df['R'])*(1 + 0.5)/(df['N2'] + df['R'] + 0.5)
        df['SAVI_6'] = (df['N2'] - df['R'])*(1 + 1)/(df['N2'] + df['R'] + 1)

        # Normalised Phaeophytinization Index (NPQI)
        df['NPQI_1'] = (df['C'] - df['B'])*(1 + 0)/(df['C'] + df['B'] + 0)
        df['NPQI_2'] = (df['C'] - df['B'])*(1 + 0.25)/(df['C'] + df['B'] + 0.25)
        df['NPQI_3'] = (df['C'] - df['B'])*(1 + 0.5)/(df['C'] + df['B'] + 0.5)
        df['NPQI_4'] = (df['C'] - df['B'])*(1 + 0.75)/(df['C'] + df['B'] + 0.75)
        df['NPQI_5'] = (df['C'] - df['B'])*(1 + 1)/(df['C'] + df['B'] + 1)

        # Chlorophyll Index Red edge (CLR)
        df['CLR'] = (df['N']/df['RE']) - 1

        # Chlorophyll Index Green (CLG)
        df['CLG'] = (df['N']/df['G']) - 1

        # Blue Normalized Difference Vegetation Index (BNDVI)
        df['BNDVI'] = (df['N'] - df['B'])/(df['N'] + df['B'])

        # Carter Index 1 (CTR1)
        df['CTR1'] = df['R']/df['C']
    
    if plot_band_images:
        for band_name in df.columns:
            band_plot(df, band_name, original_shape=real_image_shape)
    
    # Add the coordinates for each point
    longitudes, latitudes = getPointCoordinates(ds)
    df['Lats'] = latitudes.ravel()
    df['Longs'] = longitudes.ravel()
    
    if manual_sample_dir is not None:
        manual_sample_tif = os.path.join(manual_sample_dir, 'Almonds_Sample.tif')
        incomplete_labels = getDigitizedBand(manual_sample_tif, save_figures=True).ravel()
        # Original labels are: 1 infected, 2 non-infected, 255 no-tree
        incomplete_labels[incomplete_labels==2] = 0
        df['incomplete_labels'] = incomplete_labels

        treeID_tif = os.path.join(manual_sample_dir, 'Almonds_Sample_ID.tif')
        incomplete_trees_ID = getDigitizedBand(treeID_tif, save_figures=False).ravel()
        # Sample of 400 trees with IDs from 0 to 399, 65535 in other case
        df['incomplete_tree_ID'] = incomplete_trees_ID
#        df = df.sort_values(["treeID"], ascending = True)
    
    if save_dataframe:
        file_management.save_lzma(df, os.path.join(save_file, 'dataset.lzma'), '')
        file_management.save_lzma(metadata_df, os.path.join(save_file, 'metadata_df.lzma'), '')  
#         df.to_csv('dataset.csv', index = True, header=True)
#         f = open("metadata_df.pkl","wb")
#         pickle.dump(metadata_df,f)
#         f.close()
    
    return df, metadata_df
