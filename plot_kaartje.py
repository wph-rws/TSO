import os
import sys

# Ensure GDAL can find its support data files (e.g. gdalvrt.xsd) before importing
# geopandas/rasterio. In conda environments GDAL_DATA is often not set, which triggers
# "Cannot find gdalvrt.xsd" warnings during reprojection. Derive it from the active
# environment if it isn't already defined.
if not os.environ.get('GDAL_DATA'):
    for _gdal_data in (os.path.join(sys.prefix, 'Library', 'share', 'gdal'),  # Windows/conda
                       os.path.join(sys.prefix, 'share', 'gdal')):            # Unix/conda
        if os.path.isfile(os.path.join(_gdal_data, 'gdalvrt.xsd')):
            os.environ['GDAL_DATA'] = _gdal_data
            break

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import pathlib

from io import BytesIO
from PIL import Image
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from rasterio.plot import show
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate, translate
from pyproj import Transformer
from scipy import ndimage
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService

from tso_functions import load_project_dirs

"""
Functie voor het plotten van het kaartje in de hoek van de profielen.
   - gebiedscode is RW/RO/SW/SO/HW/HO/RP of een van de TSO-trajecten
   - ax is een object vanuit fig, ax = plt.subplots()
   - type kaart kan 'map' of 'feature' zijn
   - force_refresh voor geforceerde refresh van opgeslagen kaartgegevens
   - debug staat standaard uit
"""

cs_rd = 'EPSG:28992' # RD New (EPSG:28992)
cs_etrs = 'EPSG:25831' # ETRS89 / UTM zone 31N (EPSG:25831)

# cs_etrs = cs_rd

def plot_kaartje(location, ax, img_data, type_kaart='map', layer='terreinvlak', force_refresh_image=False, debug=False, rotation_angle=0, map_cache=None):
 
    type_kaarten = ('map', 'feature')
    if type_kaart not in type_kaarten:
        raise ValueError(f'Onbekend type kaart "{type_kaart}" gekozen, kies uit {type_kaarten}')     
       
    # Get location data from locaties.yaml
    xll, yll, dx, dy = get_location_data_for_map(location)

    # Bepaal bounding box voor gebied
    bbox = [xll, yll, xll + dx, yll + dy]  
        
    max_pixels = 4000
    if np.max([dx, dy]) > max_pixels:
    
        # Compute the scaling factor to keep aspect ratio of the input     
        scale_factor = np.min([max_pixels / dx, max_pixels / dy])
        
        # Calculate the new dimensions in pixels
        dx_img = int(scale_factor * dx)
        dy_img = int(scale_factor * dy)

    # Map: figure of area, different types available
    if type_kaart == 'map':
    
        # Get different layer to match layer of Belgium better
        if location not in ('RW', 'RO', 'SW', 'SO', 'HW', 'HO', 'RP'):  
            # map layer
            layer = 'top10nl' 
            
        if location in ('anka', 'kvgt'):
            coord_system = cs_etrs
        else:                
            coord_system = cs_rd
            
        output_format ='image/png'
        transparent = True
        
        project_dirs = load_project_dirs()
        image_dir = pathlib.Path(project_dirs.get('images_dir'))
        
        # Create the directory if it doesn't exist
        image_dir.mkdir(parents=True, exist_ok=True)

        # Specify a path for the map image file        
        map_image_path_png = image_dir / f'{location.lower()}_map.png'        
        map_image_path_tiff = image_dir / f'{location.lower()}_reprojected.tif'
    
        # geef limiet op voor leeftijd plaatje, als ouder: opnieuw ophalen
        age_limit = 3 * 60 * 60 * 24 * 31 # seconds in three months

        # Check if map exists and is not too old, else get new map
        if os.path.exists(map_image_path_png):

            # Get age of map from last modification time
            image_age = time.time() - os.path.getmtime(map_image_path_png)

            if image_age < age_limit and not force_refresh_image:

                # Load img_data only on first iteration
                if not img_data:
                    if rotation_angle == 0:
                        img_data = Image.open(map_image_path_png)
                    elif os.path.exists(map_image_path_tiff):
                        img_data = Image.open(map_image_path_tiff)
                    else:
                        img = get_new_map(location, bbox, layer, coord_system, output_format, transparent, dx_img, dy_img)
                        img_data = img
                        img_data.save(map_image_path_png)
                        
            # If too old or force_refresh == True
            else:
                img = get_new_map(location, bbox, layer, coord_system, output_format, transparent, dx_img, dy_img)
                img_data = img
                img_data.save(map_image_path_png)

        # if map doesn't exist, get new map
        else:
            img = get_new_map(location, bbox, layer, coord_system, output_format, transparent, dx_img, dy_img)
            img_data = img
            img_data.save(map_image_path_png)
        
        # Rotate map for new map or for map loaded from file
        if rotation_angle != 0:
            ax, img_data, bbox = rotate_image(location, ax, img_data, coord_system, rotation_angle, bbox, map_cache=map_cache)

        else:
            # Define the spatial extent of the image
            extent = [xll, xll + dx, yll, yll + dy]
    
            # Display the image with the defined extent
            ax.imshow(img_data, extent=extent)
            
            # Set number of x- and y-ticks and gridlines
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))            
            ax.grid()

    #%% Debug code om te bekijken welke maps er zijn en diverse andere eigeschappen
    if type_kaart == 'map' and debug == True:

        wms_url = 'https://service.pdok.nl/brt/top10nl/wms/v1_0?'
        wms = WebMapService(wms_url)

        # Weergave mogelijke stijlen, wordt niet gebruikt in aanroep
        styles = wms[layer].styles

        print(f'Beschikbare stijlen: {styles}')
        print(wms.identification.version)
        print(wms.identification.title)
        print(wms.identification.abstract)
        print(list(wms.contents))
        
        [op.name for op in wms.operations]
        print(wms.getOperationByName('GetMap').methods)
        print(wms.getOperationByName('GetMap').formatOptions)

    # Feature: polygonen met grove eigeschappen van gebied
    if type_kaart == 'feature':
        
    # Kies uit provincie of gemeente
        type_regio = 'provincie'
        wfs_url_part1 = 'https://service.pdok.nl/cbs/gebiedsindelingen/2024/wfs/v1_0?'
        wfs_url_part2 = f'request=GetFeature&service=WFS&version=1.0.0&typeName={type_regio}_gegeneraliseerd&outputFormat=json'
        wfs_url = wfs_url_part1 + wfs_url_part2
    
        wfs = WebFeatureService(wfs_url)

        # Send HTTP request to WFS and load data into GeoDataFrame
        response = requests.get(wfs_url)
        wfs_data = response.json()
        wfs_features = gpd.GeoDataFrame.from_features(wfs_data['features'])
        
        # Fetch bbox of Zeeland
        provincie_features = wfs_features[wfs_features['statnaam'] == 'Zeeland']
        # bbox_provincie = provincie_features.total_bounds

        provincie_features.plot(ax=ax)

        # Set plot limits
        ax.set_xlim([xll, xll + dx])
        ax.set_ylim([yll, yll + dy])

    # Debug code om te bekijken welke features er zijn en diverse andere eigeschappen
    if type_kaart == 'feature' and debug == True:

        print(wfs.identification.version)
        print(wfs.identification.title)
        print(wfs.identification.abstract)
        list(wfs.contents)
        
        [op.name for op in wfs.operations]
        print(wfs.getOperationByName('GetFeature').methods)
        print(wfs.getOperationByName('GetFeature').formatOptions)

    if rotation_angle != 0:
        return ax, img_data, bbox
    else:
        bbox = None
        return ax, img_data, bbox
    
#%%

def get_new_map(location, bbox, layer, coord_system, output_format, transparent, dx, dy, rotation_angle=0):
    wms_url = 'https://service.pdok.nl/brt/top10nl/wms/v1_0?'
    wms = WebMapService(wms_url)
    
    # Coördinaten transformeren naar Europees coördinatenstelsel, anders
    # sluiten kaarten niet goed op elkaar aan
    if location in ('anka', 'kvgt'):            
                    
        # Transformeer coördinaten
        transformer = Transformer.from_crs(cs_rd, cs_etrs, always_xy=True)
        bbox = transformer.transform_bounds(*bbox)
        
        img = wms.getmap(layers=[layer],
                     styles=[],
                     srs=coord_system,
                     bbox=bbox,
                     size=(dx, dy),
                     format=output_format,
                     transparent=transparent)           
    
        layer_belgium = 'GRB_BSK'
        wms_url_belgium = 'https://geo.api.vlaanderen.be/GRB-basiskaart/wms/service?request=GetCapabilities&service=WMS&version=1.3.0'
        wms_belgium = WebMapService(wms_url_belgium, version='1.3.0')
        
        img_belgium = wms_belgium.getmap(layers=[layer_belgium],
                     styles=[],
                     srs=coord_system,
                     bbox=bbox,
                     size=(int(dx), int(dy)),
                     format=output_format,
                     transparent=transparent)
        
        # Open afbeeldingen
        image_nl = Image.open(BytesIO(img.read())).convert('RGBA')
        image_be = Image.open(BytesIO(img_belgium.read())).convert('RGBA')            

        combined_image = Image.new('RGBA', (image_nl.width, image_nl.height))
        combined_image.paste(image_be, (0, 0), image_be)
        combined_image.paste(image_nl, (0, 0), image_nl)
        combined_image.save(f'images/{location}_combined_map.png')
        
        img = combined_image

    # Haal Nederlandse kaart op
    else:        

        img = wms.getmap(layers=[layer],
                      styles=[],
                      srs=coord_system,
                      bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                      size=(dx, dy),
                      format=output_format,
                      transparent=transparent)
        
        img = Image.open(img)

    return img  
    
def get_location_data_for_map(location):
    
    # Load data from YAML file
    with open('locaties.yaml', 'r') as file:
        df = pd.DataFrame(yaml.safe_load(file)['locations'])
    
        # Filter the DataFrame for the given location name
        result = df[df['location'] == location]
        if not result.empty:
            xll = result['xll'].values[0]
            yll = result['yll'].values[0]
            dx = result['dx'].values[0]
            dy = result['dy'].values[0]
            return xll, yll, dx, dy   
        else:
            raise ValueError(f"Invalid gebiedscode: {location}") 

# Apply the same transformation to the line points as to polygon
def transform_points_old(x, y, origin_x, origin_y, angle_degrees):
    
    # Translate points to the origin
    x_translated = np.array(x) - origin_x
    y_translated = np.array(y) - origin_y
    
    # Apply rotation
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    x_rotated = cos_angle * x_translated - sin_angle * y_translated
    y_rotated = sin_angle * x_translated + cos_angle * y_translated
    
    # Translate points back
    x_final = x_rotated + origin_x
    y_final = y_rotated + origin_y
    
    return x_final, y_final

def transform_points(x, y, origin_x, origin_y, angle_degrees):
    
    # Create the affine transformation matrix for translation to the origin
    translation_matrix_to_origin = np.array([[1, 0, -origin_x],
                                             [0, 1, -origin_y],
                                             [0, 0, 1]])
    
    # Create the affine transformation matrix for rotation
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                [sin_angle, cos_angle, 0],
                                [0, 0, 1]])
    
    # Create the affine transformation matrix for translation back from the origin
    translation_matrix_back = np.array([[1, 0, origin_x],
                                        [0, 1, origin_y],
                                        [0, 0, 1]])
    
    # Combine the transformations (matrix multiplication, order from right to left)
    transformation_matrix = translation_matrix_back @ rotation_matrix @ translation_matrix_to_origin
    
    # Apply the transformation to each point
    points = np.vstack((x, y, np.ones_like(x)))
    transformed_points = transformation_matrix @ points
    
    x_final, y_final = transformed_points[0, :], transformed_points[1, :]
    
    return x_final, y_final

def rotate_image(location, ax, img_data, coord_system, rotation_angle, bbox, map_cache=None):
    # Keep the requested RD extent as the plotting extent. For border locations the
    # WMS request is made in ETRS, whose axis-aligned bounds are larger because RD
    # and ETRS are locally rotated relative to each other.
    plot_bbox = bbox.copy()
    image_bbox = bbox.copy()
   
    # Coördinaten transformeren naar Europees coördinatenstelsel, anders
    # sluiten kaarten niet goed op elkaar aan
    if location in ('anka', 'kvgt'):        
        
        # RD New (EPSG:28992) to ETRS89 / UTM zone 31N (EPSG:25831)
        transformer_rd_to_etrs = Transformer.from_crs(cs_rd, cs_etrs, always_xy=True)
        bbox = transformer_rd_to_etrs.transform_bounds(*bbox)

        # Keep the enlarged RD bounds for raster placement after reprojection. The
        # final visible crop still uses plot_bbox below.
        transformer_etrs_to_rd = Transformer.from_crs(cs_etrs, cs_rd, always_xy=True)
        image_bbox = transformer_etrs_to_rd.transform_bounds(*bbox)

    # Calculate dx, dy and the coordinates of the other corners
    xll = bbox[0]
    yll = bbox[1]
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    xur, yur = xll + dx, yll + dy   

    image_path = f'images/{location}.tif'
    reprojected_image_path = f'images/{location}_reprojected.tif'

    # Reproject image, rotation is always done later and not saved as file
    if not os.path.exists(reprojected_image_path):

        # Decode the PNG to an array. Only needed when (re)creating the GeoTIFF;
        # np.array() forces the full pixel decode (expensive), so keep it out of
        # the hot path once the reprojected TIFF exists on disk.
        image_data = np.array(img_data)

        transform = (Affine.translation(xll, yll+dy)
                     * Affine.scale(dx / image_data.shape[1], dy / image_data.shape[0])
                     * Affine.scale(1, -1))       
          
        # Create a georeferenced raster dataset with the image data and
        # alpha channel based on png/img_data
        with rasterio.open(
            image_path, 'w', driver='GTiff', height=image_data.shape[0], width=image_data.shape[1],
            count=4, dtype=image_data.dtype, crs=coord_system, nodata=None, transform=transform
        ) as dst:
            # Write each color channel to the raster including the alpha channel
            for i in range(4):  # Assuming the image has three channels (R, G, B) plus alpha
                dst.write(image_data[:, :, i], i + 1)                    
        
        # Reproject the image back to cs_rd if image is in cs_etrs
        with rasterio.open(image_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, cs_rd, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': cs_rd,
                'transform': transform,
                'width': width,
                'height': height
            })
        
            with rasterio.open(reprojected_image_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=cs_rd,
                        nodata=None,
                        resampling=Resampling.nearest)

    # Read the reprojected image and rotate it. This is the expensive, ax-independent
    # step (GeoTIFF read + ndimage.rotate on 4 bands), identical for every parameter,
    # so cache the result and reuse it across figures.
    rot_key = ('rotated', location, rotation_angle)
    if map_cache is not None and rot_key in map_cache:
        rotated_rgba_tr, transform_of_image, bounds = map_cache[rot_key]
    else:
        # Open the reprojected image
        with rasterio.open(reprojected_image_path) as src:
            image_data = src.read()
            transform_of_image = src.transform
            bounds = src.bounds

        # Rotate each color band, stack array, transpose for correct order for show()
        rotated_bands = [ndimage.rotate(band, rotation_angle, reshape=True) for band in image_data]
        rotated_rgba = np.dstack([rotated_bands[i] for i in range(4)])
        rotated_rgba_tr = np.transpose(rotated_rgba, [2, 0, 1])

        if map_cache is not None:
            map_cache[rot_key] = (rotated_rgba_tr, transform_of_image, bounds)

    def _rotated_polygon_for_bbox(source_bbox):
        xll = source_bbox[0]
        yll = source_bbox[1]
        dx = source_bbox[2] - source_bbox[0]
        dy = source_bbox[3] - source_bbox[1]
        xur, yur = xll + dx, yll + dy
        origin_x, origin_y = xll + dx/2, yll + dy/2

        polygon = Polygon([(xll, yll), (xur, yll), (xur, yur), (xll, yur), (xll, yll)])
        polygon_at_origin = translate(polygon, -origin_x, -origin_y)
        rotated_polygon_at_origin = shapely_rotate(polygon_at_origin, rotation_angle, origin=(0, 0))
        return translate(rotated_polygon_at_origin, origin_x, origin_y)

    def _limits_for_polygon(rotated_polygon):
        x_poly, y_poly = rotated_polygon.exterior.xy
        xx_poly = x_poly.tolist()
        yy_poly = y_poly.tolist()

        if rotation_angle == -90:
            # Switch x and y
            ylims = [xx_poly[3], xx_poly[1]]
            xlims = [yy_poly[0], yy_poly[2]]
        elif rotation_angle > 0:
            xlims = [xx_poly[0], xx_poly[2]]
            ylims = [yy_poly[1], yy_poly[3]]
        elif rotation_angle < 0:
            xlims = [xx_poly[3], xx_poly[1]]
            ylims = [yy_poly[0], yy_poly[2]]

        return xlims, ylims

    def _clip_limit_to_coverage(lim, coverage):
        lim_min, lim_max = min(lim), max(lim)
        cov_min, cov_max = min(coverage), max(coverage)
        clipped_min = max(lim_min, cov_min)
        clipped_max = min(lim_max, cov_max)
        if clipped_min > clipped_max:
            raise ValueError('Map crop does not overlap available map coverage')
        if lim[0] > lim[1]:
            return [clipped_max, clipped_min]
        return [clipped_min, clipped_max]

    def _limits_inside(inner_xlim, inner_ylim, outer_xlim, outer_ylim, tolerance=1e-6):
        return (
            min(inner_xlim) >= min(outer_xlim) - tolerance
            and max(inner_xlim) <= max(outer_xlim) + tolerance
            and min(inner_ylim) >= min(outer_ylim) - tolerance
            and max(inner_ylim) <= max(outer_ylim) + tolerance
        )

    image_polygon = _rotated_polygon_for_bbox(image_bbox)
    plot_polygon = _rotated_polygon_for_bbox(plot_bbox)

    # Calculate offsets from the enlarged image bbox. The final crop is calculated
    # separately from plot_bbox, so the route remains aligned without manual margins.
    x_off = (image_bbox[0] - image_polygon.bounds[0]) / transform_of_image[0]
    y_off = (image_bbox[1] - image_polygon.bounds[1]) / transform_of_image[4]

    # Combine transformation of image with translation by rotation
    combi_transform = transform_of_image * Affine.translation(-x_off, y_off)

    xlims, ylims = _limits_for_polygon(plot_polygon)
    image_xlims, image_ylims = _limits_for_polygon(image_polygon)

    if location in ('anka', 'kvgt'):
        if np.abs(rotation_angle) == 90:
            crop_xlim, crop_ylim = ylims, xlims
            coverage_xlim, coverage_ylim = image_ylims, image_xlims
        else:
            crop_xlim, crop_ylim = xlims, ylims
            coverage_xlim, coverage_ylim = image_xlims, image_ylims

        if not _limits_inside(crop_xlim, crop_ylim, coverage_xlim, coverage_ylim):
            crop_xlim = _clip_limit_to_coverage(crop_xlim, coverage_xlim)
            crop_ylim = _clip_limit_to_coverage(crop_ylim, coverage_ylim)
            if np.abs(rotation_angle) == 90:
                ylims, xlims = crop_xlim, crop_ylim
            else:
                xlims, ylims = crop_xlim, crop_ylim
    
    show(rotated_rgba_tr, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], 
         transform=combi_transform, ax=ax)    
    
    # Disabled, needed for debugging rotation of image
    # ax.plot(x_poly, y_poly, color='red', linewidth=2, label='Rotated Polygon', zorder=100)    
    # ax.hlines(yll, xll, xur, colors='orange', linewidth=3)
    # ax.hlines(yur, xll, xur, colors='orange', linewidth=3)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # Set number of x-ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  # Set number of y-ticks

    # The ticks must be fetched AFTER the final (cropped) limits are set: MaxNLocator
    # recomputes the tick labels for the active view, so reading them on the pre-crop
    # view (the full image bbox) leaves the grid anchored to a different step than the
    # labels that are actually rendered. Pull them from the locator on the final limits
    # so grid lines and y-axis labels stay in sync for every location.
    def _ticks_for(axis, lim):
        return axis.get_major_locator().tick_values(min(lim), max(lim))

    if np.abs(rotation_angle) == 90:

        ax.set_xlim(ylims)
        ax.set_ylim(xlims)

        x_ticks = _ticks_for(ax.xaxis, ax.get_xlim())
        y_ticks = _ticks_for(ax.yaxis, ax.get_ylim())

        # Calculate new ticks based on the rotation angle
        dy_grid = xlims[1] - xlims[0]
        interval_x_ticks = x_ticks[1] - x_ticks[0]

        dx_grid = ylims[1] - ylims[0]
        interval_y_ticks = y_ticks[1] - y_ticks[0]

        # Set rotation_angle to zero for gridlines and switch xlims and ylims
        rotation_angle = 0
        xlims_temp = xlims
        ylims_temp = ylims
        xlims = ylims_temp
        ylims = xlims_temp

    else:

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        x_ticks = _ticks_for(ax.xaxis, ax.get_xlim())
        y_ticks = _ticks_for(ax.yaxis, ax.get_ylim())

        # Calculate new ticks based on the rotation angle
        dy_grid = ylims[1] - ylims[0]
        interval_x_ticks = x_ticks[1] - x_ticks[0]

        dx_grid = xlims[1] - xlims[0]
        interval_y_ticks = y_ticks[1] - y_ticks[0]
    
    # Generate values before and after the original ticks
    before_x_ticks = np.arange(x_ticks[0] - 5 * interval_x_ticks, x_ticks[0], interval_x_ticks)
    after_x_ticks = np.arange(x_ticks[-1] + interval_x_ticks, x_ticks[-1] + 6 * interval_x_ticks, interval_x_ticks)
    x_ticks_extended = np.concatenate((before_x_ticks, x_ticks, after_x_ticks))

    x_grid_end = x_ticks_extended - dy_grid * np.tan(np.deg2rad(rotation_angle))          

    before_y_ticks = np.arange(y_ticks[0] - 5 * interval_y_ticks, y_ticks[0], interval_y_ticks)
    after_y_ticks = np.arange(y_ticks[-1] + interval_y_ticks, y_ticks[-1] + 6 * interval_y_ticks, interval_y_ticks)
    y_ticks_extended = np.concatenate((before_y_ticks, y_ticks, after_y_ticks))
    
    y_grid_end = y_ticks_extended + dx_grid * np.tan(np.deg2rad(rotation_angle))    
    
    # Plot (rotated) vertical grid lines
    for x_start, x_end in zip(x_ticks_extended, x_grid_end):
        ax.plot([x_start, x_end], [ylims[0], ylims[1]], color='gray', linewidth=0.5, alpha=0.5)
    
    # Plot (rotated) horizontal grid lines
    for y_start, y_end in zip(y_ticks_extended, y_grid_end):
        ax.plot([xlims[0], xlims[1]], [y_start, y_end], color='gray', linewidth=0.5, alpha=0.5)
               
    bbox = bounds

    return ax, rotated_rgba_tr, bbox

#%%
    
def test_plot_kaartje(gebiedscode):
    fig, ax = plt.subplots()
    plot_kaartje(gebiedscode, ax, force_refresh_image=True)

# test_plot_kaartje('RO')
