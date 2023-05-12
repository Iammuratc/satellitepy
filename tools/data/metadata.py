try:
  import gdal, osr
except ModuleNotFoundError:
  from osgeo import gdal, osr
from shapely.geometry import Polygon
import numpy as np
import pyproj
from shapely.ops import transform

def get_geo_polygon(gdal_ds):
    img_width, img_height = gdal_ds.RasterXSize, gdal_ds.RasterYSize
    x_top_left, xres, _, y_top_left, _, yres  = gdal_ds.GetGeoTransform()
    poly_coords = (
        (x_top_left, y_top_left),
        (x_top_left + img_width * xres, y_top_left),
        (x_top_left + img_width * xres, y_top_left + img_height * yres),
        (x_top_left, y_top_left + img_height * yres),
    )
    return Polygon(poly_coords)

def get_geo_data(file):
    result = {}
    ds = gdal.Open(file, gdal.GA_ReadOnly)

    result["srs"] = str(ds.GetProjection())
    img_width, img_height = ds.RasterXSize, ds.RasterYSize
    _, xres, _, _, _, yres  = ds.GetGeoTransform()
    
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    result["srid"] = proj.GetAttrValue('AUTHORITY',1)

    geom_polygon = get_geo_polygon(ds)

    result["spatial_resolution_dx"] = xres
    result["spatial_resolution_dy"] = yres
    result["image_width"] = img_width
    result["image_height"] = img_height
    result["geometry"] = geom_polygon.wkt

    if result["srid"] == "4326":
        result["geography"] = result["geometry"]
    else:
        orig_projection = pyproj.CRS(result["srs"])
        target_projection = pyproj.CRS(4326)
        project = pyproj.Transformer.from_crs(orig_projection, target_projection,always_xy=True).transform
        result["geography"] = transform(project, geom_polygon).wkt
    
    del ds
    return result

import sys

if __name__ == "__main__":
    #image_path = '/home/simon/unibw/data/dem/gaofen/train/images/1.tif'
    image_path = sys.argv[1]
    print(get_geo_data(image_path))
