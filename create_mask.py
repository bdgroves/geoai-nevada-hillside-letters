import fiona
from rasterio.features import rasterize
import rasterio
import rasterio.warp
import numpy as np

vector_path = "rock_letters.shp"
raster_path = "m_3911926_sw_11_060_20220614.tif"
mask_path = "rock_letters_mask_fixed.tif"

# Open raster for transform
with rasterio.open(raster_path) as src:
    meta = src.meta.copy()
    width, height = src.width, src.height
    transform = src.transform
    raster_crs = src.crs

# Open vector and reproject if needed
with fiona.open(vector_path, "r") as shp:
    if shp.crs != raster_crs:
        import shapely.geometry
        import shapely.ops
        import pyproj
        from functools import partial

        project = partial(
            pyproj.transform,
            pyproj.CRS(shp.crs),
            pyproj.CRS(raster_crs)
        )

        geoms = [
            shapely.ops.transform(project, shapely.geometry.shape(f["geometry"])).__geo_interface__
            for f in shp
        ]
    else:
        geoms = [f["geometry"] for f in shp]

# Rasterize
mask = rasterize(
    geoms,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    default_value=1,
    dtype="uint8"
)

# Save raster
meta.update({"count": 1, "dtype": "uint8"})
with rasterio.open(mask_path, "w", **meta) as dst:
    dst.write(mask, 1)

print("Mask created:", mask_path)
print("Unique values in mask:", np.unique(mask))
print("Non-zero pixels:", np.count_nonzero(mask))