import rasterio
import numpy as np

# Paths
naip_path = "C:/Users/brook/Documents/geoai-rockletters/m_3911926_sw_11_060_20220614.tif"
mask_path = "C:/Users/brook/Documents/geoai-rockletters/rock_letters_mask_fixed.tif"
output_path = "C:/Users/brook/Documents/geoai-rockletters/naip_letter_only.tif"

# Open NAIP image
with rasterio.open(naip_path) as naip:
    naip_data = naip.read()        # Read all bands
    meta = naip.meta.copy()        # Copy metadata for output

# Open mask
with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)        # Read single-band mask

# Apply mask: multiply each band by mask
masked_data = naip_data * mask   # Broadcast works if mask shape matches NAIP

# Update metadata (optional: make nodata 0)
meta.update({"dtype": naip_data.dtype, "count": naip.count, "nodata": 0})

# Write masked output
with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(masked_data)

print("Masked NAIP image saved to:", output_path)