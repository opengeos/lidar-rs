# Command-Line Reference

The `lidar` CLI wraps every algorithm and geospatial utility in `lidar-rs`:

```sh
# Smooth a DEM
lidar median-filter dem.tif out/median.tif -k 3
lidar mean-filter dem.tif out/mean.tif -k 3
lidar gaussian-filter dem.tif out/gaussian.tif -s 1.0

# Extract sinks and refine DEM
lidar extract-sinks out/median.tif out --min-size 1000

# Delineate nested depression hierarchy (Wu et al. 2019)
lidar delineate-depressions out/sink.tif out --min-size 1000 --min-depth 0.3 --interval 0.3 --level-shp

# Delineate elevated features (mounts / canopies)
lidar delineate-mounts dsm.tif out_mounts --min-size 1000 --min-height 0.3 --interval 0.3

# Raster & vector utilities
lidar polygonize out/region.tif out/regions.geojson --field-name id
lidar mosaic tile1.tif tile2.tif -o mosaic.tif
lidar reproject dem.tif dem_4326.tif --crs EPSG:4326 -r bilinear
lidar resample dem.tif dem_2m.tif --resolution 2.0 -r bilinear
lidar clip dem.tif mask.gpkg dem_clip.tif
lidar to-cog dem.tif dem_cog.tif --compression DEFLATE
lidar add-crs dem.tif EPSG:26915 -o dem_crs.tif
lidar join-tables out/regions.shp out/regions_info.csv out/joined.gpkg
lidar csv-to-vector points.csv points.shp --latitude latitude --longitude longitude

# LiDAR point clouds (LAS, LAZ, COPC, PLY, E57)
lidar convert-lidar input.las output.laz --point-format 6
lidar lidar-to-dsm input.laz dsm.tif --resolution 1.0 --radius 1.5
lidar info dem.tif
```
