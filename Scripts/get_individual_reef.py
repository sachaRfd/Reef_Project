import geopandas as gpd
import sys
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np
import traceback


class read_reefs:
    # "Reef_12/RGB_test.tiff"
    def __init__(self, input_image: str, output_path: str, reef_number: int) -> None:  # noqa
        # Load the reef shapefile as a GeoDataFrame
        print("Reading Shape file ...")
        self.gdf = gpd.read_file("Reef_Shape_Files_2m_error/Reef_Shape_Files_2m_error.shp")  # noqa
        self.file_location = input_image
        self.output_path = output_path
        self.reef_number = reef_number

    def read_each_reef(self) -> None:
        """
        Reads the reef shapefile and the tiff file and creates a new tiff file for each reef in the tiff file.  # noqa        

        Returns
        -------
        None
        """
        # Reads the RGB image
        with rasterio.open(self.file_location) as src:

            # Get the bounding box of the tiff file
            # is equivalent to: box(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])  # noqa
            tiff_extent = box(*src.bounds)

            # Reproject to match CRS
            gdf_reprojected = self.gdf.to_crs(src.crs)
            # Remove polygons with INF or NaN values
            gdf_reprojected = gdf_reprojected[~np.isinf(gdf_reprojected.geometry.bounds).any(axis=1)]
            gdf_reprojected = gdf_reprojected[~np.isnan(gdf_reprojected.geometry.bounds).any(axis=1)]



            # New GeoDataFrame with the bounding box as a Polygon geometry
            print("Creating GeoDataframe ...")
            bbox_gdf = gpd.GeoDataFrame(
                {'geometry': [tiff_extent]}, crs=src.crs)

            # Use Intersect to find reefs that match the locations of the polygons from the file  # noqa
            print("Finding reefs in image ...")
            selected_reefs = gdf_reprojected[gdf_reprojected.intersects(
                bbox_gdf.unary_union)]
            print(f"There are {selected_reefs.shape[0]} reefs in this image !")
            print(selected_reefs.head(5))

            # Clip the tiff file if not empty:
            clipped = None
            if selected_reefs.empty:
                print("No reefs found in the tiff file extent.")

            else:
                # Convert to CRS of the tiff file
                selected_reefs = selected_reefs.to_crs(src.crs)
                # print("SRC: CRS = ", src.crs)
                # print("Reef_list ", selected_reefs.crs)

                # # Get the geometry as shapely polygons
                # reef_geoms = selected_reefs.geometry.tolist()
                # Get the geometry as shapely polygons, removing any NaN values
                reef_geoms = [geom for geom in selected_reefs.geometry.tolist() if not geom.is_empty]

                for count, reef in enumerate(reef_geoms):
                    # Has to be iterable list to be masked
                    reef_as_list = [reef]

                    # Clip the reef
                    clipped, out_transform = mask(src, reef_as_list, crop=True)

                    # Update transform
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": clipped.shape[1],
                        "width": clipped.shape[2],
                        "transform": out_transform,
                    })

                    # Write the clipped image to a new tiff file
                    print(
                        f"Creating masked Reefs [{count+1}|{len(reef_geoms)}]")
                    with rasterio.open(self.output_path + "reef_" + str(self.reef_number) + "_" + str(count) + ".tiff", "w", **out_meta) as dst:  # noqa
                        dst.write(clipped)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <path/of/image> <path_directory> <Image Number>")
        sys.exit(1)
    
    # Get the inputs: 
    input_path, output_path, reef_number = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
    
    # have to change both the path and the reef Number
    # input_path = "Reef_11/RGB.tiff"
    # output_path = "Clipped_Reefs/clean/"
    # reef_number = 11
    setup = read_reefs(input_path, output_path, reef_number)
    setup.read_each_reef()
