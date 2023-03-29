import os
import rasterio
import numpy as np


class Create_Image:

    def __init__(self, path: str) -> None:
        self.path_to_image = path
        self.files = os.listdir(self.path_to_image)
        # filter for bands (assuming they end in '.jp2')
        self.bands = [file for file in self.files if file.endswith('.jp2')]
        # Create Tiff File to get width ect..:
        self.test_path = self.path_to_image + self.bands[0]  # To get width
        self.test_band = rasterio.open(
            self.test_path, driver='JP2OpenJPEG')  # Open the file

    # create 3 count Tiff File:
    def RGB_tiff(self, root_path: str, file_name: str) -> None:
        """
        Creates a RGB tiff file from the JP2 files in the path_to_image folder.

        Parameters
        ----------
        root_path : str
            The path to the folder where the tiff file will be stored.
        file_name : str
            The name of the tiff file.

        Returns
        -------
        None.
        """
        # Create Tiff File
        trueColor = rasterio.open(root_path + file_name + ".tiff", 'w', driver='Gtiff',  # noqa
                                  width=self.test_band.width, height=self.test_band.height,  # noqa
                                  count=3,
                                  crs=self.test_band.crs,
                                  transform=self.test_band.transform,
                                  dtype=self.test_band.dtypes[0]
                                  )
        # loop through vidible bands and store them in TIFF file
        for band in self.bands:
            if 'B02' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 3)
                print("Blue Done")
            elif 'B03' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 2)
                print("Green Done")
            elif 'B04' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 1)
                print("Red Done")
        trueColor.close()

    # create 4 count Tiff File:

    def TIC_RGB_tiff(self, root_path: str, file_name: str) -> None:
        """
        Creates a TIC RGB tiff file from the JP2 files in the path_to_image folder.  # noqa

        Parameters
        ----------
        root_path : str
            The path to the folder where the tiff file will be stored.
        file_name : str
            The name of the tiff file.

        Returns
        -------
        None.
        """
        # Create Tiff File
        trueColor = rasterio.open(root_path + file_name + ".tiff", 'w', driver='Gtiff',  # noqa
                                  width=self.test_band.width, height=self.test_band.height,  # noqa
                                  count=4,
                                  crs=self.test_band.crs,
                                  transform=self.test_band.transform,
                                  dtype=self.test_band.dtypes[0]
                                  )
        # loop through vidible bands and store them in TIFF file
        for band in self.bands:
            if 'B02' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 3)
                print("Blue Done")
            elif 'B03' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 2)
                print("Green Done")
            elif 'B04' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 1)
                print("Red Done")
            elif 'TIC' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    trueColor.write(src.read(1), 4)
                print("TIC DONE")
        trueColor.close()

    def NDVI_Index(self, root_path: str, file_name: str) -> None:
        """
        Creates a NDVI tiff file from the JP2 files in the path_to_image folder.  # noqa

        Parameters
        ----------
        root_path : str
            The path to the folder where the tiff file will be stored.
        file_name : str
            The name of the tiff file.

        Returns
        -------
        None.

        Normalised Difference Vegetation Index:
        (B8 - RED) / (B8 + RED)
        """
        # Create Tiff File
        trueColor = rasterio.open(root_path + file_name + ".tiff", 'w', driver='Gtiff',  # noqa
                                  width=self.test_band.width, height=self.test_band.height,  # noqa
                                  count=3,  # 1 Count for NDVI
                                  crs=self.test_band.crs,
                                  transform=self.test_band.transform,
                                  dtype=np.float64
                                  )

        # loop through vidible bands and store them in TIFF file
        for band in self.bands:
            if 'B04' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    # trueColor.write(src.read(1), 1)
                    red_band = src.read(1)
                    print("Red Done")
            elif 'B02' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    blue_band = src.read(1)
                    # trueColor.write(src.read(1), 3)
                print("Blue Done")
            elif 'B03' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    green_band = src.read(1)
                    # trueColor.write(src.read(1), 2)
                print("Green Done")
            elif 'B08' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    NIR_band = src.read(1)
                print("Near Infra-red found")

        # Normalise bands: between 0 and 1
        print("Normalising bands ...")
        NIR_band = (NIR_band - NIR_band.min()) / (NIR_band.max() - NIR_band.min())  # noqa
        red_band = (red_band - red_band.min()) / (red_band.max() - red_band.min())  # noqa
        green_band = (green_band - green_band.min()) / (green_band.max() - green_band.min())  # noqa
        blue_band = (blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())  # noqa

        print("Calculating Index ...")
        denominator = (NIR_band + red_band)
        NDVI_index = np.where(denominator == 0, 1e-6, (NIR_band - red_band) / denominator)  # noqa

        print("Writing to Tiff File ...")
        trueColor.write(NDVI_index, 1)  # Write the index to the 1st band
        trueColor.write(green_band, 2)
        trueColor.write(blue_band, 3)
        trueColor.close()  # Close file
        print(f"Min is: {NDVI_index.min()}, max is : {NDVI_index.max()}")

    def MNDWI_Index(self, root_path: str, file_name: str) -> None:
        """
        Creates a MNDWI tiff file from the JP2 files in the path_to_image folder.

        Parameters
        ----------
        root_path : str
            The path to the folder where the tiff file will be stored.
        file_name : str
            The name of the tiff file.

        Returns
        -------
        None.


        Only works if data includes the B9 band

        Modified Normalised Difference Water Index (MNDWI)  --> Check where water and land meet  # noqa
        (Green - B9) / (Green + B9)

        Comments:
            - Not great because clouds are still aparent
            - Can still see difference between land and sea quite well --> BUT WITH Clouds

        """
        # Create Tiff File
        trueColor = rasterio.open(root_path + file_name + ".tiff", 'w', driver='Gtiff',  # noqa
                                  width=self.test_band.width, height=self.test_band.height,  # noqa
                                  count=1,  # 1 Count for NDVI
                                  crs=self.test_band.crs,
                                  transform=self.test_band.transform,
                                  dtype=np.float64
                                  )

        # loop through vidible bands and store them in TIFF file
        for band in self.bands:
            if 'B03' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    green_band = src.read(1, out_shape=(src.count, 1830, 1830),  # noqa Resampling to size of the SWIR band
                                          resampling=rasterio.enums.Resampling.bilinear  # noqa
                                          )
                print("Green Done")
            elif 'B09' in band:
                with rasterio.open(self.path_to_image + band) as src:
                    SWIR_band = src.read(1, out_shape=(src.count, 1830, 1830),  # noqa Resampling to size of the SWIR band
                                          resampling=rasterio.enums.Resampling.bilinear  # noqa
                                         )
                print("SWIR 1 Done")

        # Normalise bands: between 0 and 1
        print("Normalising bands ...")
        SWIR_band = (SWIR_band - SWIR_band.min()) / (SWIR_band.max() - SWIR_band.min())  # noqa
        green_band = (green_band - green_band.min()) / (green_band.max() - green_band.min())  # noqa

        print("Calculating Index ...")

        denominator = (SWIR_band + green_band)

        MNDWI_index = np.where(denominator == 0, 1e-6, (SWIR_band - green_band) / denominator)  # noqa
        trueColor.write(MNDWI_index, 1)  # Write the index to the 4th band
        trueColor.close()  # Close file
        print(f"Min is: {MNDWI_index.min()}, max is : {MNDWI_index.max()}")


if __name__ == '__main__':
    root_path = input("What is the path to the images file:  ")
    print()

    img_data_path = None  # Initialize img_data_path to None

    # Find R20m folder
    for root, dirs, files in os.walk(root_path):
        if "R10m" in dirs:
            # Get Full Path:
            img_data_path = os.path.join(root, "R10m/")
            # img_data_path += "/"
            break

    if img_data_path:
        print("File was found at:  ", img_data_path)
    else:
        print("No R20m File present --> Will use IMG_DATA instead !!")

        # Find IMG_DATA folder
        for root, dirs, files in os.walk(root_path):
            if "IMG_DATA" in dirs:
                # Get Full Path:
                img_data_path = os.path.join(root, "IMG_DATA")
                img_data_path += "/"
                break

        if img_data_path:
            print("File was found at:  ", img_data_path)
        else:
            print("Neither R20m nor IMG_DATA found in the given path - Make sure your folder is correct")  # noqa
            exit()

    # Instantiate the image class
    image = Create_Image(img_data_path)

    # Get user input for num_bands
    num_bands = input("""Would you want to get a:
        NDVI: Normalised Difference Vegetation Index
        MNDWI: Modified Normalised Difference Water IndeX
        2: band (NIR and Red)
        3: band (RGB)
        4: band (TIC + RGB)?
        """)

    # Process the selected num_bands
    if num_bands == '3':
        file_name = input("What would you like to call your TIFF file:  ")
        image.RGB_tiff(root_path, file_name)
        print("3 Band image is done")
    elif num_bands == '4':
        file_name = input("What would you like to call your TIFF file:  ")
        image.TIC_RGB_tiff(root_path, file_name)
        print("4 Band Image is done")
    elif num_bands == "NDVI":
        file_name = input("What would you like to call your TIFF file:  ")
        image.NDVI_Index(root_path, file_name)
        print("NDVI Index calculated")
    elif num_bands == "MNDWI":
        file_name = input("What would you like to call your TIFF file:  ")
        image.MNDWI_Index(root_path, file_name)
        print("MNDWI Index calculated")
    else:
        print("Please Retry with correct integer")
