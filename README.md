# Reef_Generator Project

## Scripts:
- Create_Images: Takes filepath to the Sentinel Image data as input then creates images with different bands.
- Download_Files: Takes coordinates as input and downloads data, if available.
- Get_Individual_Reef: Takes path to RGB image and creates images of all the reefs present in that image.
    - Bugs fixed --> works fine now.

- Small Reef dataset --> uses smaller chunks of data to test initial model and to see where to improve on
    - Downloads each image as numpy array depending on an image_shape parameter
    - Resizes the smaller images to a the image_shape parameter and ignores the larger images
    - Min-Max normalises the images
    - Saves each image-array locally as npy file 
        - For now it is each image in one file --> LATER WILL have to group them together --> but this led to errors when I tried

## Basic Notebooks:
Simple implementations of the models:

   - Pytorch notebook: just an implementation of an autoencoder --> Just to see if it worked
   - MNIST notebook: double checking model works with the given dataset --> also used to understand how to run code
   - Basic Reef Notebook: Implementation of the model with our reef data
    
### Basic Reef notebook:
- Worked fine on small reefs of size 26x26, 50x50 and 75x75--> but size of the dataset was still significantly small
- Model works when each reef is padded instead of just resized.
