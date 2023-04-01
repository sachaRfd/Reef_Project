# Reef_Generator Project

## Scripts:
- Create_Images: Takes filepath to the Sentinel Image data as input then creates images with different bands.
- Download_Files: Takes coordinates as input and downloads data, if available.
- Get_Individual_Reef: Takes path to RGB image and creates images of all the reefs present in that image.
    - This function has some bugs, Not sure how to fix yet --> Happens when the reefs are very small I think --> More work to be done.

- Small Reef dataset --> uses smaller chunks of data to test initial model and to see where to improve on
    - Downloads each image as numpy array and saves it locally as npy file 
    - For now it is each image in one file --> LATER WILL have to group them together --> but this led to errors when I tried

## Basic Notebooks:
- Super Simple implementations of the models 
    - For now I am converting from tensorflow to pytorch for better memory managment and testing
    - Check the PYTORCH Notebooks for notes and next steps