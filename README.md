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
    - For now I am converting from tensorflow to pytorch for better memory managment and testing  --> Not great
    - Check the PYTORCH Notebooks for notes and next steps
    
### Basic Reef notebook:
- Worked fine on small reefs of size 26x26  --> but size of the dataset was literally 100 images 
- When the data included 50x50 imgages --> The images that were resized (those smaller than 50 in the original dataset) are really bad but the others are better
    - May have to rething the resizing of the images
