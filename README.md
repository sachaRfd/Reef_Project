# Robust-Autoencoder for reef images: 

## Project Description:
- Read TIFF Sentinel 2 satelite images
- Create Robust Auto-encoder style model for reef generation:
    - Should be able to split image into a low rank L matrix and a sparse S matrix that denoises the image and removes Outliers.
    - The noise we are looking to remove is induced by atmospheric variety, and different lighting conditions between the images.
    - We are looking to only use "clean" images for the models, meaning there is no induced noise to the images before training.



## Script Usage: 

- Sentinel 2 images were downloaded using the python sentinel API. 
    - Using coordinates from the Reef Shape file. This file can be read from QGIS and the coordinates can be taked from there. 
    - Use the ``Download_files.py`` file and input your desired coordinates to download. Be aware that the current program will try and download the images if they are available in the Sentinel Database. If the images are not available and the program triggers a Warning, Re-run the script after around 20-40 minutes.
    - The current download script is ready prepared to download the image with the lowest amount of cloud coverage, and 2A type images. If you would like to make sure that no cloud coverage is present in your final image, make sure to check the image in QGIS.
- Once your desired .zip file has been downloaded, unzip the folder and run the ``create_images.py`` script. This script will allow you to choose from a variety of image types to eb created from your satelite data. 
    - For a simple RGB image, choose number 3 for the number of channels, and you may call your image RGB for example. 
    - After running this script with the above inputs, a 3 channels .TIFF file will be created in the same repository where your satelite data is located in.
    - For visualisation, and to make sure that your image has been created correctly, open the tiff file in QGIS. 
        - There could be issues with the satelite data, which could mean you have images with all sorts of colourful pixels. This is to do with the wrong channels being read. It can be fixed by re-running the code, but changing lines ``260`` and ``262`` to either ``R10m`` or ``R20m``. Sometimes the erros persits, and you may have to redownload new data using the download script. 

- Once you have created your large TIFF image, you may want to run the ``get_indiviudal_reef.py`` script. This script will mask all the reefs that are present in your image and create individual tiff files containing the pixels correspondonding to known reefs, into a directory you have chosen. 
    - The script makes sure to create masks that are contained in the image, if a reef is only partly contained in the image, the reef will be clipped. 

- Once the clipped reefs have been placed in your desired directory, you are able to train your Robust Auto-encoder for outlier prediction and noise removal using the ``train_L12_RAE.py`` script.

- The output images were created using the following 2 scripts: 
    - ``Model_output.py`` to get the normal output image from the model.
    - ``Model_output_with_noise.py`` to get the output images with low-noise and high-noise added to the image.



## Dataset: 

- Large images of reefs from the pacific, indian ocean as well as off the coast of Australia. 
- The images were looked at initially to make sure they were not corrupted
- The images were then clipped to only contain the images of the reefs. 
- A training and test set were then created from the clipped images. Initially, we started small with images of reefs of size 28 by 28 pixels. 
    - This dataset contained only reefs of size 28 by 28, meaning clipped reefs that were larger than this size were cropped into smaller images. 
    - We also removed from the dataset images that contained the same pixel values. These images are most likly corrupt images from the clipped reef script. 
    - The reefs were then scaled down with Min-Max normalisation, which is important in our case as our model predicts pixel values between 0 and 1.
    - The dataset is then split into a 90% training (around 127_000 images) and 10% testing set.

- Example images from dataset: 
<br>

<div style="text-align:center">
  <img src="Project Notes/dataset_images_examples/3_0.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/5_18.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/5_55.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/5_61.png" alt="image" width="100"/>
  <br>
  <img src="Project Notes/dataset_images_examples/5_78.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/5_95.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/5_109.png" alt="image" width="100"/>
  <img src="Project Notes/dataset_images_examples/6_510.png" alt="image" width="100"/>
</div>

- Observations: 

    - Not all images have the same intensity of pixels. 
    - Some images are anomalous, with clear seperation between black corrupt pixels from the clipped and the wanted image. A way around this could be to add a line of code in the dataset class that removes corrupt image that have maybe over 40% black 0.0 pixels.



## L-2.1 Model:

This model may look like a simple feed-forward auto-encoder but it differs from the classic by its training method. 

This model has a L-2.1 Regularisation in its training where the input image is split into L and S matrices which stand for a low-rank matrix and a Sparse matrix. The lambda parameter controls how sparse the S matrix should be.

The L matrix should contain the main information in the image whereas the S matrix contains the noise and outlier pixels from the image. 

## Training: 

- The model was trained in a similar fashion as the paper we followed, meaning in between each splitting of the images into L and S matrices, the auto-encoder was trained for 10 epochs. 
- Multiple lambda values were tested in a range from 100 to 0.001, but more research will be done on their usage in the optimisation phase of the study.


## Observations: 


- Simple model output: model with Lambda value of 0.5:

    - Plots showing True images, L and S splits from the last training iteration:

        <br>
        <div style="text-align:center">
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_0.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_1.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_2.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_3.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/example_0.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/example_1.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/example_2.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/example_3.png" alt="image" width="300"/>
        <img src="Project Notes/L21_training_files/Short_inner_iterations/lambda_0.5/example_4.png" alt="image" width="300"/>
        </div>
        <br>



- Model Output of different bands and overall 3 band images: 
        <br>
        <div style="text-align:center">
        <img src="Project Notes/Simple_reconstruction/Blue_bands_20170825.png" alt="image" width="300"/>
        <img src="Project Notes/Simple_reconstruction/Green_bands_20170825.png" alt="image" width="300"/>
        <img src="Project Notes/Simple_reconstruction/Red_bands_20170825.png" alt="image" width="300"/>
        <img src="Project Notes/Simple_reconstruction/RGB_bands_20170825.png" alt="image" width="300"/>
        </div>
        <br>


- Model Output with random Gaussian noise added to the input images :

    <br>
    <div style="text-align:center">

    Blue Band:
    <br>
    <img src="Project Notes/added_noise/blue_band.png" alt="image" width="800"/>
    <br>
    Green Band: 
    <br>
    <img src="Project Notes/added_noise/green_band.png" alt="image" width="800"/>
    <br>

    Red Band:
    <br>
    <img src="Project Notes/added_noise/red_band.png" alt="image" width="800"/>
    </div>
    <br>
    
    - Might not be the most appropriate way of testing our model, but we can see that the model is able to output the most important features in the input image, wether there is noise present in the image or not. 


<br>

## Future work: 

- Optimisation for the Lambda parameter.

- Dataset is still not the best:
    - Using 28 by 28 images.
    - Using mean average of the 3 colour bands --> Paper suggests only using the Green Band:  https://www.mdpi.com/2072-4292/13/23/4948
    - Some images can have more than half of its pixels black --> This is because of the clipping script which returns images that only contain the pixels which are reefs from the reefs shape file.
        - Could work around this by either removing these images from training set, or at least reducing their number so they can be detected as anomalies. 