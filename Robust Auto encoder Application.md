# Current Progress on Project: 

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


## Dataset:

- Large images of reefs from the pacific, indian sea as well as off the coast of Australia. 
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
  <img src="dataset_images_examples/3_0.png" alt="image" width="100"/>
  <img src="dataset_images_examples/5_18.png" alt="image" width="100"/>
  <img src="dataset_images_examples/5_55.png" alt="image" width="100"/>
  <img src="dataset_images_examples/5_61.png" alt="image" width="100"/>
  <br>
  <img src="dataset_images_examples/5_78.png" alt="image" width="100"/>
  <img src="dataset_images_examples/5_95.png" alt="image" width="100"/>
  <img src="dataset_images_examples/5_109.png" alt="image" width="100"/>
  <img src="dataset_images_examples/6_510.png" alt="image" width="100"/>
</div>

- Observations: 

    - Not all images have the same intensity of pixels. 
    - Some images are anomalous, with clear seperation between black corrupt pixels from the clipped and the wanted image. A way around this could be to add a line of code in the dataset class that removes corrupt image that have maybe over 40% black 0.0 pixels.

## The models: 

- The models were adapted from the following paper: Anomaly Detection with Robust Deep Autoencoders (https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf)
    
    - The main change to the models were code for plotting intermediate results, as well as adding dropout layers for better generalisation of the model. 

- L-1 Robust Auto-encoder Model: 

- L-2,1 Robust Auto-encoder Model: 


## Training

- First used the L1- Model but quickly switched to using the L2,1 model instead, as it is this one that is able to both de-noise and find outlier images. 

- Two different training runs were conducted with varying Lambda values: 
    - The first was conducted with 80 Inner iterations (training of the auto-encoder) and 10 outer iterations (Splitting of images into the L and S matrix for further AE training).

    - The second training setup was conducted with 10 Inner iterations, and 200 outer iterations, close to what was done with in the paper describing these initial models. 

## Results and Observations: 

### Results

- 10 Inner AE iteration and 200 Outer L-S Splitting:
    
    - Lambda value of 100.0:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_100/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_100/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_100/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_100/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_100/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

    - Lambda value of 10.0:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_10/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_10/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_10/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_10/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_10/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

    - Lambda value of 5.0:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_5/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>


        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_5/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_5/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_5/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_5/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>
    
    - Lambda value of 1.0:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_1/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>


        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_1/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_1/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_1/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_1/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>
    
    - Lambda value of 0.5:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.5/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>
    
    - Lambda value of 0.1:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.1/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

    - Lambda value of 0.05:

        - Plots showing True images, L and S splits from the 180-th training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/Examples_180_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/Examples_180_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/Examples_180_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>


        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Short_inner_iterations/lambda_0.05/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>




<br>
<br>

- Longer AE training and less L-S Splitting:

    - Lambda value of 1.0:
        - Plots showing True images, L and S splits from the final training iteration:

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_1/Examples_9_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/Examples_9_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/Examples_9_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_1/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_1/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_1/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_1/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

    - Lambda value of 0.5:
        - Plots showing True images, L and S splits from the final training iteration:

        <div style="text-align:center">
         <img src="L21_training_files/Long_inner_iterations/lambda_0.5/Examples_9_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/Examples_9_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/Examples_9_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.5/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

        
         
    - Lambda value of 0.1:
        - Plots showing True images, L and S splits from the final training iteration:

        <div style="text-align:center">
         <img src="L21_training_files/Long_inner_iterations/lambda_0.1/Examples_9_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/Examples_9_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/Examples_9_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.1/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>


    - Lambda value of 0.05:
        - Plots showing True images, L and S splits from the final training iteration:

        <div style="text-align:center">
         <img src="L21_training_files/Long_inner_iterations/lambda_0.05/Examples_9_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/Examples_9_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/Examples_9_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

       - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.05/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

    - Lambda value of 0.01:
        - Plots showing True images, L and S splits from the final training iteration:

        <div style="text-align:center">
         <img src="L21_training_files/Long_inner_iterations/lambda_0.01/Examples_9_iteration/true_images.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/Examples_9_iteration/L_matrices.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/Examples_9_iteration/S_matrices.png" alt="image" width="300"/>
        </div>
        <br>

       - 5 Plots showing reconstruction from training images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/_training_example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/_training_example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/_training_example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/_training_example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/_training_example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - 5 Plots showing reconstruction from test images: 

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/example_0.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/example_1.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/example_2.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/example_3.png" alt="image" width="300"/>
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/example_4.png" alt="image" width="300"/>
        </div>
        <br>

        - Full Convergence plot (May want to rescale so that it can be better visualised log):

        <br>
        <div style="text-align:center">
        <img src="L21_training_files/Long_inner_iterations/lambda_0.01/Full_Convergence_plot.png" alt="image" width="500"/>
        </div>
        <br>

### Observations:

    - Short Iteration Training: 
        - Lambda 100: 
        - Lambda 10: 
        - Lambda 5:
        - Lambda 1: 
        - Lambda 0.5:
        - Lambda 0.1:
        - Lambda 0.05:


    - Long Iteration Training:
        - Lambda 1: 
        - Lambda 0.5:
        - Lambda 0.1:
        - Lambda 0.05:
        - Lambda 0.01:




## Comments

- Dataset is still not the best:
    - Using 28 by 28 images
    - Using mean average of the 3 colour bands --> Paper suggests only using the Green Band:  https://www.mdpi.com/2072-4292/13/23/4948
    - Some images can have more than half of the image as black --> This is because of the clipping script which returns images that only contain the pixels which are reefs from the reefs shape file.
        - Could work around this by either removing these images from training set, or at least reduce their number so they can be detected as anomalies and not the most normal of the images. 

