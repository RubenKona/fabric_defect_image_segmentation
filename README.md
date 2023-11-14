# loopr_image_segmentation
This repository holds the code to train a UNet for defect segmentation in a data provided by: https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnexuswho%2Faitex-fabric-image-database

# Setting up your environment
The code in this repository use the following repositories. Please ensure your python environment is installed with the following repos (and versions):
- skimage version == 0.19.3  -- Note that this code will not work with skimage version 0.21.0 (e.g. in current (11/13/23) Kaggle environment)
- matplotlib version >= 3.7.1
- numpy version >= 1.23.5
- scipy version >= 1.11.3
- h5py version >= 3.9.0
- tensorflow version >= 2.14.0
- keras version >= 2.14.0
- pandas version >= 1.5.3

On a system with Python 3 installed, an environment with these packages can be installed with the following bash commands:
~~~~
# Using venv (Python 3.3 and newer)
python3 -m venv loopr_image_segmentation

# Activate the virtual environment
source loopr_image_segmentation/bin/activate  # On Windows, use `loopr_image_segmentation\Scripts\activate`

# Now, install the specific versions of the required packages
pip install scikit-image==0.19.3 matplotlib==3.7.1 numpy==1.23.5 scipy==1.11.3 h5py==3.9.0 tensorflow==2.14.0 pandas==1.5.3
~~~~

# Clone repo to local environment

To get started with the code in this repository, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/mushroom-matthew/loopr_image_segmentation.git
    ```
    *Please note that the default repo name is the same as was used above for the virtual environment. This code will not work if you run it in the parent directory that contains your environment directory.*
    
2. Change into the repository directory:

    ```bash
    cd loopr_image_segmentation
    ```

3. Now, you can explore the code and start working with the provided Jupyter notebooks or Python scripts.

If you encounter any issues or have questions, feel free to reach out or open an issue on GitHub.

# Running inference on an image (or directory of images)

The current supported functionality comes in the form of an inference pipeline. To run that pipeline and generate a printed report, please run the following line of code from outside the `loopr_image_segmentation` module directory.

~~~~bash
python3 -m loopr_image_segmentation.scripts.defect_segmentation --model {/absolute/path/to/}loopr_image_segementation/models/pretrained_model.h5 --image {/absolute/path/to/image/or/directory/of/images/such/as/}loopr_image_segmentation/data/sample_data/
~~~~

This script can take additional arguments which include save options for masks, logits, and csv reports for found defects.

*Please note that only one model is currently supported. This model is known to be susceptible to false positives (in particular). Reducing the `--grain` input argument can help for those false positives related to edge effects of patch operations. On the other hand, another source of false positives seems related to illumination variations across samples. A v2 model is being trained and will be added upon completion.*
