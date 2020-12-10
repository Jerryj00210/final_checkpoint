# final_checkpoint

### Introduction
* This project implements a Grad-CAM algorithm and use it on the COCO dataset. The model that we use is a pre-trained model called Resnet50, which is pretrained with ImageNet. The model will take COCO dataset as input and Grad-CAM will visualize it.

##### Config
* Parameters for specifing the path to load image as input and the path to save image as output.

##### Examples
* Results from the Grad-CAM algorithm.

##### Notebook
* Results from `Examples` and explanation/analysis on the results.

##### src
* gradcam.py file includes grad-cam algorithm and image processing.

##### run.py
* Load the images and generate results from the Grad-CAM algorithm.

##### submission.json
* `submission.json` contains the docker image required for running this project and a github link of this project.

### How to run
1. please specify your own image input path and save image path inside the image_path.json from the Config folder.
2. Run `python run.py --image-path <path_to_image>`. If you already specify the load image path from the json file inside the Config folder, you can simply run `python run.py --image-path <path_to_image>`.
3. If you wish to use cuda, then run `python run.py --image-path <path_to_image> --use-cuda`. If you already specify the load image path from the json file inside the Config folder, you can simply run `python run.py --use-cuda`.
