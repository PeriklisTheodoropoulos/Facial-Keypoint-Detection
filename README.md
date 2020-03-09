# Facial-Keypoint-Detection
## Project Overview
This repository contains project files for Computer Vision Nanodegree program at Udacity. 
It combines knowledge of Computer Vision Techniques and Deep learning Models to build a 
facial keypoint detection system that takes in any image with faces, and predicts the location of keypoints on each face. 
Facial keypoints include points around the eyes, nose, 
and mouth on a face.

## Project Structure
The project will be broken up into a few main parts in four Python notebooks:
* models.py
* Notebook 1 : Loading and Visualizing the Facial Keypoint Data
* Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints
* Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN
## Local Environment Instructions
1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
$ git clone https://github.com/nalbert9/Facial-Keypoint-Detection.git

2. Create (and activate) a new Anaconda environment (Python 3.6). Download via Anaconda
* Linux or Mac 
conda create -n cv-nd python=3.6
source activate cv-nd
* Windows:
conda create --name cv-nd python=3.6
activate cv-nd
3. Install PyTorch and torchvision; this should install the latest version of PyTorch;
* conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

## Licence 
This project is licensed under the terms of the License: MIT






