# Depth Estimation
Depth Estimation from Single Image using Deep Learning


## Project Overview

The goal of this project is to develop a deep learning model that can estimate depth information from a single RGB image. Depth estimation is a fundamental task in computer vision and has numerous applications in areas such as 3D reconstruction, augmented reality, robotics, and autonomous navigation.

## Approach

To tackle this problem, we will explore and implement deep learning architectures specifically designed for the task of depth estimation from single images. The project will involve the following steps:

1. **Dataset Preparation**:
   - We will be using the NYU Depth V2 dataset, which consists of RGB images and their corresponding depth maps captured using a Microsoft Kinect sensor.
   - The dataset will be preprocessed, including resizing, normalization, and splitting into training and testing sets.

2. **Deep Learning Model Architecture**:
   - We will investigate and implement state-of-the-art deep learning architectures for depth estimation, such as Fully Convolutional Residual Networks (FCRN) [1], Multi-Scale Depth Prediction Networks [2] and Depth Prediction Transformers
   - These architectures leverage techniques like encoder-decoder structures, residual connections, and multi-scale feature fusion to effectively estimate depth from single images.

3. **Model Training**:
   - The chosen deep learning model will be implemented using a framework like Tensorflow.
   - Appropriate loss functions, such as scale-invariant loss and gradient loss, will be employed to train the model effectively.
   - Techniques like transfer learning, data augmentation, and hyperparameter tuning will be explored to improve model performance.

4. **Evaluation and Testing**:
   - The trained model's performance will be evaluated on the test set using standard evaluation metrics for depth estimation, such as root mean squared error (RMSE), absolute relative error (ARD), and others.
   - Qualitative analysis will be performed by visualizing and comparing the estimated depth maps with the ground truth depth maps.

5. **Deployment and Demonstration**:
   - A demo application or web interface will be developed to showcase the trained model's capability in estimating depth from single RGB images.
   - Efforts will be made to optimize the model for efficient inference, enabling real-time depth estimation capabilities.

## Refernces

Dataset - https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
[1] https://arxiv.org/abs/1606.00373
[2] https://arxiv.org/abs/1606.00373
