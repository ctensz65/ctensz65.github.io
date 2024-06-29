---
layout: article
title: "Obstacle Avoidance using Mask RCNN Image Segmentation"
key: 20231112
categories: [robotics]
aside:
  toc: true
tags: 
  - deep-learning 
  - cnn
  - image-segmentation
---

## Image Segmentation
Digital images serve as fundamental building blocks in various fields, including elements like pattern, shape, texture, and a rich tapestry of colors. At their core, images are composed of pixels – tiny matrix elements that determine the color intensity at each point. This bunch of pixels forms images that are discernible to the human eye. The realm of image processing involves structure tasks such as the acquisition, analysis, and manipulation of digital images. Advances in image processing techniques have significantly expanded their applications, notably in fields like non-invasive precision measurements and automated robot navigation.

![Type of image segmentation](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_image-segmentation%202.jpeg)

In the context of computer vision, segmentation is a critical preliminary step preceding feature extraction and classification. Segmentation involves dividing an image into smaller segments or blocks, each representing a specific region defined by criteria such as edge detection or regional characteristics. Common segmentation methods include edge-based and region-based approaches. The latter focuses on establishing regions based on pixel similarity. Within this domain, the Convolutional Neural Network (CNN) emerges as a powerful tool, reducing reliance on intensive computing. A notable application within CNNs is the Region-based method, where an image is segmented based on proposed regions. These regions are then labeled with specific classes and bounded by boxes.

## Mask Region CNN

The development of Region-Based Convolutional Neural Networks (R-CNNs) took a significant leap forward in 2018 with the introduction of Mask R-CNN. This model is primarily built on two components: image segmentation and instance segmentation. Image segmentation involves dividing a digital image into multiple segments, where each segment is a collection of pixels that collectively represent an object or feature. Instance segmentation goes a step further by not only identifying specific objects within an image but also providing a precise outline (mask) of each object.

Mask R-CNN is an evolutionary step beyond its predecessor, Faster R-CNN. It enhances the basic framework of Faster R-CNN, making Mask R-CNN an advanced and more refined version of the earlier model.

**The Two-Stage Architecture of Mask R-CNN**

![Mask RCNN](https://www.researchgate.net/publication/332998337/figure/fig2/AS:757014435225602@1557497714813/Mask-Region-Convolutional-Neural-Network-R-CNN-framework.ppm)

Mask R-CNN operates in two stages:

- Stage One - Regional Proposal Network (RPN): This stage involves feature extraction from the input image. The RPN also generates a set of region proposals, which cover potential areas of interest across the entire image.

- Stage Two - Bounding Box Regression, Class Prediction, and Binary Mask Prediction: This stage refines the region proposals. Unlike its predecessor models, Mask R-CNN replaces the Region of Interest (RoI) Pooling method with RoIAlign. RoIAlign offers a more accurate feature extraction, especially in terms of aligning the extracted features with the input. This stage culminates in the prediction of bounding boxes, object classification, and generating a binary mask for each instance.

The key innovation in Mask R-CNN lies in its ability to accurately segment objects at the instance level, making it a powerful tool in the field of computer vision.

## Detectron2

[Detectron2](https://github.com/facebookresearch/detectron2), developed by Facebook's Artificial Intelligence research team, stands out as a prominent framework supporting the training and visualization of Mask R-CNN models. This framework is integral to implementing object detection algorithms, including Mask R-CNN, offering several advantages:

1. Open-Source Accessibility: Detectron2 is fully open-source, providing unrestricted access to its comprehensive library.

2. Efficient Training Pipelines: The framework is optimized for GPU utilization, ensuring speed and scalability in training processes. This capability is particularly beneficial in handling the computationally intensive tasks associated with Mask R-CNN.

3. Support for Instance Segmentation: A crucial feature of Mask R-CNN is instance segmentation, which Detectron2 efficiently supports. This functionality is vital in various practical applications, such as in laboratory settings.

4. Integration with PyTorch: Detectron2 is built on the PyTorch machine learning framework. PyTorch is known for its dynamic computing capabilities, enabling automatic differentiation and facilitating the implementation of deep neural networks.

5. Compatibility and Efficiency: The framework efficiently supports low-level tensor operations on both GPU and CPU, enhancing its compatibility and performance.

In the context of robotics, Detectron2 plays a pivotal role. For example, in the development of obstacle avoidance systems in robots, Detectron2 is the primary framework used for training the Mask R-CNN model. The trained model is then employed in the robot's object detection system, significantly enhancing its ability to navigate and avoid obstacles.

## Design

### System
In order to facilitate detailed design efforts, a simple visualization of the outline of the working functions of the object detection system is carried out as follows.

<div style="width:70%; margin:0 auto; text-align: center;">
    <img src="https://ctensz65.github.io/assets/images/desain1.png" style="max-width: 100%; height: auto;">
    <p style="text-align: center; margin-top: 5px;">General overview workflow</p>
</div>

Above diagram provides a streamlined visualization of the workflow for designing an object detection system powered by Mask R-CNN. The process begins with training data (denoted as "Training Objek"), which consists of images that the system will learn to identify and categorize. The training proceeds with the aid of Detectron2 and PyTorch, which facilitate the model training process through advanced algorithms and GPU optimization.

Upon successful training, the Mask R-CNN object detection system is deployed. This system is capable of recognizing and interpreting new data inputs, as represented by the feedback loop to the "New Data" stage. The final application, as shown, is an autonomous robot capable of avoiding obstacles, utilizing the trained Mask R-CNN for real-time detection and navigation.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/desain2.png" style="max-width: 100%; height: auto;"></div>

### Hardware

The chassis is constructed from aluminum profiles, each side measuring 39 cm in length, extending from the front to the back of the robot. An additional two aluminum pieces serve as the central frame, connecting the longer sides. The robot's width is set to be 30 cm, which does not include the tires. The main body of the robot is crafted from acrylic material, with the base floor being 5 mm thick and the second floor 3 mm thick.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/desain3.jpg" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

The 5 mm thick base floor is designed to serve as a compartment or storage area for a laptop, ensuring sturdiness and strength. The second floor, which is thinner, is designated for mounting electronic modules, boards, and their associated components. The vertical clearance between the base floor and the second floor is 7 cm, providing ample space for the electronics.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/desain4.jpg" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

The design adheres to the foundational concept of the robot, which is to be as lightweight as possible to enable more agile and effective maneuvering. In the preliminary design, the electronic board is intended to be housed in a dedicated compartment at the top of the robot, ensuring easy access and protection. The bumper is manufactured through 3D printing, allowing for a custom shape that aligns with the original design vision.

Support for the second floor is provided by two aluminum profiles, one attached to each side of the chassis. This structure ensures stability and strength where electronic components are placed.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/desain5.jpg" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

For mobility, the robot employs four RC car, selected for their lightweight and durability. A motor couplinganism is used to securely attach each tire to the shaft of a DC motor. These motors are strategically positioned at the intersections of the side aluminum profiles and the central frame's aluminum profiles, optimizing both space and weight distribution.

## Implementation

### Training using Google Colab

I utilize Google Colab for training datasets in the development of an image segmentation model using Detectron2. While Colab is the primary cloud-based option for this purpose, training can also be conducted locally on a personal device equipped with a suitable GPU that meets the requirements of Detectron2.

The Colab notebook is designed for user-friendly interaction, organizing the workflow into distinct columns. These columns display essential information such as the dataset's zip file name, the storage path, and the count of labeled objects within the dataset. A dedicated code cell is included for visualizing datasets that have been successfully registered. Furthermore, the labels or class names of the identified objects in the dataset can be defined using specific function codes.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/trainingdata.png" style="max-width: 100%; height: auto;"></div>

The next phase involves the actual training of the model with Detectron2. Users can input details into various columns, such as the model name and training parameters. Default parameters are provided to ensure that even users new to the platform can achieve satisfactory training outcomes without the need for extensive custom settings. The Python code utilized for this process leverages the Detectron2 library, which is integrated into the Colab project.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/tensorboard_result.png" style="max-width: 100%; height: auto;"></div>

Post-training, the Mask R-CNN object detection model's performance can be assessed via TensorBoard. The Colab Notebook contains a specialized code cell for initiating TensorBoard, which facilitates the evaluation of the Mask R-CNN model. TensorBoard provides a suite of scalar graphs that offer insights into the model's accuracy, the behavior of the loss function, and other pertinent metrics.

### Robot Fabrication

The fabrication of the robot encompasses several components, including the chassis, body sections, supports, tires, and rear bumper. The chassis is constructed using 2040 aluminum profiles, which are cut to specific dimensions determined during the design phase. The robot's body is comprised of two levels: the base section and the second section, both fashioned from acrylic material. The base section is 5 mm thick, providing a sturdy platform for a laptop, which serves as the primary processing unit. In contrast, the second section is 3 mm thick and is secured to the chassis with four supports mounted on the aluminum profile rails.

<div style="width:40%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/rover5.JPG" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

The supports for the rear and the bumper are produced with 3D cutting techniques. The robot is equipped with 14-inch rubber tires for mobility. Additionally, the Arduino Mega 2560 is situated on the lower tier of the electronics board, providing direct support to the second floor's body section.

### Board Electronic

The electronic board is designed to integrate and connect the microcontroller with various components, including the L298N Motor Driver, Encoder Motor, and 12 Volt DC Motor. Utilizing Printed Circuit Board (PCB) simplifies the setup by minimizing the need for excessive cabling, which is generally undesirable. The PCB features several terminal blocks dedicated to connections for switches, battery voltage sources, and the motor voltage sources required for the 12 Volt DC Motors.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/electronic.png" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/rover2.jpg" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

## Result

### Trained Model Mask R-CNN

The Python code executed in the Colab Notebook ran flawlessly, free from any errors. The training process typically spans 10 to 15 minutes, relying on the default settings provided within the Notebook. The training culminates in the generation of two essential files for the web-based monitoring application: 'modelfinal.pth' and 'config.yaml', both of which were successfully produced and downloaded.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/trainingprocess.jpg" style="max-width: 100%; height: auto;"></div>

The evaluation phase, as outlined in the Practical Notebook, effectively displays the model's performance metrics. Accuracy assessment and total loss value were analyzed using TensorBoard. In addition, the Detectron2 framework facilitated the calculation of quality metrics, producing **Mean Average Precision** (mAP) values. The model's evaluation further demonstrated a proficient comparison mechanism between Ground Truth data and the Output Predictions.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/resultraining.jpg" style="max-width: 100%; height: auto; border: 1px solid #000000;"></div>

### Detection Accuracy

The test results confirm that the dataset validation feature of the web-based monitoring app aligns with the initial objectives and design. For the purpose of demonstrating prediction visualization, three sample images have been selected to showcase the capabilities of the trained Mask R-CNN model. Additionally, the website provides detailed information about the names of objects identified in the dataset. This feature is integral for the validation of the system or monitoring app, ensuring the smooth operation of the robot.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/validate_dataset.png" style="max-width: 100%; height: auto;"></div>

The image presentebelow offers a clear visualization of the predictions made by the trained Mask R-CNN model. Additionally, the web dashboard provides details about the names of objects recognized in the dataset. This information is crucial for validating the monitoring application or system to ensure the robot's operations are executed effectively. The objects within the image have been accurately identified, with proper segmentation and labeling conducted.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/feed.png" style="max-width: 100%; height: auto;"></div>

The prediction data encompasses parameters and attributes precisely extracted from the identified objects. This data is dynamically displayed on the website's widget in real-time, reflecting the images captured by the camera. Moreover, the prediction data can be documented in an Excel file, which users have the option to download for further analysis.

<div style="width:70%; margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/camerafeed.png" style="max-width: 100%; height: auto;"></div>

### Maneuverability based on Object Detection

In its default state (without detecting a trained object), the robot car is programmed to move forward upon activation of the power button. It maintains this forward motion until it encounters a trained object. Upon detection, the robot then makes a decision to either turn right or left, stop, or possibly move backward, depending on the specific object it encounters.

1. Avoidance Manuver
   
   The obstacle avoidance robot adeptly navigates around the cone, opting to turn left or right based on the calculations of a safe, effective distance.

    <div style="margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/demoforward.gif" style="max-width: 150%; height: auto;"><img src="https://ctensz65.github.io/assets/images/demo.gif" style="max-width: 150%; height: auto;"></div>

2. Stop
   
   During testing, the obstacle avoidance robot successfully came to a halt as originally planned upon detecting a baseball
    
    <div style="margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/demostop.gif" style="max-width: 150%; height: auto;"></div>

3. Move Backward
   
   <div style="margin:0 auto; text-align: center;"><img src="https://ctensz65.github.io/assets/images/demobackward.gif" style="max-width: 150%; height: auto;"></div>

## Conclusion

In conclusion, the collaborative effort of using Google Colab and Detectron2 for training the Mask R-CNN model has proven successful. With the ability to run error-free on Colab Notebook and the efficient training time of 10 to 15 minutes, the model has been effectively trained and evaluated. The system's validation through web-based monitoring, real-time prediction data display, and the successful documentation of this data underscore the robustness of the setup. With functionalities like forward movement in the absence of trained objects and adaptive responses (such as turning, stopping, or reversing) upon their detection, the robot effectively showcases the practical application of advanced machine learning models in real-world scenarios. 

Last words, Thanks fellas for reading.

- [Colab Notebook](https://colab.research.google.com/drive/16vrmhOQQQ57TF1HTYZ7Hvblm2xW0IKE9?usp=drive_link)
- [Trained Model](https://drive.google.com/file/d/1Ie-AnKniLQ15QO5NyLCCoqDov6pb0uBK/view?usp=drive_link)
- [Github](https://github.com/ctensz65/Obstracle-Avoidance-MaskRCNN)


<!--more-->

---
