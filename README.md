# Pedestrian Detection Using Mask RCNN for Detection and Segmentation

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

1. [person.py](https://github.com/bonie09/pedestrian-detection/blob/master/person.py) - This python file is for the different pre-processing steps to prepare the training data.
2. [trained.py](https://github.com/bonie09/pedestrian-detection/blob/master/trained.py) - This file goes into depth of steps to perform detection and segmentation.
3. [setup.py](https://github.com/bonie09/pedestrian-detection/blob/master/setup.py) - This file handles all the packages for the projects.
4. [pedestrian.py](https://github.com/bonie09/pedestrian-detection/blob/master/pedestrian.py) - The file to detect on any random image.
5. [app.py](https://github.com/bonie09/pedestrian-detection/blob/master/app.py) - This file is used to develop simple web application using flask in python and it gets integrated with the model of out project.
6. [evaluate.py](https://github.com/bonie09/pedestrian-detection/blob/master/evaluate.py) - This file is used to calculate mAP for the model.
7. [balloon.py](https://github.com/bonie09/pedestrian-detection/blob/master/balloon.py) - [The blog which covers from annotating images to training to using the results in a sample application.](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)

# Training and Testing commands

    # Train a new model starting from pre-trained COCO weights
    python pedestrian.py train --dataset=/path/to/pedestrian/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python pedestrian.py train --dataset=/path/to/pedestrian/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python pedestrian.py train --dataset=/path/to/pedestrian/dataset --weights=imagenet

    # Apply color splash to an image
    python pedestrian.py splash --weights=/path/to/pedestrian/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python pedestrian.py splash --weights=last --video=<URL or path to file>


Input:

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/1.PNG)   ![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/3.PNG)

Output:

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/2.PNG)  ![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/4.PNG)

# Website Screenshots:

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/9.jpeg)

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/8.jpeg)

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/7.jpeg)

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/6.jpeg)

After the segmentation the output will get saved in a images/masked folder.

# Evaluation:

![](https://github.com/bonie09/pedestrian-detection/blob/master/assests/5.PNG)

# References 

https://github.com/matterport/Mask_RCNN

https://github.com/krishnaik06/Deployment-Deep-Learning-Model

https://gist.github.com/PhanDuc/abf8bf5a8eed78cc03e4df9b1f1a276c
