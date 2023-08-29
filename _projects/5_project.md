---
layout: page
title: HuBMAP Kidney Blood Vessel Segmentation Challenge on Kaggle 
description: Participation in Kaggle Competition to Sharpen Image Segmentation Technique 
moredescription: <i> Personal Side Project (2023) </i>
img: assets/img/5_project/kaggle-thumbnail.png
importance: 5
category: fun
---

---

### ***Project Motivation & Background:***

Since this is a [Kaggle competition (link)](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/overview), below is the copy-and-pasted "motivation" or context of designing the competition for *annotating blood vessels, or microvasculature from healthy human 
H&E kidney slides*:

<blockquote>
"The proper functioning of your body's organs and tissues depends on the interaction, spatial organization, and specialization of your cells—all 37 trillion of them. With so many cells, determining their functions and relationships is a monumental undertaking.
Current efforts to map cells involve the Vasculature Common Coordinate Framework (VCCF), which uses the blood vasculature in the human body as the primary navigation system. The VCCF crosses all scale levels--from the whole body to the single cell level--and provides a unique way to identify cellular locations using capillary structures as an address. 
However, the gaps in what researchers know about microvasculature lead to gaps in the VCCF. If we could automatically segment microvasculature arrangements, researchers could use the real-world tissue data to begin to fill in those gaps and map out the vasculature.
Competition host Human BioMolecular Atlas Program (HuBMAP) hopes to develop an open and global platform to map healthy cells in the human body. Using the latest molecular and cellular biology technologies, HuBMAP researchers are studying the connections that cells have with each other throughout the body.
There are still many unknowns regarding microvasculature, but your Machine Learning insights could enable researchers to use the available tissue data to augment their understanding of how these small vessels are arranged throughout the body. Ultimately, you'll be helping to pave the way towards building a 
Vascular Common Coordinate Framework (VCCF) and a Human Reference Atlas (HRA), which will identify how the relationships between cells can affect our health.
</blockquote>

If interested, the raw data and the information about the data can also be found [here](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data).
Writing all about the background of this competition regarding motivation, data, evaluation metric, etc. would be too repetitive since it's all in the website, so I'll skip this part and focus on the 
methods."

---

### ***Methods:***

The main steps that I took to tackle this problem is: 
<br>
* **Preprocessing the raw data**

- Preprocessing .json annotation files and data/metadata in .csvs. Dividing different versions of datasets depending on which approach of training to take
  (ex. training a single object detection model for blood vessel, or train a segmentation model with two classes (blood vessel and glomerulus))
- Performing exploratory data analysis (EDA) to really dive into the metadata (ex. how many WSI images there are, how many tiles in each WSI, tile size, etc.)
- Then, using the .json annotation files to create binary masks (image) or a .txt file with (x,y) coordinate contours. This is paramount so that it can be utilized as labels for supervised training later.
- After training image and label is properly paired, using either each WSI number or class label as a stratification method to create cross-validation folds of the tiled images. Due to class
imbalance of training data, it can usually be a good idea to stratify based on class and create a 5-fold cross validation pipeline. 
<br>

* **Training and validating different models**
    1. *Image augmentation/normalization:*
      - Experimenting with different degrees and techniques of augmentation is crucial, and also the same with image normalization. Some common image augmentations such as horizontal/vertical flips,
      resizes, color jittering, noising/blurring, etc are explored. 
      - [RandStainNA](https://arxiv.org/abs/2206.12694), which is a pipeline that does image augmentation and normalization at the same time by utilizing other color spaces such as LAB and HSV is also
      utilized.

    2. *Choosing a model architecture to train:*
      - UNet (semantic segmentation), YOLOv8 (object detection/instance segmentation), and Mask2Former (semantic segmentation) have been chosen. I've decided to choose a 
      wide variety of models (2 CNN-based and 1 ViT-based) to explore different models. 
      - Then, if a pretrained model is available, explore if pretrained models will be useful (domain-specific or non-domain-specific). A pretrained domain-specific (histopathology) model hub developed [here](https://github.com/lunit-io/benchmark-ssl-pathology) was tested on model architectures if compatible.
    3. *Choosing a loss function and validation metric:*
      - Depending on the model, different loss functions are explored. 
      - For example, UNet can be explored with different functions like Dice/Jaccard(IOU)/BCE losses, while YOLOv8 will be a combination of Focal, Bbox, and IOU loss. 
      - Because the evaluation metric is known (average precision (AP) @0.6 IOU), returning the metric during validation is important.
     
* **Running inference locally and tuning (if possible)**

- Testing and validation metric need to be tested to see if they are closely correlated to see if training process is accurate and robust (no under/over-fitting).
- Once a decent model is trained, if possible, tuning the hyperparameters may be necessary to further enhance the model performance.

*Note that detailed methods can be seen in the preprocessing/training/infer/tune [code on github](https://github.com/chokevin8/Kaggle-hubmap)*

---

### ***Results & Discussion:***

While the goal of this project wasn't focused on winning the competition (lack of time and experience/skills for me to do that yet), but there were
still a list of things that I tried doing that worked and didn't work in increasing the performance of the model. 

- *Things that worked:*
    - Dilating the dataset 2 annotations, since the annotations are inconsistent (some including the endothelial cells around blood vessels and some not) to make it consistent. 
    - Stratification of WSI tiles based on dataset number and making sure each CV fold has dataset 1 for training and validation since testing dataset is solely based on dataset 1. 
    - Using Albumentation's normalization and augmentation pipeline boosted performance compared to Torchvision or RandstainNA.
    - YOLOv8 instance segmentation model worked pretty well, with some edits to the loss function weightings. 
    - Utilizing Test-time augmentation (TTA) on YOLOv8 and UNet boosted performance compared to not utilizing it. 

- *Things that didn't seem to work:*
    - Stratification of WSI tiles based on its classes didn't work. Probably because the testing dataset was solely on dataset 1, so the validation needs to be against mostly dataset 1.
    - Using RandStainNA as an augmentation + normalization tool together did not boost performance.
    - Surprisingly, using pretrained model on histopathology dataset (Resnet50 backbone) didn't boost performance at all compared to random initialization.
    - Using UNet as a sole semantic segmentation model, probably because the metric is an object-detection based metric, and UNet doesn't naturally create Bboxes and a confidence value. 
    - Using Mask2Former as a semantic segmentation model, but probably because I may have used the wrong pretrained model (imagenet pretrained).
    - Self-supervised learning, this is something I couldn't do due to only having a single GPU.


---

### ***Personal Comments:***

### Q: Why did I choose this project? ###
I decided to work on this project on the side whenever I had time to work on it, since I thought that I would need more experience in working on an entire project by myself. I thought that performing the entire DL research pipeline of
EDA -> Data Preprocessing -> Training/Validating -> Testing/Tuning -> Post-processing by myself would be massively helpful to learn more about the technical skills (Python, PyTorch, etc) required for a DL researcher/scientist.

### Q: What did I do outside of this project? ###
Background research (reading literature/online documentations) on:
- Techniques for H&E image normalization and augmentation (ex. Reinhard, Vahadane, StainTools, RandStainNA, Albumentations, Torchvisions, etc.)
- List of self-supervised learning techniques (ex. Barlow Twins, MoCoV2, etc.)
- Different model architectures (ex. UNet, Mask R-CNN, DeepLabv3+, YOLO, etc.)
- Different loss functions and their differences (ex. Dice, Jaccard (IOU), Focal, BCE, Tversky, Bbox, etc.)
- Tuning techniques (learning how to use Ray Train and Tune)

### Q: What impact did this project have on me? ###

In academic research as a master's student, you often work together on a project or work independently but for a part of a pipeline. As this was my first time going through a full DL research pipeline by myself,
I discovered that so much time and effort could be saved if you actually *planned out your experiments*. This may sound cliché, but I think new DL researchers like me suffer with this a lot. For example, in retrospect, for some of my training runs 
I had no idea why I was performing them, as I treated the DL model as a complete "black box" (while this is partially true, it's not 100% a black box) even when I already set the random seeds for all modules for full reproducibility. Some other mistakes I made 
was training a huge model with the entire dataset before performing a smaller-scale experiment as a proof of concept. This led me to waste unnecessary time and efforts. Starting from a smaller-scale model and data with small number of epochs as a proof-of-concept is crucial, 
as often times switching to a bigger model and more dataset is more about tuning the hyperparameters (which can be done after successful training) like batch size, learning rate, number of epochs, etc. 

---