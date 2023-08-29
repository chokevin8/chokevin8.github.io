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
The proper functioning of your body's organs and tissues depends on the interaction, spatial organization, and specialization of your cells—all 37 trillion of them. With so many cells, determining their functions and relationships is a monumental undertaking.
Current efforts to map cells involve the Vasculature Common Coordinate Framework (VCCF), which uses the blood vasculature in the human body as the primary navigation system. The VCCF crosses all scale levels--from the whole body to the single cell level--and provides a unique way to identify cellular locations using capillary structures as an address. 
However, the gaps in what researchers know about microvasculature lead to gaps in the VCCF. If we could automatically segment microvasculature arrangements, researchers could use the real-world tissue data to begin to fill in those gaps and map out the vasculature.
Competition host Human BioMolecular Atlas Program (HuBMAP) hopes to develop an open and global platform to map healthy cells in the human body. Using the latest molecular and cellular biology technologies, HuBMAP researchers are studying the connections that cells have with each other throughout the body.
There are still many unknowns regarding microvasculature, but your Machine Learning insights could enable researchers to use the available tissue data to augment their understanding of how these small vessels are arranged throughout the body. Ultimately, you'll be helping to pave the way towards building a 
Vascular Common Coordinate Framework (VCCF) and a Human Reference Atlas (HRA), which will identify how the relationships between cells can affect our health.
</blockquote>

If interested, the raw data and the information about the data can also be found [here](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data).
Writing all about the background of this competition regarding motivation, data, evaluation metric, etc. would be too repetitive since it's all in the website, so I'll skip this part and focus on the 
methods.

---

### ***Methods:***

The main steps that I took to tackle this problem is: 
1. Preprocessing the raw data
- Preprocessing .json annotation files and data/metadata in .csvs. Dividing different versions of datasets depending on which approach of training to take
  (ex. training a single object detection model for blood vessel, or train a segmentation model with two classes (blood vessel and glomerulus))
- Performing exploratory data analysis (EDA) to really dive into the metadata (ex. how many WSI images there are, how many tiles in each WSI, tile size, etc.)
- Then, using the .json annotation files to create binary masks (image) or a .txt file with (x,y) coordinate contours. This is paramount so that it can be utilized as labels for supervised training later.
- After training image and label is properly paired, using either each WSI number or class label as a stratification method to create cross-validation folds of the tiled images. Due to class
imbalance of training data, it can usually be a good idea to stratify based on class and create a 5-fold cross validation pipeline. 

2. Training different models
-Image augmentation/normalization:
    - Experimenting with different degrees and techniques of augmentation is crucial, and also the same with image normalization. Some common image augmentation 
   RandStainNA
-d
-d
3. Running inference locally and tuning (if possible)
- CV/LB correlation 
- 


*Note that detailed methods can be seen in the preprocessing/training/infer/tune [code on github](https://github.com/chokevin8/Kaggle-hubmap)

---

### ***Results & Discussion:***
Things that worked:


Things that didn't seem to work:

---

### ***Personal Comments:***

### Q: Why did I choose this project? ###

### Q: What did I do outside of this project? ###

### Q: What impact did this project have on me? ###

---