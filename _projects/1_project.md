---
layout: page
title: Virtual Stain Conversion of IHC/Unstained to H&E Images (IHC2HE/US2HE)
description: Design of Generative Models to Virtually Stain/Convert IHC/Unstained (Bright-field) Images to H&E Images 
moredescription: <i>Master's Thesis Project (2023 ~ Present)</i>
img: assets/img/1_project/ihc2he_thumbnail.png
importance: 1
category: research
---

---

### **Project Motivation & Background:**
<br>

**Let's play a simple guessing game:** Look at the two pictures below. At the most left is the ground truth IHC image to be converted. The next two images on the right
are the ground truth H&E version of the ground truth IHC image and the virtually stained H&E version of the ground truth IHC image. Can you guess which is "real" and
which is "fake"?

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/ihc2_he_intro.png" title="IHC2HE_intro" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The correct answer is that the images in the right (column) are the "fake" ones, or the ones sampled from the generative model. 
<br>
<br>
Let's play another game, this time for a ground truth unstained, bright-field image to be converted. The next two images on the right are the ground truth H&E version of the ground truth unstained image and the 
virtually stained H&E version of the ground truth unstained image. Can you guess which is "real" and which is "fake"?

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/us2_he_intro.png" title="US2HE_intro" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The correct answer again is that the images in the right (column) are the "fake" ones, or the ones sampled from the generative model. 
If you look closely, you could tell that the images on the right have slight artifacts and may be a bit "awkward" in general, this is because
this is still a work in progress. With correct training, I hope to make them indistinguishable! The above two examples tell us the two different kinds of stain conversions 
that exist: 1) label-free-to-stain conversions and 2) stain-to-stain conversions. But someone may ask, **why do we want to do this? Why virtually stain or convert stains?**

First of all, commonly used histopathological stains such as H&E (Hematoxylin & Eosin), IHC (Immunohistochemistry), IF (Immunofluorescence), and others are all used to
examine different types of body tissues for various research of diseases (pathology). For example, H&E and specific types of IHC stains are commonly used in cancer pathology research, as each stain 
visualizes the tissue and cellular structures of the tissue in its own way, and therefore allow new digital biomarkers to be developed by utilizing the stained images.
However, histopathological stains are also commonly used for disease diagnostics, as for example, IHC staining for the specific tumor/cancer biomarker of HER2 can reveal different treatment plans for
HER2 positive or negative patients. 

While histopathological stains are widely used in disease research and diagnostics, the procedures of staining carried out in pathology labs are often time-consuming, expensive,
and also can be extremely tricky to perform. The procedure is time-consuming due to the complexity of the staining process- tissue extraction, fixation, embedding, mounting, and staining 
are mostly all laborious tasks. Furthermore, specific types of stains, like IHC stains, can be very expensive, especially if they are used for disease research instead of
disease diagnostics as obtaining large amounts of IHC-stained tissue slides can easily cost a fortune. Furthermore, the staining process itself also causes mechanical and chemical damages to the
tissues, and if poorly performed, can damage the tissue. Most importantly, applying a stain on a tissue slide prevents other types of staining on the same slide, preventing further biological analysis on the
same slide. 

While virtual staining cannot expedite the time-consuming process of tissue extraction, fixation, embedding, and mounting, it can resolve most of the other issues mentioned above. Virtual stain conversions from label-free
(unstained) slides to stained slides can not only expedite the staining process, it can be more accurate due to no mechanical/chemical damage on the tissue and be much cheaper as the 
stain does not need to be purchased. Furthermore, stain-to-stain conversions have a huge benefit as it allows staining on the same tissue slide, enabling further biological analysis on the same slide. 
**Therefore, the main motivation of the unstained-to-HE virtual stain conversion of skin tissue images is to expedite and refine the staining process, while the main motivation of the
IHC-to-HE virtual stain conversion of pancreatic tissue images is to allow multiple stain analysis on the same tissue slide.**

---

### **Methods:**

Below is the scheme for training and evaluating the trained model for the IHC-to-HE virtual stain conversion project. The same pipeline is applied to the unstained-to-HE virtual stain conversion project as well.

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/IHC2HE_training.png" title="IHC2HE_training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/IHC2HE_evaluation.png" title="IHC2HE_eval" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

As seen above, during training,


---

### **Results:**

The most recent results is 

***Note that more technical details/explanations and further results are omitted on purpose as I focus on motivation/personal comments in introducing the project. More technical details
will be shown in the technical powerpoint presentation (PPT).***

---

### **Personal Comments:**

### Q: Why did I choose this project? ###


### Q: What did I do outside of this project? ###

### Q: What impact did this project have on me? ###


---

*Image credits to:*