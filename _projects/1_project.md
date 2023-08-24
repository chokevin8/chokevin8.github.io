---
layout: page
title: 2/3D Semantic Segmentation of Skin H&E Tissue Images
description: <i>Segmentation of Skin H&E Tissue Images to Analyze Novel Cellular Biomarkers of Aging</i>
img: assets/img/1_project/h&e_thumbnail.PNG
importance: 1
category: research
---

### **Motivation & Background:**
Let's play a simple guessing game: Look at the pictures below. Can you guess who is older?

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/real_95yr_man.png" title="95 Year Old Man" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/Tom_Brady_44yr_old.png" title="44 Year Old Tom Brady" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    On the left, we have a 95 year old man, and on the right, we have 44 year old Tom Brady. 
</div>

*Pretty obvious, right?* Obviously left man is older and right man (Tom Brady) is younger. 
Now, assume we were able to receive the skin tissue samples of both men, cut them into serial sections, applied H&E stain to them, scanned/digitized them, 
and then looked at one part of the whole slide images (WSI). Two pairs of pictures are shown below, and each man's H&E pictures are on the 
left and the right. ***Can you answer the same question now?***

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/44yr_4x.PNG" title="44yr_4x" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/95yr_4x.PNG" title="95yr_4x" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/44yr_20x.PNG" title="44yr_20x" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1_project/95yr_20x.PNG" title="95yr_20x" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Which tissue belongs to the 95-year old man? The one on the left or the right?
</div>

The answer is: ***44-year old Tom Brady is on the left, and 95-year old man is on the right.*** While these tissues are not actually from the people
pictured above, they actually are tissue samples from patients who are 44-year old and 95-year old. *The point is, compared to just looking at someone's face, 
it's much harder to determine a person's age by just looking at their skin H&E tissue sample.* But someone may ask, ***Why do we want to do this? Why study aging, 
specifically in skin H&E tissue samples?***

It's a well-known fact that old age is correlated with virtually every disease that are the leading causes of death- heart disease, cancer, stroke, Alzheimer's, and recently,
COVID-19. To help facilitate possible interventions to prevent these chronic age-related diseases, it is critical to develop methods to accurately calculate one's biological age.
And as we've seen above, the easiest way of telling the approximate age of someone is to look at their face, or *skin*. Therefore, this study aims to find features related to aging
in the skin, and like any other organs, skin function also  deteriorates with age. For example, as we age, the skin loses its structural integrity and self-renewal capabilities, 
and experiences increased inflammation and poor temperature regulation. Look at the below diagram:

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/skin_background.png" title="Skin Background" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diagram comparing major differences between young and old skin. Diagram from <i>Orioli et al (Cells 2018) Epigenetic Regulation of Skin Cells in Natural Aging and Premature Aging Diseases</i>.
</div>

As seen in the above diagram, we already know the major differences between young and old skin. However, there has been lack of research done in the biomarkers of aging
at a ***cellular level***. [Previous Research](https://www.nature.com/articles/s41551-017-0093) in our lab has revealed biomolecular and biophysical biomarkers at the cellular level
by analyzing primary dermal fibroblasts. The purpose of this study is to extend the results of this study in finding ***specific morphological biomarkers correlated with aging by utilizing skin H&E images
to shed more light in the biological aging process.***

---

### ***Methods:***
<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/1_project/methods.png" title="Methods" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Simplified diagram showing overall methods of this project.
</div>

As seen in above diagram, we first obtain skin tissue samples, cut them into serial sections, apply H&E stain, and scan and digitize each
slide to obtain a z-stack of 2D WSIs. Then, we undergo image registration to registrate all of the z-stack together and divide the images into
a train-val-test split. 

We train two different models: 1) Nuclei segmentation model using [HoVer-Net](https://arxiv.org/pdf/1812.06499v5.pdf) and 2) Tissue segmentation model
using [DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf). In order to train these supervised models, we manually annotate the nuclei or the twelve different tissue
classes in the training and validation images.

Then, after confirming the model performance on the test dataset, we use a post-processing pipeline to extract different meaningful 2D features. Some example features extracted are 
individual tissue compositions and its area, distance between different cells and tissues (ex. distance between fibroblast and sweat glands), and more. A total of 1090 2D features were 
extracted.

---

### ***Results & Discussion:***

Results and Discussion will be updated after the conclusion of my research. Currently, 2D results are being finalized and 3D version of the same pipeline is being developed.

---

*Image credits to:*
- [95 Year Old Man](https://www.gq.com/story/how-does-a-95-year-old-runner-stay-in-shape)
- [Tom Brady](https://en.wikipedia.org/wiki/Tom_Brady)
- [Young and Old Skin Comparison Diagram](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6315602/)