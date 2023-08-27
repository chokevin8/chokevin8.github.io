---
layout: page
title: Evaluation and Improvement of Gene Signatures Using Single-cell RNA Sequencing (scRNA) Dataset in Lung Cancer 
description: Deconvolution of In-House Gene Signature Usage for Better Utilization of Gene Signatures 
moredescription: <i>Oncology Data Science Summer Intern Project at Novartis Institutes of Biological Research (NIBR) (2022)</i>
img: assets/img/3_project/novartis-intern_thumbnail.png
importance: 3
category: internship
---

---

### **Project Motivation & Background:**

<p>
As the precision medicine era comes closer and closer, gone are the times when breast cancer patients are all grouped together for a new clinical trial.
Clinical biomarker studies that are conducted by pharmaceutical companies reveal the importance of biomarker-driven oncology clinical trials to not only boost
the clinical effectiveness of the drug, but also to position their drugs properly in the market. Nowadays, all cancer patients are divided into various tumor subtypes
based on a positive/negative testing for certain prognostic biomarkers for that specific type of cancer. 
</p>
<p>
For example, in breast cancer, the first-line therapy for triple-negative breast cancer (TNBC) and HER2-positive cancer are different. Additionally, even within a tumor subtype such as TNBC, the first-line therapy
may be different depending on whether one tests positive for a BRCA mutation or/and PD-L1 protein. Furthermore, while not a specific protein biomarker, general biomarkers such as
tumor mutation burden (TMB) and microsatellite instability (MSI) can both be a great predictive biomarker to determine the best therapy for a patient.
</p>
<br>

To develop the above tumor subtype differentiation, analyzing the patient's cancer transcriptome via **RNA sequencing (RNA-seq)** is the main method to analyze the heterogeneity of the tumor subtype
and possibly discover novel biomarkers or therapeutic strategies. From the RNA-Seq data, ***gene expression signatures (GS)*** are constructed after pre-processing of the raw RNA-Seq data (QC, normalization, etc). 
*GS are sets of genes that are comprised of multiple individual member genes that show a unique gene expression pattern (GEP), which is a result of a biological or pathogenic process.*
Below is an example of a cross-correlation heatmap of all pairwise member gene pairs that show high correlation for B cell related GS in lung cancer.

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/3_project/b_cell_GS.png" title="BCell GS" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cross-correlation heatmap of member gene pairs showing high correlation for B cell related GS in lung cancer.
</div>

<p>
Above, we see several highly correlated genes, with the member genes being all related to B cell function such as activation and proliferation. Compared to 
evaluating individual gene expression and its correlation to lung cancer, evaluating GS as seen above is statistically much more significant. However, while it is true that these GS
can be significant in discovering new prognostic/predictive biomarkers and therapeutic strategies, its potential can be often times exaggerated. 
</p>

Nowadays, GS are used everywhere in RNA-Seq studies for clinical biomarker studies, and as a result, there are a plethora of different GS that are available (whether the GS
is an in-house or publicly available GS). Since GS is constructed from RNA samples, which can be vastly different depending on the experimental conditions and the indication fo the sample as well,
we often have numerous GS for a single biological/pathological process- which makes GS highly redundant. For example, a B-cell GS that is different from the one above can be constructed if 
the experimental conditions, or the patient pool is different. Furthermore, many biological processes (ex. immune cells) are studied at the same time, which results in a GS with significant gene overlap.
Therefore, understanding interdependencies among the member gnes of the GS can have implications on how a GS should be used. 

***Therefore, these redundancies and complexities calls for a method that could be utilized to continuously evaluate, improve, and finally deconvolve the 
usage of GS in these studies. In this internship project, I worked on seeing if the in-house curated GS called "OncDS GS" and the publicly available and wisely used "MSigDB C6/Hallmark GS" were 
"good enough", or applicable to some of the commonly used RNASeq datasets (bulk and pseudobulk single-cell (sc) RNASeq dataset) for clinical biomarker analysis.*** 

---

### **Methods:**


---

### **Results:**

*Note that unfortunately, due to company (Novartis) guidelines, only the results available in the [poster](https://docs.google.com/presentation/d/1YHDFwXkiKVQMFpiFV2PlcCRim0-ycFSD/edit?usp=sharing&ouid=102273945805745041682&rtpof=true&sd=true) are available, and any other results cannot be 
shown. For the same reason, the pipeline of this analysis, which was fully written in R, cannot be shown either.*



---

### **Personal Comments:**

### Q: Why did I choose this project? ###

### Q: What did I do outside of this project? ###

### Q: What impact did this project have on me? ###

---