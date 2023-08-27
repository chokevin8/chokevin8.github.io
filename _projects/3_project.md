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

As the precision medicine era comes closer and closer, gone are the times when breast cancer patients are all grouped together for a new clinical trial.
Clinical biomarker studies that are conducted by pharmaceutical companies reveal the importance of biomarker-driven oncology clinical trials to not only boost
the clinical effectiveness of the drug, but also to position their drugs properly in the market. Nowadays, all cancer patients are divided into various tumor subtypes
based on a positive/negative testing for certain prognostic biomarkers for that specific type of cancer. For example, in breast cancer, the first-line therapy for 
triple-negative breast cancer (TNBC) and HER2-positive cancer are different. Additionally, even within a tumor subtype such as TNBC, the first-line therapy
may be different depending on whether one tests positive for a BRCA mutation or/and PD-L1 protein. Furthermore, while not a specific protein biomarker, general biomarkers such as
tumor mutation burden (TMB) and microsatellite instability (MSI) can both be a great predictive biomarker to determine the best therapy for a patient.

To develop the above tumor subtype differentiation, analyzing the patient's cancer transcriptome via RNA sequencing (RNA-seq) is the main method to analyze the heterogeneity of the tumor subtype
and possibly discover novel biomarkers or therapeutic strategies. From the RNA-Seq data, gene expression signatures (GS) are constructed after pre-processing of the raw RNa-Seq data. 
GS are sets of genes that are comprised of multiple individual member genes that show a unique gene expression pattern (GEP), which is a result of a biological or pathogenic process.
Below is an example of a cross-correlation heatmap of all pairwise member gene pairs that show high correlation for B cell related GS in lung cancer. 

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/3_project/b_cell_GS.png" title="BCell GS" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cross-correlation heatmap of member gene pairs showing high correlation for B cell related GS in lung cancer.
</div>

In above, we see several highly correlated genes, with the member genes being all related to B cell function such as activation and proliferation. Compared to 
evaluating individual gene expression and its correlation to lung cancer, evaluating GS as seen above is statistically much more significant. However, while it is true that these GS
can be significant in discovering new prognostic/predictive biomarkers and therapeutic strategies, its potential can be often times exaggerated. 







---

### **Methods:**


---

### **Results:**

*Note that unfortunately, due to company (Novartis) guidelines, only the results available in the [poster]() are available, and any other results cannot be 
shown.*

---

### **Personal Comments:**



---