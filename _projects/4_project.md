---
layout: page
title: Differential Gene Analysis and Pathway Analysis of TCGA Patient RNA-Seq Dataset 
description: Design of Pipeline to Search for Possible Drug Targets via RNA-Seq Analysis 
moredescription: <i>Data Science Intern Project at Cowell Biodigm Co. (2020)</i>
img: assets/img/4_project/cbd_intern-thumbnail.jpg
importance: 4
category: internship
---

---

### **Project Motivation & Background:**
A drug target is a specific molecule or a biological entity (ex. PD-L1) that a drug will target as a therapeutic intervention. 
In the drug discovery space, we have top-down and a bottom-up approach in discovering drug targets. A top-down approach is where
researchers start with a certain disease (ex. liver cancer or hepatocellular carcinoma) and work backwards to identify potential drug targets. 
If the pathology of the disease such as underlying molecular mechanisms and associated pathways, experiments must be followed to identify the key molecules and
pathways that are associated with the disease. A bottom-up approach is the opposite, as it starts with a focus on the molecular components and pathways that are known to be involved 
in various different cellular processes. This is different, as these molecular components and pathways will be associated with many different
diseases, and once a specific molecular component and pathways are defined, researchers look for certain possible targets for certain diseases. 

Computational drug target identification has been widely used over the industry, and there are many different methods to do this. One method that is widely used, and the method that I used,
was the phenotype-based method. Phenotype-based computational drug target identification compares the biological phenotype, which are the -omics data. Out of the four major -omics data (genomics, transcriptomics, proteomics, metabolomics),
I focused on transcriptomics, or RNA-Seq data. RNA-Seq data contains many key components, but the most important component is the **gene annotation and its read count.**
Gene annotations are essentially the labels for the gene, and allows to identify the gene and its associated pathways. Read counts are the gene expression levels, higher the read count the higher the gene expression level is.
Typically, when doing a RNA-Seq analysis, data from two opposite groups are given (ex. healthy vs disease, treated vs untreated), and in my case, I looked 
cancer vs healthy RNA-Seq data. When given these two sets of data, the standard protocol is to first perform a **differential gene expression (DGE) analysis** and then a 
**pathway analysis** with the list of genes that are differentially expressed. This makes sense because the list of genes that show different level of expressions between the two group
probably has something to do with the disease, and then we take those list of genes and perform a pathway analysis to see which biological pathways are associated with the list of genes.

bonferroni correction multiple testing

*Context: For some context, my research internship was at a small start-up company focused on early drug discovery. The company aimed to discover new drug compounds or new drug targets for a known compound. Then,
the company would aim to out-license these early "hits" or "leads" to a bigger pharmaceutical company for profit.*

---

### **Methods:**


---

### **Results & Discussion:**


---

### **Personal Comments:**

### Q: Why did I choose this project? ###

### Q: What did I do outside of this project? ###

### Q: What impact did this project have on me? ###

---