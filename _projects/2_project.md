---
layout: page
title: Gene Delivery Using Lipid Nanoparticle-based Immunoengineering Approach 
description: Development of Targeted mRNA/pDNA Vaccines for Cancer/Malaria Prevention and Protection 
moredescription: <i>Undergraduate Research Assistant Project (2021 ~ 2022)</i>
img: assets/img/2_project/lnp_thumbnail.png
importance: 2
category: research
---

---
### **Project Motivation & Background:**
Nucleic acid-based therapeutics have been emergent, as more and more nucleic acid therapeutics are being approved due to their
ability to treat diseases by targeting the genetic blueprint itself. Four main types of nucleic acid-based therapeutics are: 1) antisense
oligonucleotides (ASO), 2) ligand conjugated small interfering RNA (siRNA), 3) adeno-associated virus vectors (AAV), and 4) lipid nanoparticles (LNP).
Recently, more and more nucleic acid-based therapeutics have been approved by the FDA, the most famous being the two COVID-19 vaccines during the pandemic (mRNA-
based therapeutic). Recently, the FDA approved Adstiladrin, which is an AAV-based therapeutic for non-muscle-invasive bladder cancer (NMIBC), so more and more
nucleic acid-based therapeutics are hitting the market in the oncology space as well. 

Compared to conventional therapeutics that usually target proteins that results in transient therapeutic effects, nucleic-acid based therapeutics are often much
longer-lasting or even permanent depending on the nucleic acid used and the target. However, most nucleic-acid based therapies require a carrier, as they will be
targeted by the immune system for degradation and clearance and therefore will cause unwanted inflammation and toxicity that can be detrimental or even fatal to the patient.
*Therefore, this project's research was focused on the design of the lipid nanoparticles, or LNPs, for effective delivery of pDNAs/mRNAs to develop vaccines for malaria/cancer.*

To briefly introduce the background of LNPs, LNPs usually are constituted of four components: 1) Ionizable lipid, 2) PEGylated lipid, 3) Cholesterol, and 4) Helper lipid. The first three components 
were fixed, and six different helper lipids were explored, with the ratios of the four lipids being varied as well. A total of 1080 different LNPs were designed, with two of the helper lipids being
cationic, anionic, and zwitterionic. The ratio and the different helper lipids have a huge impact on the surface charge, pKa, and size of LNPs, which all have a huge influence in the successful delivery of the 
drug payload. *Therefore, the project's main motivation is not only for successful delivery of the drug payload, but also about optimizing the formulation of the LNP beforehand.*

**To briefly introduce the two projects:**
<br>
For the *malaria pDNA vaccine project*, the proposed route of administration was oral, or through the mouth. This makes sense, as it is a wide known fact that oral drugs experience a
"first pass effect", where the orally taken drugs go through the intestines and end up in the liver. This is why drugs that do not target cells in the liver cannot be taken
orally as it is often times metabolized in the liver before it even has a chance to circulate in the bloodstream. In addition, the low pH environment of the stomach due to
gastric acid also necessitates a delivery mechanism such as LNPs. However, because the target of the pDNA malaria vaccine is hepatocytes, an oral administration was fitting. Furthermore,
pDNA is used rather than mRNA because pDNA has a more long-lasting effect than mRNAs do, however they are therefore more likely to elicit an unwanted immune response, which was known to be 
alleviated by co-delivery of anti-inflammatory siRNA.

For the *cancer mRNA vaccine project*, the proposed method of administration was intramuscular, or through the muscle tissue. This is the same route of administration as the COVID-19 vaccines,
as it is most effective to elicit a potent immune response. The cancer vaccine can either be prophylactic or therapeutic, but either way, it has to be able to deliver the mRNA that codes for the 
specific tumor-specific antigens (TSA) that the T cells can recognize and kill the tumor cells. In order to ensure this happens effectively, below is the proposed entire uptake and trafficking scheme of the mRNA LNPs:

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/2_project/mRNA_uptake.png" title="mRNA uptake" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Proposed scheme showing the uptake and trafficking of mRNA LNPs.
</div>

As we see above, mRNA LNPs are designed to enter the target antigen-presenting cells (APCs), which in our case are dendritic cells (DCs). They enter the cells via endocytosis, and it is important
to design the LNPs in a way so they initiate *endosomal escape*, or else they are degraded by the lysosomes. After escaping the endosome by breaking the barrier of it, the mRNA is released in to the cytoplasm 
of the DCs, which initiates translation of the desired TSAs. Now, in DCs, DCs use the proteasome-TAP pathway as a main, conventional route for cross-presentation of the TSAs via MHC Class I molecules. In short, the 
translated TSA is broken into peptides by the proteasome, and the peptide is translocated by the TAP transporters into the lumen of the endoplasmic reticulum (ER). Then, the MHC Class I complex containing peptides
and other molecules are assembled in the ER and transported to the cell surface. Lastly, as seen in the diagram, the CD8+ T-cells recognize the MHC Class 1 complex/molecule via its surface receptor, or T cell receptor (TCR). 
When this cross-presentation is successful, with the help of other signals called co-stimulary signals, the CD8+ T cell is *activated*, meaning it can now detect and kill target tumor cells via recognizing their TSAs.

---

### **Methods & Results:**

The methods for screening the 1080 LNPs for the pDNA malaria vaccine project are shown in below diagram. 

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/2_project/lnp_optimization_method.PNG" title="LNP optimization" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Proposed method to screen 1080 LNPs of different lipid composition and helper lipid types.
</div>

***Screening Process:***
<br>
1) Component and ratio screening was done via *in vitro* transfection for each of the 1080 LNPs. Each LNPs were synthesized and tested for pDNA transfection in human HepG2 cells via using
50% mCherry and 50% Luc pDNA. Then, subsequent luciferase protein expression (more pDNA delivered, more bioluminescence in the cells) was tested and as a result, top 32 formulations for each of the six helper lipids
were selected.
2) An *in vivo* screening via intrahepatic injection was performed, but in *batch mode*. Just like in COVID-19 testing, where multiple patients' samples are merged together to test for COVID, the same logic was applied here.
8 of the top 32 formulations were grouped, meaning there were four clusters per helper lipid with a total of 24 clusters tested for intrahepatic injection. The same mix of mCherry and Luc pDNA were injected,
and transfection efficiency were determined by imaging the mice liver via in vivo imaging system (IVIS). Furthermore, flow cytometry was used to measure mCherry levels in different types of cells in the liver to see which cells
were transfected. As a result, top 12 clusters were selected for having the highest bioluminescence in the liver.
3) Again, subsequent *in vivo* screenings via intravenous injection was performed, which was in batch mode as well. The same procedure as before was repeated twice, until only top four formulations remained.

***Post-Screening Process:***
<br>
Then, after the screening process was over that left us with top four formulations, a final *in vivo* screening was performed via using Cre-Ai9 mouse. A Cre-Ai9 mouse, when injected with a Cre pDNA loaded LNP, 
allows the expression of tdTom only if the pDNA is successfully delivered. Following experiments using confocal imaging and flow cytometry was used to compare the top four formulations effectiveness. Lastly,

The above method was for the pDNA malaria vaccine project, and more details and further experiments are explained in the published paper [here](https://www.nature.com/articles/s41467-022-31993-y). 
For the mRNA cancer vaccine project, the same screening process can be used, but just in a different applied method:
1) Component and ratio screening was done via *in vitro* transfection for the same 1080 LNPs. Each LNPs were synthesized, and tested for mRNA transfection in bone-marrow dendritic cells (BMDCs). Then, after waiting for
a few days, we collect the BMDCs and co-culture them with CD8+ T cell. Then, we measure level of cell proliferation to test if MHC class I cross-presentation happened or not.

Note that I cannot reveal the detailed methods and results for the mRNA cancer vaccine project since it has not been published yet. Below is the summary of the experiments that I partaked in- some experiments were independent,
and others were in a group:

1. In pDNA malaria vaccine project:
- Formulated and screened ~1080 LNPs via cluster-mode in vitro transfection assays and in vivo intrahepatic/intraduodenal injections.
- Tested Ai9 Cre reporter mice for co-delivery of anti-inflammatory siRNA with pDNA for in vivo assays of top formulations. 
- List of other experiments not mentioned: 

2. In mRNA cancer vaccine project:
- Formulated and screened ~1080 LNPs via cluster-mode in vitro transfection and immunostimulatory assays on dendritic cells.
- Tested Ai9 Cre reporter mice for mRNA delivery and participated in subsequent therapeutic and prophylactic mice tumor studies.
- List of other experiments not mentioned:

---

### **Personal Motivation:**

covid-19 vaccine motivation, cancer diagnosis/prevention 

---