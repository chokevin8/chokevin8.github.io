---
layout: page
title: {% assign words = "2/3D Semantic Segmentation of Skin H&E Tissue Images" | split: ' ' %}
    {% capture titlecase %}
      {% for word in words %}
        {{ word | capitalize }}
      {% endfor %}{% endcapture %}
    {{ titlecase }}
description: Segmentation of Skin H&E Tissue Images to Analyze Novel Cellular Biomarkers of Aging
img: assets/img/h&e_thumbnail.PNG
importance: 1
category: research
---


