# OBJECT DETECTION WITH TRANSFORMERS

## Overview
Object detection is one of the fundamental tasks in computer vision that involves locating and classifying objects within an image. Over the years, convolutional neural networks (CNNs) have been the primary backbone for object detection models. However, the recent success of transformers in natural language processing (NLP) has led researchers to explore their potential in computer vision as well. The transformer architecture has been shown to be effective in capturing long-range dependencies in sequential data, making it an attractive candidate for object detection tasks.

In 2020, Carion et al. proposed a novel object detection framework called DEtection TRansformer (DETR), which replaces the traditional region proposal-based methods with a fully end-to-end trainable architecture that uses a transformer encoder-decoder network. The DETR network shows promising results, outperforming conventional CNN-based object detectors while also eliminating the need for hand-crafted components such as region proposal networks and post-processing steps such as non-maximum suppression (NMS).

Since the introduction of DETR, several modificationsand improvements have been proposed to overcome its limitations, such as slow training convergence and performance drops for small objects. In this repository, I will explore the DETR model and its improved versions."

-   <a href=#> DETR </a>
-   <a href="./docs/deformable-attn.md"> Deformable-Attention </a> (Done)
-   <a href=#> DEB DETR </a>
-   <a href=#> DN DETR </a>
-   <a href="./docs/dino-detr.md"> DINO </a> (Done)
-   <a href=#> GLIP model </a> 
-   <a ref=#> Grounding DINO </a>

