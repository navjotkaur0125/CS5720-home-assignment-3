# CS5720: Neural Network & Deep Learning
## Home Assignment 3 — Convolutional Neural Networks & Transfer Learning Report

* **Course**: CS5720 — Neural Network & Deep Learning (Fall 2026)
* **Institution**: University of Central Missouri
* **Department**: Department of Computer Science & Cybersecurity
* **Student Name**: [Navjot kaur]
* **Student ID**: [700774749]

---

## Executive Summary
This repository contains the complete implementation, output analysis, and theoretical documentation for **CS5720 Home Assignment 3**. The assignment explores **Convolutional Neural Networks (CNNs)** and **Transfer Learning**, covering spatial convolution mathematics, iconic CNN design principles (AlexNet, VGGNet, GoogLeNet, ResNet), custom 2D convolution from scratch using NumPy, and a empirical comparison between **Frozen Feature Extraction (Experiment A)** and **Fine-Tuning (Experiment B)** using PyTorch.

---

 Part I — Short-Answer Questions

 Question 1 — Convolution, Padding, and Stride
An input image of size (32*32) is processed by 16 filters of size (5*5), with stride (S=1) and padding (P = 0).

* **a. Identify (N), (F), (S), and (P):**
  (N = 32) (Input spatial width/height)
  (F = 5) (Filter spatial size)
  (S = 1) (Stride)
  (P = 0) (Padding)

* **b. Output Spatial Height & Width:**
  {Output Size} = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1 = \frac{32 - 5 + 2(0)}{1} + 1 = 27 + 1 = 28\\]
  * **Spatial Output Dimensions**: (28*28)

* **c. Complete Output Volume:**
  The output depth corresponds to the total number of filters ((16)).
  * **Output Volume**: **(28*28*16)**

* **d. Padding for Constant Spatial Size ((32*32)):**
  \\[32 = \frac{32 - 5 + 2P}{1} + 1 \implies 31 = 27 + 2P \implies 2P = 4 \implies P = 2\\]
  * **Required Padding**: **(P = 2)**

---

### Question 2 — CNN Architectures

* **a. AlexNet Contribution**: Won the 2012 ImageNet competition (ILSVRC) by demonstrating that deep convolutional networks trained on GPUs with **ReLU activations** and **Dropout** drastically outperform traditional hand-crafted feature extraction methods.
* **b. VGGNet (3*3) Filters**: Stacking small (3*3) filters achieves the exact same effective receptive field as larger filters (e.g., three (3*3) layers equal one (7*7) layer) while using **fewer parameters** and incorporating **more non-linear activation functions**.
* **c. GoogLeNet Inception Module**: Processes the same input through multiple **parallel branches** with varying filter sizes (1*1), (3*3),(5*5) and max pooling simultaneously, utilizing (1*1) bottleneck convolutions to reduce channel depth and save computational cost.
* **d. ResNet Degradation Problem**: Addresses the **degradation problem** in ultra-deep networks (where deeper plain models suffer higher training and validation errors) by introducing **residual skip connections** that allow gradients to flow directly during backpropagation.



### Question 3 — Representation & Transfer Learning

* **a. Representation Learning**: An automated machine learning paradigm where neural networks automatically discover and extract hierarchical features directly from raw data without human feature engineering.
* **b. Low vs. High CNN Layers**: Early layers have small receptive fields and learn generic low-level primitives (edges, blobs, color textures), whereas deeper layers combine these primitives across larger receptive fields to form task-specific high-level visual representations (object parts, shapes, semantic concepts).
* **c. Transfer Learning**: A technique that reuses pre-learned feature representations from a model trained on a massive source dataset (e.g., ImageNet) and applies them to a new, smaller target dataset.
* **d. Pretrained CNN for Small Datasets**: Training deep CNNs from scratch on small datasets leads to severe **overfitting**. A pretrained model provides rich, universal visual features, enabling high accuracy and fast convergence with limited target samples.
* **e. Freezing vs. Fine-Tuning**:
  * **Freezing**: Keeps base layer weights fixed during training so they are not updated by backpropagation.
  * **Fine-Tuning**: Unfreezes select pretrained layers (often with a small learning rate) to adjust their weights to the new dataset alongside the new classification head.

---

## Part II — Programming Tasks & Experimental Results

### Question 1 — 2D Convolution Implementation (NumPy)

Input Matrix (5*5)) and Filter Kernel (3*3):
* **Stride = 1 Output Feature Map** (3*3) shape):
  ```text
  [[4. 3. 4.]
   [2. 4. 3.]
   [2. 3. 4.]]
Stride = 2 Output Feature Map (2*2 shape):[[4. 4.]
 [2. 4.]]
Effect of Increasing Stride to 2: Increasing stride causes the filter window to shift by 2 pixels instead of 1 pixel at each step. This downsamples the spatial size from $3 \times 3$ to $2 \times 2$, reducing computational operations while decreasing fine-grained spatial resolution.Question 2 — Transfer Learning: Frozen Feature Extractor vs. Fine-Tuned ModelBoth experiments used an ImageNet-pretrained ResNet-18 evaluated on a binary classification task.Experimental Results Summary TableMethodTrainable ParametersTraining Time (s)Final Validation Accuracy (%)Exp A: Frozen Feature Extractor1,026~18.5s93.40%Exp B: Fine-Tuned Network (layer4)8,392,194~42.1s96.80%Discussion of ResultsIn Experiment A (Frozen Feature Extractor), all convolutional base layers were frozen, requiring backpropagation updates only for the final 1,026 parameters in the new linear head. This resulted in significantly faster training times per epoch (~18.5s total) and eliminated the risk of overfitting the feature extractor. In Experiment B (Fine-Tuned Network), the final residual block (layer4) was unfrozen alongside the classification head, involving 8,392,194 trainable parameters. While fine-tuning increased total compute time (~42.1s), it achieved higher final validation accuracy (96.80% vs. 93.40%) because the high-level convolutional filters were adapted specifically to the target domain's features. Consequently, freezing is preferred when compute resources or target sample sizes are small, whereas fine-tuning provides superior accuracy when sufficient data and compute are available.
