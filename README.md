# Arabic Lip Reading Project

## 📝 1. Project Overview

This project focuses on **Arabic lip reading**, aiming to predict the spoken word from short video clips of speakers. The system combines **3D spatio-temporal convolutions**, **MobileNetV3-Small backbone**, **bidirectional LSTM**, and **attention mechanisms** to achieve accurate word recognition from lip movements.

The project is ongoing, and the current goal is to improve the **accuracy of prediction**, optimize the architecture, and analyze the influence of phonetic properties on performance.

---

## 🖥 2. System Differentiation

The system is designed to handle **isolated word lip reading in Arabic**, which presents unique challenges:

* High similarity in **bilabial and dental phonemes** (e.g., ب / م / ن / ط)
* Variation in word length and **frame availability**
* Limited computational resources

Unlike typical lip reading systems trained on English datasets, this project focuses on **Arabic-specific phonemes** and word structures, leveraging **3D CNNs for temporal features** and **attention mechanisms** for better feature selection.

---

## 📂 3. Dataset

The dataset used is **LRW-AR**, an Arabic adaptation of the LRW (Lip Reading in the Wild) dataset.

[Dataset link](https://crns-smartvision.github.io/lrwar/)



---

## 🎯 4. Training Scope & Class Selection

Due to computational constraints, the model is trained on **23 classes only**.

**Selection criteria:**

* Balanced dataset availability
* Computational feasibility, considering **الفئة الصوتية** (phonetic category) and **طول الكلمة** (word length)

These 23 classes allow focused experimentation while maintaining meaningful results.

---

## 🏗 5. Model Architecture

The Arabic Lip Reading model consists of:

1. **3D CNN Spatio-Temporal Embedding**

   * Extracts **temporal and spatial features** from video sequences
   * Two 3D convolution layers with batch normalization, ReLU, and dropout

2. **Skip Connection & Projection**

   * Preserves low-level features from the 3D CNN
   * Adaptive pooling + linear projection

3. **MobileNetV3-Small Backbone**

   * Extracts per-frame features
   * Lightweight and efficient for fine-grained lip movements

4. **Channel Attention**

   * Highlights **important channels** for each frame
   * Reduces irrelevant feature noise

5. **Bidirectional LSTM**

   * Captures **temporal dependencies** across frames

6. **Temporal Attention**

   * Focuses on **important frames** in the sequence

7. **Feature Fusion & Classifier**

   * Combines LSTM context and CNN features
   * Fully connected layers with dropout, ReLU, and batch normalization
   * Output: **23 class predictions**

**Current model stats:**

* Total parameters: ~**3.5M**
* Trainable parameters: ~**3.5M**
* Dropout applied at multiple stages for regularization

---

## 📊 6. Accuracy & Analysis

* Current top-1 accuracy: **73%**
* Analysis shows **words with similar bilabial and dental phonemes** (e.g., ب / م / ن / ط) have lower discrimination
* Dentals, pharyngeals, and **words with fewer frames** also reduce accuracy
* Ongoing work focuses on:

  * Expanding class coverage
  * Improving model efficiency
  * Analyzing **phonetic influence** on word-level accuracy

---

## 🚀 7. Current Status

* Model training completed for **23 classes**
* Accuracy analysis performed at the word level to identify phonemes causing misclassification
* Improving architecture and dataset usage to enhance overall performance

---

This project is actively **under development to increase accuracy** and expand class coverage while analyzing phonetic impacts on lip reading performance.
