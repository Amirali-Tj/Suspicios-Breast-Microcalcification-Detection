# 🔬 Suspicious Breast Microcalcification Detection

> A deep learning pipeline for **automatic detection and classification of suspicious breast microcalcifications** in digital mammography using a **two-stage patch-based framework** powered by **ResNet-50 + CBAM**.

---

## 📖 Overview

Breast microcalcifications are among the earliest radiological indicators of breast cancer. Their extremely **small size**, **low contrast**, and **scattered distribution** make automatic analysis a challenging computer vision task.

To address this problem, this project adopts a **patch-based deep learning pipeline**. Instead of processing an entire mammogram at once, the image is divided into smaller patches, enabling the models to focus on subtle local patterns while preserving high-resolution details.

The proposed framework consists of **two sequential deep learning models**:

1. 🔍 **Detection Model**
   - Detects patches containing microcalcifications.
   - Filters out normal tissue.

2. 🩺 **Classification Model**
   - Receives the detected candidate patches.
   - Classifies each patch as:
     - 🔴 Suspicious
     - 🟢 Benign

Both models are based on **ResNet-50 enhanced with the Convolutional Block Attention Module (CBAM)** to improve feature learning.

---

## 🧪 Experimental Setup

The proposed framework was evaluated using **5-fold cross-validation** for both stages of the pipeline:

- 🔍 **Stage 1:** Microcalcification Presence Detection
- 🩺 **Stage 2:** Suspicious vs. Benign Classification

For each fold, an independent model was trained and evaluated. To facilitate reproducibility and further research, this repository includes the **best-performing model weights** obtained from each fold for both stages.

These pretrained weights can be used directly through the provided CLI for inference or as initialization for future research and development.

---

## 📦 Pretrained Models

The repository provides pretrained weights for:

- 🔍 Detection models (Microcalcification Presence)
- 🩺 Classification models (Suspicious vs. Benign)

The released weights correspond to the **best-performing model from each fold** of the 5-fold cross-validation experiments.


## ✨ Features

- 🧠 ResNet-50 backbone enhanced with **CBAM**
- 🔍 Two-stage detection and classification pipeline
- 🧩 Patch-based mammogram analysis
- 📍 Detection of candidate microcalcification regions
- 🩺 Classification into **Suspicious** or **Benign**
- 💻 Easy-to-use Command Line Interface (CLI)
- 🛠 Modular and extensible project structure

---

## 🏗️ Model Architecture

### 🔍 Stage 1 — Candidate Detection

**Input**
- Mammography patches

**Backbone**
- ResNet-50 + CBAM

**Output**
- Probability of microcalcification presence

Only patches predicted to contain microcalcifications are forwarded to the second stage.

---

### 🩺 Stage 2 — Candidate Classification

**Input**
- Detected candidate patches

**Backbone**
- ResNet-50 + CBAM

**Output**
- 🔴 Suspicious
- 🟢 Benign

---

## 🔄 Pipeline

```text
📷 Digital Mammogram
          │
          ▼
🧩 Patch Extraction
          │
          ▼
🔍 Detection Model
(Microcalcification?)
          │
     Positive Patches
          │
          ▼
🩺 Classification Model
          │
    ┌─────┴─────┐
    ▼           ▼
🔴 Suspicious  🟢 Benign
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Amirali-Tj/Suspicios-Breast-Microcalcification-Detection.git

cd Suspicios-Breast-Microcalcification-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

The repository provides a convenient **Command Line Interface (CLI)** for inference. the CLI generates intuitive heatmaps that visualize the spatial distribution of the model's confidence across the mammogram. Depending on the selected inference mode, the heatmaps represent either the presence of microcalcifications or the likelihood of suspicious microcalcifications, providing an interpretable visualization of the model's predictions.

```bash
python mammoAnalyzer.py <input_image> <output_image> -T <type of analysis>
```
### Available Types

#### 🔍 `presence`

Detects the **presence of microcalcifications** in the mammogram.

Output :
- 🌡️ Presence probability heatmap

---

#### 🩺 `sus`

Detects **suspicious microcalcifications**.

Output :
- 🌡️ Suspiciousness heatmap

> ⚠️ **Research Use Only:** The pretrained models included in this repository are intended solely for research and educational purposes. They are **not fully approved for clinical use** . 

Typical workflow:

1. 📥 Load trained models
2. 🖼 Read the mammography image
3. 🔍 Detect candidate patches
4. 🩺 Classify detected patches
5. 📊 Generate heatmaps

---

## ⚙️ Methodology

The complete inference pipeline follows these steps:

1. 🖼 Preprocess the mammogram
2. 🧩 Extract image patches
3. 🔍 Detect patches containing microcalcifications
4. 🚫 Discard negative patches
5. 🩺 Classify remaining candidates
6. 📊 Report suspicious findings

---

## 🧠 CBAM Attention Module

To improve feature representation, the backbone network incorporates the **Convolutional Block Attention Module (CBAM)**.

CBAM sequentially applies:

- 🎯 Channel Attention
- 🎯 Spatial Attention

This allows the network to emphasize the most informative regions and subtle microcalcification patterns.

---

## 📚 Citation

If you use this repository in your research, please consider citing the corresponding publication (if available).

