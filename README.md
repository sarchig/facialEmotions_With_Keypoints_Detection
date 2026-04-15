# Facial Expression Detection using Emotion & Keypoints

## 📌 Overview
This project implements a **dual-pipeline deep learning system** for facial analysis:
1. **Facial Keypoint Detection** – predicts 15 facial landmarks (30 coordinates)
2. **Facial Emotion Classification** – classifies images into 7 emotion categories

The system combines both outputs to generate a **unified prediction**, overlaying facial landmarks along with the detected emotion.

---

## 🚀 Key Features
- Dual-model architecture (Keypoints + Emotion)
- Transfer learning with pretrained CNN backbones
- Frozen backbone strategy for efficient training
- Data augmentation for improved generalization
- Comparative benchmarking across multiple architectures
- Combined inference pipeline

---

## 🧠 Models Used (Backbones)
The following ImageNet-pretrained CNN architectures were evaluated:
- MobileNetV2  
- ResNet50V2  
- ResNet152V2  
- VGG16  
- DenseNet201  
- InceptionV3  

All models use:
- Frozen convolutional layers (feature extraction)
- Custom classification/regression head

---

## 🏗️ Architecture

### Shared Custom Head
- Global Average Pooling
- Dense Layer (512 units)
- Dropout (0.3)
- Output Layer

### Task-Specific Outputs
- **Keypoint Model**
  - Output: 30 values (x, y coordinates)
  - Activation: ReLU / variants
  - Loss: MSE / Huber

- **Emotion Model**
  - Output: 7 classes
  - Activation: Softmax
  - Loss: Categorical Cross-Entropy

---

## 📊 Datasets

### 1. Facial Keypoints Dataset
- Source: Kaggle
- Image Size: 96×96 (grayscale)
- Labels: 15 facial landmarks (x, y pairs)
- ~2,000 images (augmented to 3×)

### 2. Facial Emotion Dataset
- Source: ICML Face Data
- Image Size: 48×48 (resized to 96×96)
- Classes:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- >30,000 images

---

## ⚙️ Data Preprocessing
- Normalization: Pixel values scaled to [0,1]
- Grayscale → RGB conversion
- Resizing to 96×96
- Data augmentation:
  - Flipping (horizontal/vertical)
  - Brightness adjustment
  - Rotation, zoom, shift (emotion dataset)

---

## 🧪 Training Configuration

| Parameter        | Keypoint Model        | Emotion Model        |
|-----------------|----------------------|---------------------|
| Optimizer       | Adam                 | Adam                |
| Learning Rate   | 1e-4, 1e-3, 1e-2     | 1e-4, 1e-3          |
| Batch Size      | 32 / 64              | 32 / 64             |
| Epochs          | 25–100               | 25–50               |

### Callbacks
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

---

## 📈 Evaluation Metrics

### Keypoint Detection (Regression)
- Mean Squared Error (MSE)
- Huber Loss
- Mean Absolute Error (MAE)

### Emotion Classification
- Accuracy (Top-1)
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🔍 Results & Insights
- **MobileNetV2**: Fastest, lightweight, good baseline performance
- **ResNet / DenseNet**: Higher accuracy potential but computationally expensive
- **InceptionV3**: Strong multi-scale feature extraction
- **Best Hyperparameters (example)**:
  - Batch size = 64
  - Learning rate = 1e-4
  - ELU activation performed better than ReLU

---

## 🔗 Inference Pipeline
The final system:
1. Takes an input image
2. Predicts facial keypoints
3. Classifies emotion
4. Outputs:
   - Landmark coordinates
   - Emotion label

---

## 📁 Project Structure
# facialEmotions_With_Keypoints_Detection
Complete deep learning project that focuses on two important areas of facial analysis: facial keypoint detection and facial emotion recognition. The project utilizes pre-trained Convolutional Neural Networks (CNNs) and various data augmentation techniques to build robust models capable of accurately performing these tasks.
