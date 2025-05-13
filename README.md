# The Librarian from Alexandria

## 👥 Team Members - AlexandriaAI
- Denise Di Franza 
- Gabriele De Ieso
- Alessia Tonicello

---

## Section 1: Introduction

This project was developed as part of the Machine Learning course at LUISS. Acting as digital archivists of the Great Library of Alexandria, our task was to design a machine learning model capable of classifying ancient digitized texts by the fonts used in their printing. These scanned pages come from various historical sources and use distinct typographic styles. 

Our goal was to automate the identification of these font styles using a neural network-based image classification system. The final model should assist in organizing and archiving ancient documents by recognizing and categorizing fonts in a scalable, reliable way.

---


## Section 2: Methods

### Dataset
The dataset consists of over 1,000 scanned pages of ancient texts, each labeled with the font used. The files are stored in a CSV (`pages.csv`) referencing image files in the `img/` directory.

### 2.1 Design Choices and Key Ideas
- **Data Quality First**
Before training we: 
- Verified image existence by checking paths and filtering out missing or unreadable files.
- Added a binary `exist` column which flagged whether each image file was present and entries with missing images were dropped to avoid issues during training.


- **Font Label Mapping**
  Since the dataset consists of font names (e.g., "cicero", "vesta") associated with each image, we mapped these strings to integer labels using a Python dictionary. We identified **11 unique fonts**, and assigned values from 0 to 10 for model compatibility, for example:
  ```python
  font_to_label = {
      'cicero': 0,
      'vesta': 1,
      'senatus': 2,
      'trajan': 3,
      # ... up to 11 entries
  }


**1. Preprocessing Strategy**

The scanned pages varied in resolution and clarity, therefore, in order to standardize inputs and reduce noise, we applied:
- Grayscale conversion: reduced complexity from 3 channels to 1.
- Resizing to 224x224, which made the input compatible with standard CNN architectures.
- Binarization with Otsu’s method: separated foreground text from noisy backgrounds.
- Normalization, which scaled pixel values to [0, 1].
- Filtering to remove visually blank or poorly segmented images.



**2. Data Augmentation**

To prevent overfitting and introduce visual variation, we implemented:
- Random rotations (±15°) to simulate scanning misalignment.
- Color jitter to mimic lighting conditions.
- Perspective transformations to replicate page distortion.

These augmentations were implemented using PyTorch’s `transforms.Compose`.



**3. Model Selection**

In order to address the classification of ancient fonts, we explored and compared two different neural network architectures: a custom-built Convolutional Neural Network (CNN) and a pretrained ResNet18 model. Each approach was chosen for its specific strengths and evaluated as part of our experimental design.

Custom CNN – Baseline Architecture:
We first designed a simple yet effective Convolutional Neural Network to serve as a baseline. This model was trained from scratch and allowed us to test the full data pipeline—including preprocessing and augmentation—under controlled conditions.


*Architecture details:*
- **4 Convolutional Blocks**:
  - Each block includes:
    - 2D Convolution layer
    - Batch Normalization
    - ReLU activation
    - Max Pooling
    - Dropout (to reduce overfitting)
- **2 Fully Connected Layers**:
  - Flattened features are passed through linear layers
  - Final output layer with 11 nodes (one per font class)
  - Softmax activation applied for multi-class classification

This architecture was intentionally lightweight to ensure faster training, but it served as a solid benchmark to measure the value added by more complex models.


ResNet18 – Transfer Learning:
To improve the performance, we used a pretrained ResNet18 model from the `torchvision.models` library. ResNet18 has been trained on millions of images from the ImageNet dataset and is known for its ability to learn deep and abstract features.


*Adaptation for our task:*
- All pretrained convolutional layers were frozen
- Only the final fully connected layer was replaced and retrained
- The classifier head was adjusted to output predictions across the 11 classes we had previosuly identified 

This technique, known as *transfer learning*, is particularly effective when dealing with small or medium-sized datasets, like ours. ResNet18 was expected to extract more robust visual features and generalize better than the custom CNN.



**4. Training Setup**
To ensure fair and reproducible training across both models, we adopted a consistent training pipeline, carefully chosen based on best practices in deep learning.

*General Configuration:*
- **Framework**: PyTorch 2.x
- **Device**: CUDA-enabled GPU (if available), otherwise CPU


*Optimization Strategy:*
- Loss Function: `CrossEntropyLoss` (suitable for multi-class classification)
- Optimizer: `Adam`
  - Learning rate: `1e-4`
  - Weight decay: `1e-5` (to prevent overfitting)
- Learning Rate Scheduler: `ReduceLROnPlateau`
  - Monitors validation loss and reduces the learning rate when the model stops improving


*Training Conditions:*
- Batch Size: 32
- Early Stopping: Applied based on validation loss stagnation
- Epochs: Up to 100, but typically stopped early
- Reproducibility:
  - Fixed `SEED = 42` used across NumPy, PyTorch, and Python's `random` module
  - Enabled deterministic behavior on GPU for consistent results


*Data Splitting:*
- 80% of data used for training
- 20% for validation
- Stratified Split: Ensured the distribution of font classes remained balanced in both sets

This setup provided a stable foundation for evaluating model performance under the same experimental conditions.



### 2.2 Environment Reproducibility
We used Python 3.12.6 and a `venv` virtual environment to manage dependencies for this project. All required libraries are listed in the `requirements.txt` file included in the repository.

To recreate our environment and run the project:

```bash
# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

To generate your own `requirements.txt`, use:

```bash
pip freeze > requirements.txt
```

### 2.3 Pipeline Overview
Below is a high-level flowchart of our system:

```
[ Start ] 
   ↓
[ Load dataset and verify image paths ]
   ↓
[ Remove missing or corrupt images ]
   ↓
[ Map font names to integer labels ]
   ↓
[ Split dataset (stratified 80/20) → Train / Validation ]
   ↓
[ Preprocessing: 
   - Grayscale conversion
   - Resize to 224x224
   - Otsu binarization
   - Normalization ]
   ↓
[ Data Augmentation (rotation, jitter, perspective) – train set only ]
   ↓
[ Select architecture: Custom CNN or pretrained ResNet18 ]
   ↓
[ Train model with early stopping and learning rate scheduler ]
   ↓
[ Evaluate on validation set using accuracy, macro F1, and confusion matrix ]
   ↓
[ Save best-performing model + export metrics and confusion matrix image ]

```



---

## Section 3: Experimental Design

With the models defined and the training setup in place, we conducted two core experiments to measure and compare the effectiveness of each approach. These experiments were designed to answer a key research question:
> How much improvement can be gained from transfer learning (ResNet18) over a simple CNN trained from scratch?

1. **Baseline CNN (EnhancedFontCNN)**
This experiment served as a baseline, allowing us to establish reference metrics for a standard convolutional architecture trained from scratch. It helped verify our data pipeline, preprocessing strategy, and label encoding.

*Architecture*: A custom CNN model with 4 convolutional layers followed by 2 fully connected layers. The architecture includes Batch Normalization, ReLU activations, MaxPooling, and Dropout to improve stability and reduce overfitting. The model was trained on augmented grayscale images resized to 224×224 pixels.

*Evaluation Metrics*:
- **Accuracy**: Measures the overall proportion of correct predictions across all font classes.
- **Macro F1-Score**: Chosen to balance precision and recall across all classes, especially important given the class imbalance in our dataset.
- **Confusion Matrix**: Used for visualizing how well the model differentiates between specific font styles.



2. **ResNet18 Fine-tuning**
This experiment aimed to assess the impact of transfer learning on classification performance, especially on a relatively small and visually complex dataset like ours. The hypothesis was that pretrained features would improve generalization and accelerate training.

*Architecture*: A ResNet18 model pretrained on ImageNet. We froze all layers except the final fully connected layer, which was replaced and retrained for 11 output classes. This allowed us to retain the rich hierarchical features learned from large-scale visual data while adapting the model to our specific font classification task.

*Evaluation Metrics*:
- **Accuracy**: For a consistent performance comparison with the baseline CNN.
- **Macro F1-Score**: Ensured balanced evaluation across classes, highlighting improvements in handling less frequent font styles.
- **Confusion Matrix**: Provided insights into class-specific performance and confusion patterns.


Both experiments were executed under the same conditions:
- Identical data splits and augmentation techniques
- Same loss function (`CrossEntropyLoss`), optimizer (`Adam`), and scheduler (`ReduceLROnPlateau`)
- Fixed seed (`SEED = 42`) for reproducibility

This experimental setup ensured a fair comparison, isolating the effect of the model architecture as the key variable.




---

## Section 4: Results

### Key Findings
- Removing corrupt data improved training
- Preprocessing and augmentation reduced overfitting
- ResNet18 significantly outperformed the baseline

### Results Table

| Model        | Accuracy | Macro F1 | Training Time |
|--------------|----------|----------|---------------|
| Baseline CNN | 63.2%    | 0.61     | ~18 min       |
| ResNet18     | 86.7%    | 0.84     | ~22 min       |

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---





