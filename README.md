# The Librarian from Alexandria

## 👥 Team Members - AlexandriaAI
- Gabriele De Ieso
- Denise Di Franza 
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
Before training we: 
- Verified image existence by checking paths and filtering out missing or unreadable files.
- Added a binary `exist` column which flagged whether each image file was present and entries with missing images were dropped to avoid issues during training.

&nbsp;

- **Font Label Mapping**
  Since the dataset consists of font names (e.g., "cicero", "vesta") associated with each image, we mapped these strings to integer labels using a Python dictionary. We identified **11 unique fonts**, and assigned values from 0 to 10 for model compatibility:
  ```python
  Font: augustus -> Label: 0
  Font: aureus -> Label: 1
  Font: cicero -> Label: 2
  Font: colosseum -> Label: 3
  Font: consul -> Label: 4
  Font: forum -> Label: 5
  Font: laurel -> Label: 6
  Font: roman -> Label: 7
  Font: senatus -> Label: 8
  Font: trajan -> Label: 9
  Font: vesta -> Label: 10


**1. Preprocessing Strategy**

The scanned pages varied in resolution and clarity, therefore, in order to standardize inputs and reduce noise, ee tested and compared multiple preprocessing pipelines. Our final steps included::
- Grayscale conversion: convert images to grayscale to reduce complexity (from 3 channels to 1), because color isn't needed for font recognition.
- CLAHE (Contrast Enhancement): to improve local contrast in noisy or unevenly lit scans
- Double-page splitting: if an image is too wide (double-page spread), we split it into two pages based on aspect ratio.
- Text patch extraction: using adaptive thresholding and a sliding window to extract 255x255 patches containing text
- Fallback center crop: used when no valid patches were found

&nbsp;


**2.3 Data Augmentation (Training Set Only)**
To prevent overfitting and introduce visual variation, we implemented:
- Random Horizontal Flip (50%)
- Random Affine Transformations (rotation ±10°, translation ±10%, scaling ±10%)
- Random Perspective Distortion (scale = 0.2)
- (Optional) Random Erasing – currently disabled


&nbsp;

**2.4 Tensor Conversion & Normalization**
- ToTensor: converts PIL images to PyTorch tensors
- Normalization: scales pixel values to [-1, 1] using mean=0.5, std=0.5
- Validation set is only resized, normalized and converted to tensor (no augmentation).


&nbsp;


**3. Model Selection**

In order to address the classification of ancient fonts, we explored and compared two different neural network architectures: a custom-built Convolutional Neural Network (CNN) and two pretrained models: ResNet18 and MobileNetV2. With these architectures we progressively improved the performance.

Custom CNN – Baseline Architecture:
We first designed a simple yet effective Convolutional Neural Network to serve as a baseline. This model was trained from scratch and allowed us to test the full data pipeline—including preprocessing and augmentation—under controlled conditions.

*Custom CNN (Baseline)*:
Our starting point was a simple Convolutional Neural Network composed of:
- 4 convolutional blocks, each with Batch Normalization, ReLU activation, MaxPooling, and Dropout
- 2 fully connected layers followed by a Softmax classifier for 11 font classes

This model helped validate our pipeline, but performance plateaued below 50% accuracy.

*EnhancedFontCNN*:
This is a deeper version of the baseline CNN and this architecture increased the model capacity by:
- Adding additional convolutional layers
- Using adaptive average pooling to manage input variations
- Increasing dropout regularization

However, despite the improvements, accuracy still remained moderate.


&nbsp;
To improve the performance, we used pretrained ResNet18 and MobileNetV2 models, both imported from the `torchvision.models` library, which provides pretrained versions of these architectures based on the ImageNet dataset.
&nbsp;
ResNet18 (Transfer Learning):
We fine-tuned a ResNet18 model pretrained on ImageNet:
- All feature extraction layers were frozen
- The final fully connected layer was replaced to output 11 classes
This approach improved performance and required less training time, but results stabilized around ~58% accuracy.

MobileNetV2 (Final Model):
- This lightweight and efficient model provided the best performance:
- Initially trained with a frozen backbone
- Later runs unfreezed the last few layers for partial fine-tuning
- Combined with data augmentation and weighted loss
Results peaked at ~73% accuracy, showing strong generalization and faster convergence, even on limited hardware.

&nbsp;
This technique, known as *transfer learning*, is particularly effective when dealing with small or medium-sized datasets, like ours. ResNet18 and MobileNetV2 were expected to extract more robust visual features and generalize better than the custom CNN.


&nbsp;


**4. Training Setup**

To ensure fair and reproducible training across both models, we adopted a consistent training pipeline, carefully chosen based on best practices in deep learning.

*General Configuration:*
- Framework: PyTorch 2.x
- Device: CUDA-enabled GPU (if available), otherwise CPU
- Seed: A fixed random seed (42) was applied across NumPy, PyTorch, and Python to guarantee reproducible results

&nbsp;

*Optimization Strategy:*
- Loss Function: `CrossEntropyLoss` weighted by inverse class frequencies to address dataset imbalance
- Optimizer: `Adam`
  - Learning rate: `1e-4`
  - Weight decay: `1e-5` (to prevent overfitting)
- Learning Rate Scheduler:
  - `CosineAnnealingLR` for smooth cyclical updates
  - `ReduceLROnPlateau` for reducing learning rate on validation plateaus

&nbsp;

*Training Conditions:*

- Batch Size: 16 (due to limited GPU memory)
- Epochs: Up to 100, with early stopping triggered by validation loss stagnation
- Precision: Mixed precision training using autocast and GradScaler to accelerate training and reduce memory usage

&nbsp;

*Data Splitting:*
- 80% of data used for training
- 20% for validation
- Stratified Shuffle Split: Ensured the distribution of font classes remained balanced in both sets to avoid bias or poor generalization for underrepresented fonts


&nbsp;


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
[ Verify dataset + remove corrupt files ]
   ↓
[ Map font names to integer labels ]
   ↓
[ Split dataset (stratified 80/20) ]
   ↓
[ Preprocessing (CLAHE, patch extraction, grayscale) ]
   ↓
[ Data Augmentation (train set only) ]
   ↓
[ Apply transforms + create datasets ]
   ↓
[ Train CNN / ResNet / MobileNetV2 ]
   ↓
[ Evaluate on validation set (metrics + confusion matrix) ]
   ↓
[ Save best model and predictions ]
```



---

## Section 3: Experimental Design

With the models defined and the training setup in place, we conducted two core experiments to measure and compare the effectiveness of each approach. These experiments were designed to answer a key research question:
> How much improvement can be gained from transfer learning (ResNet18) over a simple CNN trained from scratch?

&nbsp;

1. **Baseline CNN (EnhancedFontCNN)**
This experiment served as a baseline, allowing us to establish reference metrics for a standard convolutional architecture trained from scratch. It helped verify our data pipeline, preprocessing strategy, and label encoding.

*Architecture*: A custom CNN model with 4 convolutional layers followed by 2 fully connected layers. The architecture includes Batch Normalization, ReLU activations, MaxPooling, and Dropout to improve stability and reduce overfitting. The model was trained on augmented grayscale images resized to 224×224 pixels.

&nbsp;

*Evaluation Metrics*:
- **Accuracy**: Measures the overall proportion of correct predictions across all font classes.
- **Macro F1-Score**: Chosen to balance precision and recall across all classes, especially important given the class imbalance in our dataset.
- **Confusion Matrix**: Used for visualizing how well the model differentiates between specific font styles.


&nbsp;


2. **ResNet18 Fine-tuning**
This experiment aimed to assess the impact of transfer learning on classification performance, especially on a relatively small and visually complex dataset like ours. The hypothesis was that pretrained features would improve generalization and accelerate training.

*Architecture*: A ResNet18 model pretrained on ImageNet. We froze all layers except the final fully connected layer, which was replaced and retrained for 11 output classes. This allowed us to retain the rich hierarchical features learned from large-scale visual data while adapting the model to our specific font classification task.
&nbsp;

*Evaluation Metrics*:
- **Accuracy**: For a consistent performance comparison with the baseline CNN.
- **Macro F1-Score**: Ensured balanced evaluation across classes, highlighting improvements in handling less frequent font styles.
- **Confusion Matrix**: Provided insights into class-specific performance and confusion patterns.


&nbsp;

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





