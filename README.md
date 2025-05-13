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

We experimented with two architectures:
- Baseline CNN: a custom 3-layer convolutional model trained from scratch.
- ResNet18: a pretrained convolutional neural network from the torchvision library, fine-tuned for our 11-class problem.
The final choice of ResNet18 was based on its ability to extract deeper hierarchical features from images, speeding up convergence and improving generalization on our relatively small dataset.



**4. Training Setup**
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate 1e-4, weight decay 1e-5)
- Scheduler: ReduceLROnPlateau
- Batch size: 32
- Early stopping: based on validation loss
- Data split: Stratified 80/20 for training and validation



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
[ Check missing/corrupt images ]
   ↓
[ Font label mapping (string → int) ]
   ↓
[ Preprocessing: grayscale, resize, binarize, normalize ]
   ↓
[ Data augmentation ]
   ↓
[ Train model: CNN or ResNet18 ]
   ↓
[ Evaluate on validation set ]
   ↓
[ Save best model + confusion matrix + metrics ]
```

---

## Section 3: Experimental Design

We conducted two core experiments to evaluate and compare different model approaches:

1. **Baseline CNN (EnhancedFontCNN)**  
   - Purpose: Establish a solid benchmark using a custom-built CNN.  
   - Architecture: 4 convolutional layers with batch normalization, ReLU activations, max pooling, and dropout. The final classifier has two fully connected layers.
   - Training: Trained from scratch using grayscale 224×224 images.
   - Metrics: Accuracy, Macro F1-score, Confusion Matrix.

2. **ResNet18 Fine-tuning**  
   - Purpose: Improve generalization using a pretrained model (transfer learning).
   - Compared with baseline.
   - Training: Only the final layer was trained on our dataset, making it faster and more robust to overfitting.
   - Metrics: Accuracy, Macro F1-score, Confusion Matrix.

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





