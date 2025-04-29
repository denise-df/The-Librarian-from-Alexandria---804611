# The Librarian from Alexandria

## 👥 Team Members - AlexandriaAI
- Denise Di Franza 
- Gabriele De Ieso
- Alessia Tonicello

---

## Section 1: Introduction

This project was developed for the Machine Learning course at LUISS. As newly appointed librarians of the Great Library of Alexandria, our mission is to automatically classify ancient digitized texts by their font style, in order to support the digital archiving process. 

We built a deep learning model capable of identifying writing styles from historical scanned documents, dealing with challenges like image noise, font variation, and dataset imbalance. The final objective is to automate the annotation of historical documents based on their font.

---

## Section 2: Methods

### Dataset
The dataset consists of over 1,000 scanned pages of ancient texts, each labeled with the font used. The files are stored in a CSV (`pages.csv`) referencing image files in the `img/` directory.

### 2.1 Design Choices and Key Ideas
- **Data Quality First**  
  Our pipeline begins with a check for missing or unreadable images. We created a new `exist` column in the dataset to verify the presence of each file. This allowed us to safely drop corrupt samples, ensuring we trained only on valid data.
  
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

**Preprocessing Strategy**
The scanned pages varied in resolution and clarity. To standardize inputs and reduce noise, we applied:
- Grayscale conversion: reduced complexity from 3 channels to 1.
- Resizing to 224x224: compatible with standard CNN input dimensions.
- Binarization with Otsu’s method: improved text/background separation.
- Normalization: scaled pixel values to [0, 1].
- Filtering: removed blank or unreadable post-binarized images.


**Data Augmentation**
To prevent overfitting and introduce visual variation (especially in a limited dataset), we implemented:
- Random rotations (±15°) to simulate scanning misalignment.
- Color jitter to mimic lighting conditions.
- Perspective transformations to replicate page distortion.


**Model Selection**
We experimented with two architectures:
- Baseline CNN: a custom 3-layer convolutional model trained from scratch.
- ResNet18: a pretrained convolutional neural network from the torchvision library, fine-tuned for our 11-class problem.
The final choice of ResNet18 was based on its ability to extract deeper hierarchical features from images, speeding up convergence and improving generalization on our relatively small dataset.


**Training Setup**
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate 1e-4, weight decay 1e-5)
- Scheduler: ReduceLROnPlateau
- Batch size: 32
- Early stopping: based on validation loss
- Data split: Stratified 80/20 for training and validation


### 2.2 Environment Reproducibility
We used Python 3.12.6 and a `venv` virtual environment to manage dependencies for this project. All required libraries are listed in the `requirements.txt` file included in the repository.

To recreate our environment and run the project:
