# The Librarian from Alexandria

## 👥 Team Members - AlexandriaAI
- Denise Di Franza 
- Gabriele De Ieso
- Alessia Tonicello

---

## 🧭 Section 1: Introduction

This project was developed for the Machine Learning course at LUISS. As newly appointed librarians of the Great Library of Alexandria, our mission is to automatically classify ancient digitized texts by their font style, in order to support the digital archiving process. 

We built a deep learning model capable of identifying writing styles from historical scanned documents, dealing with challenges like image noise, font variation, and dataset imbalance. The final objective is to automate the annotation of historical documents based on their font.

---

## 🧪 Section 2: Methods

### Dataset
The dataset consists of over 1,000 scanned pages of ancient texts, each labeled with the font used. The files are stored in a CSV (`pages.csv`) referencing image files in the `img/` directory.

### Preprocessing
We applied several preprocessing steps to improve image quality and prepare data for training:
- Grayscale conversion
- Image resizing (224x224)
- Binarization (Otsu's method)
- Normalization
- Removal of unreadable pages
