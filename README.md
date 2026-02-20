# ✍️ MNIST Handwritten Digit Classification

**Executive Summary:**
In this computer vision project, I built a multiclass Machine Learning classifier to accurately identify handwritten digits (0-9). By combining Data Augmentation techniques with Principal Component Analysis (PCA) for dimensionality reduction, I successfully optimized a K-Nearest Neighbors (KNN) model to achieve a test accuracy of over 97%, balancing high predictive performance with computational efficiency.

---

## 1. The Technical Problem

Image classification is a foundational task in Machine Learning. The challenge lies in translating raw pixel intensities into structured patterns that a model can generalize. 


**Objective:** Develop a model capable of correctly classifying 28x28 pixel grayscale images of handwritten digits.
* **Problem Type:** Supervised Learning (Multiclass Classification)
* **Target Metric:** Accuracy (> 97%)
* **Dataset:** MNIST (70,000 images, split into 60k for training and 10k for testing).

## 2. Data Preprocessing & Feature Engineering

Before training, the dataset was explored using **t-SNE** to visualize the multidimensional clusters of each digit.  To prepare the data and improve the model's robustness, I implemented the following pipeline:

* **Pixel Scaling:** Applied `MinMaxScaler` to normalize pixel intensities, ensuring distance-based algorithms like KNN perform optimally.
* **Data Augmentation:** To make the model shift-invariant and improve generalization, I synthetically expanded the training set by shifting each image by 1 pixel in various directions using `scipy.ndimage.shift`.
* **Dimensionality Reduction (PCA):** Data augmentation significantly increased the computational load. To solve this, I applied PCA to compress the feature space (from 784 pixels down to a smaller subset of principal components), preserving **95% of the variance** while drastically speeding up training and inference times.

## 3. Machine Learning Models & Tech Stack

Several algorithms were evaluated using Cross-Validation, including Logistic Regression (One-Vs-Rest), Stochastic Gradient Descent (SGD), and Support Vector Machines (SVM). 

While SVM showed strong initial accuracy, its computational cost was prohibitive. **K-Nearest Neighbors (KNN)** was selected as the optimal model due to its excellent balance between complexity and performance. The model was fine-tuned using `RandomizedSearchCV` and `GridSearchCV` to optimize the number of neighbors and weight functions.

🛠️ **Tech Stack:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)

> 💡 **Detailed EDA (including t-SNE visualizations), Error Analysis with Confusion Matrices, and the final test metrics are documented directly inside the Jupyter Notebook.**
