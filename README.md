# I. Project Title

**Improving Classification Accuracy on Long-Tail Distributions**

---

# II. Project Introduction

### Objective
The main objective of this project is to **improve classification accuracy** on datasets with a **long-tail distribution**. A long-tail distribution refers to a scenario where a few classes have many samples, while the majority of classes have only a few. This imbalance causes machine learning models to be overly optimized for the head classes with more samples, resulting in poor performance for tail classes with fewer samples.

### Motivation
Many real-world datasets exhibit **biases in data distribution**, which negatively affect model performance. For example, in image classification tasks, datasets often contain a large number of images for some classes (head classes) and very few images for others (tail classes). This project aims to address this challenge by focusing on long-tail distributions and developing strategies to improve model accuracy across all classes, including the under-represented ones.

---

# III. Dataset Description

In this project, we use long-tail versions of popular datasets: **CIFAR-100-LT** and **Places-LT**. These datasets have been specifically modified to reflect a long-tail distribution, where certain classes have significantly more samples than others.

We conduct experiments on four long-tail datasets, including:
- **ImageNet-LT** (Liu et al., 2019): This dataset contains 115.8K images across 1000 classes, with the number of images per class ranging from a maximum of 1280 to a minimum of 5.
- **Places-LT** (Liu et al., 2019): Comprising 62.5K images from 365 classes, with class sizes ranging from a maximum of 4980 to a minimum of 5 images.
- **iNaturalist 2018** (Van Horn et al., 2018): This dataset includes 437.5K images spread across 8142 species, with the number of images per species ranging from 2 to 1000.
- **CIFAR-100-LT** (Cao et al., 2019): A long-tail version of the CIFAR-100 dataset with various imbalance ratios, such as 100, 50, and 10.

In addition to measuring overall accuracy, we follow the evaluation protocol introduced by Liu et al. (2019) to report accuracy across three subsets of classes:
- **Head classes**: Classes with more than 100 images.
- **Medium classes**: Classes with 20 to 100 images.
- **Tail classes**: Classes with fewer than 20 images.

### Dataset Splitting
The dataset will be split into three subsets:
1. **Training Dataset**: This subset will consist of input data and ground truth labels. It will be used by participants to train their models.
2. **Validation Dataset**: Similar to the training dataset, this will include both input data and ground truth labels, allowing participants to verify and fine-tune their models.
3. **Test Dataset**: The test dataset will contain only input data, with no ground truth labels provided. Participants will submit their predicted outputs for this dataset, and we will evaluate the models based on these predictions.

By evaluating models across these splits, we can assess how well they generalize to unseen data, especially in the context of long-tail distributions.

It’s important to note that in CIFAR-100, the validation set is the same as the test set. Additionally, for long-tail datasets, validation sets are generally not used because the tail classes have too few data points for effective validation.


### Dataset Download

While there are many other long-tail distribution datasets available, we chose this well-known benchmark because downloading and training on other datasets can be time-consuming
1. **CIFAR-100-LT**:  
   The CIFAR-100-LT dataset will be automatically downloaded when you run the provided code. No additional steps are required for this dataset.
