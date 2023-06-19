# Principal Component Analysis (PCA) from Scratch

This repository contains a Python implementation of Principal Component Analysis (PCA) from scratch. PCA is a widely used dimensionality reduction technique that helps uncover the underlying structure and patterns in high-dimensional data.

## Contents

- [Background](#background)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Results](#results)

# Background


This project focuses on performing Principal Component Analysis (PCA) from scratch on the 130 Hospitals Diabetic Dataset from the US. PCA is a widely used dimensionality reduction technique that helps uncover the underlying structure and patterns in high-dimensional data.

In this project, we aim to explore the applicability of PCA on healthcare data by applying it to the 130 Hospitals Diabetic Dataset. This dataset contains comprehensive information about diabetic patients, including patient demographics, medical history, treatment details, and hospital-specific data. By implementing PCA from scratch, we gain a deeper understanding of the mathematical principles involved in the PCA algorithm and how it can be applied to real-world healthcare datasets.

The dataset consists of a large number of features, making it a suitable candidate for dimensionality reduction techniques. By leveraging PCA, we aim to identify the most important features that contribute to the overall variance in the dataset. This process allows us to effectively reduce the dimensionality of the dataset while retaining the most relevant information.

Through this project, we aim to showcase the practical application of PCA in the healthcare domain and provide insights into the potential benefits of using dimensionality reduction techniques for analyzing and understanding complex healthcare datasets.

Feel free to customize and enhance the background section further based on your project's specific goals and context!


## Features

- Perform PCA on your datasets without relying on external libraries
- Gain a deeper understanding of the mathematics behind PCA and its implementation
- Customize the PCA algorithm based on your specific needs
- Efficiently calculate principal components and transform data into a lower-dimensional space.


## Dataset

The dataset used in this project is the 130 Hospitals Diabetic Dataset from the US. It contains information on diabetic patients, including patient demographics, medical history, and treatment details. The dataset provides a rich source of information for clustering analysis and algorithm comparison.


## Implementation

The Python code in this repository is organized as follows:

- `Jash_Shah_Data_Mining_Assignment_05_Question_01.py`: Contains the implementation of the PCAl from scratch.Also has a detailed overview with comments about comparison of different models.


## Results

![image](https://github.com/jashshah-dev/Principal-Component-Analysis-from-Scratch/assets/132673402/c192684a-ae39-437e-aba0-1fccdd9d2ccf)
![image](https://github.com/jashshah-dev/Principal-Component-Analysis-from-Scratch/assets/132673402/12b41102-fc17-4fa9-8eaa-48431abe0674)
![image](https://github.com/jashshah-dev/Principal-Component-Analysis-from-Scratch/assets/132673402/ca6e58f5-2d7f-401a-bbc8-dba067e145b4)

• Have ran K means algorithm on dataset after applying PCA
• The number of columns after performing dimensionality reduction are 373
• Have plotted graphs for Before and After PCA and we can infer.
• Within Sum of Square Errors
• The WSS reduced after PCA and can be shown from the box plot
• The median of WSS is lower for Clustering after PCA
• Silheoutte Analysis
• Silhouette analysis is a method used to evaluate the quality of clustering results by measuring how
well each data point fits within its assigned cluster. The analysis provides a silhouette score for each
data point, which indicates the degree of similarity of that point to its own cluster compared to other
clusters.
• The silhouette score ranges from -1 to 1, where a score of -1 indicates that the data point is assigned
to the wrong cluster, a score of 0 indicates that the data point is near the decision boundary between
two clusters, and a score of 1 indicates that the data point is well- clustered and belongs to its own
cluster.
• We can infer there is a slight decrease in the silheoutte Score which indicates there might be Loss of
Information after performing PCA
• We can increase the covered Variance to 97 percent or more and increase the number of PC’s to increase
the silheoutte score
• Run Time From the graphs it’s clearly visible the Run time decreases after applying PCA because
the number of dimensions reduce.






