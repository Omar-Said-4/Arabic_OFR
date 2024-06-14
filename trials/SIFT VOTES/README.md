## 1)Pipeline


<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/1f03e7d0-cbd4-4fd3-b602-070ab61884e9">
</p>


## 2) Preprocessing Module
<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/d8a146ef-18f2-4a6f-a377-6b4359cd4144">
</p>




### Overview

The `SIFT` (Scale-Invariant Feature Transform) is a feature extraction technique used in computer vision to detect and describe distinctive features in grayscale images. It operates by identifying keypoints at various scales and orientations, which are then described using local gradients and orientations. SIFT descriptors are invariant to scale and rotation, making them valuable for tasks like object recognition, image stitching, and augmented reality.

## 3) Feature Extraction/Selection Module 

<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/dc4ea69d-aa9f-4761-8780-1334c9a3d08a">
</p>

### Implementation


### `create_codebook(features, num_clusters, batch_size)`
#### Parameters:

```
features: Array of SIFT descriptors.
num_clusters: Number of clusters for the codebook.
batch_size: Size of mini-batches used in clustering.

```
#### Returns:
```
A MiniBatchKMeans clustering object fitted to the SIFT features.

```
#### `extract_sift_features(image)`

Extracts SIFT (Scale-Invariant Feature Transform) features from a given image.

#### Parameters:
```
image: The input grayscale image array.
```
#### Returns:
```
Descriptors representing keypoint features in the image.
```
`prepare_data(images)`
Prepares SIFT features for a list of images, filtering out non-detectable features.

#### Parameters:
```
images: List of input grayscale images.
```
#### Returns:
```
A numpy array containing SIFT descriptors for all images.
```
`compute_bovw_representation(features, codebook, num_clusters=100)`
#### Parameters:
```
features: Array of SIFT descriptors.
codebook: Pre-trained codebook for clustering.
num_clusters: Number of clusters in the codebook (default: 100).
```
#### Returns:
```
BoVW representation as a normalized histogram.
```

## 4) Model Selection/Training Module
![image](https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/356b58fc-f946-40e6-b0b5-041443804455)


## 5) Performance Analysis Module


After conducting experiments with SVM, KNN, Random Forest, Decision Tree, and AdaBoost, our analysis revealed that the SVM model achieved the best performance. It attained an accuracy of 99.1% on the test data. using voting over a group of image syllables.






