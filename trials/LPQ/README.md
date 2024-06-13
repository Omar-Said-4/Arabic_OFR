## 1)Pipeline


<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/1f03e7d0-cbd4-4fd3-b602-070ab61884e9">
</p>


## 2) Preprocessing Module
<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/10045550-428f-40d7-b8af-c19b3398983d">
</p>


## 3) Feature Extraction/Selection Module `(Local Phase Quantization (LPQ) Feature Extraction Module)`


![image](https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/6c68a18e-e4b3-429f-95bd-47734b1596be)


### Overview

The `LPQ` Feature Extraction Module analyzes the texture of grayscale images. It uses the Local Phase Quantization method to generate a feature
vector that describes the unique patterns and textures within an image. This vector is then used to train our model.

### Implementation

#### `validate_image(img)`

Validates that the input is a grayscale image.

#### Parameters:

```
img: The input image array.
```
#### Returns:

```
The image converted to a 64-bit floating-point format if valid.
```
#### Raises:

```
ValueError: If the input image is not a grayscale image.
```
#### `create_filters(win_size)`

Creates a set of filters for the LPQ method.

#### Parameters:

```
win_size: The window size for the filters.
```
#### Returns:

```
A tuple of three filters (w0, w1, w2) used for the LPQ method.
```
#### Filters:

```
Uniform Filter (w0): This filter is a simple averaging filter. It's represented by an array of ones which means it treats all parts of the image
equally averaging the pixel values.
Complex Exponential Filter (w1): This filter captures the 'phase' information in the image, which relates to the position of the image's
patterns and edges.
Conjugate Complex Exponential Filter (w2): The third filter is the complex conjugate of the second filter (w1).
```

#### `apply_filters(img, filters, conv_mode='valid')`

Applies the LPQ filters to the image and returns the frequency response.

#### Parameters:

```
img: The input image array.
filters: A tuple containing the LPQ filters.
conv_mode: The mode of convolution ('valid' by default).
```
#### Returns:

```
A stack of the real and imaginary parts of the frequency responses.
```
#### Filter Pairs:

```
Combination (w0, w1): This pair works together to find patterns that are spread out in one direction.
Combination (w1, w0): This pair does the opposite, looking for patterns spread out in the other direction.
Combination (w1, w1): This pair detects diagonal patterns.
Combination (w1, w2): This pair is like a cross-check.
```
#### `compute_codewords(freq_resp)`

Computes the LPQ codewords from the frequency response obtained from the filters. It converts the frequency response into a binary code by
checking if each response is greater than zero and then sums these binary codes weighted by powers of two.

#### Parameters:

```
freq_resp: The frequency response array obtained from apply_filters.
```
#### Returns:

```
An array of LPQ codewords.
```
#### `lpq(img)`

The main function that applies the LPQ method to an input image and returns a normalized histogram as a feature vector.

#### Parameters:

```
img: The input image array.
```
#### Returns:

```
A normalized histogram representing the LPQ feature vector.
```
## 4) Model Selection/Training Module
![image](https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/356b58fc-f946-40e6-b0b5-041443804455)


## 5) Performance Analysis Module


After conducting experiments with SVM, KNN, Random Forest, Decision Tree, and AdaBoost, our analysis revealed that the SVM model achieved the best performance. 
It attained an accuracy of `97.5%` on the test data and completed the evaluation in under `35` seconds on `1000` samples.





