## Local Binary Patterns (LBP) Feature Extraction

The Local Binary Patterns (LBP) feature extraction method analyzes grayscale images by capturing local texture patterns. It works by comparing each pixel with its surrounding neighbors. Here's how it works:

1. **Local Neighborhood Definition**: For each pixel in the image, a local neighborhood is defined around it.

2. **Binary Comparison**: Compare the intensity of the central pixel with its neighbors. Each neighbor's intensity value is compared with the central pixel's intensity:
   - If the neighbor's intensity is greater than or equal to the central pixel's intensity, assign a binary value of `1`.
   - If the neighbor's intensity is less than the central pixel's intensity, assign a binary value of `0`.

3. **Pattern Encoding**: Encode these binary comparisons into a local binary pattern (LBP) code, typically represented as a binary number (e.g., `11001001`).

4. **Histogram Calculation**: Calculate a histogram of these LBP codes across the image. Each bin in the histogram represents a unique pattern or texture present in the image.

5. **Feature Vector**: The histogram of LBP codes serves as a feature vector that describes the texture and patterns within the image.

<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/c5f79eea-10d3-4ce5-ada3-81264ea1b624">
</p>

### Implementation:

### `lbp_calculated_pixel(img, x, y)`
Calculates the LBP values for a pixel neighborhood in an image.
#### Parameters:

```
img: numpy array
    The input grayscale image.
x, y: numpy arrays
    Arrays containing the x and y coordinates of the pixel neighborhoods.
```
#### Returns:

```
numpy array
    LBP values computed for the specified pixel neighborhoods.
```




#### `lbp_feature_vector(img)`

Calculates the Local Binary Patterns (LBP) feature vector for a grayscale image.

#### Parameters:

```
img: numpy array
    The input grayscale image.
```
#### Returns:

```
numpy array
    Flattened LBP feature vector representing the texture of the image.
```


#### `prepare_data(images)`

Prepares LBP feature vectors for a list of images.


#### Parameters:

```
images: list of numpy arrays
    List of input grayscale images.

```
#### Returns:

```
list of numpy arrays
    List of flattened LBP feature vectors for each input image.

```

### Performance Analysis 

After conducting experiments with KNN using voting over a group of image syllables, it attained an accuracy of `79.9%` on the validation data. This indicates promising potential, suggesting that further refinement and exploration with different classifiers could potentially yield even better results.

