## Hu Moments for Image Analysis

Hu Moments are a set of seven numbers that describe the shape of an image invariant to transformations such as translation, scale, and rotation. Here's how to calculate Hu Moments using MathJax notation:

1. **Calculate the central moments** ( ${\mu_{pq}}$ ):

$$
\mu_{pq} = \sum_{x,y} (x - \bar{x})^p (y - \bar{y})^q I(x, y)
$$

where \( \bar{x} \) and \( \bar{y} \) are the centroid coordinates of the image, and \( I(x, y) \) is the intensity of the pixel at position \( (x, y) \).

2. **Normalize the central moments**:

Normalize the central moments to achieve scale invariance:

$$
\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{(1 + \frac{p+q}{2})}}
$$

where \( ${\mu_{00}}$ ) is the zeroth central moment (total intensity).

3. **Construct the covariance matrix**:

Formulate the covariance matrix using the normalized central moments:

$$
\mathbf{C} = \begin{bmatrix}
\eta_{20} + \eta_{02} & (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2 \\
(\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2 & (\eta_{30} + \eta_{12})^2 + (\eta_{21} + \eta_{03})^2
\end{bmatrix}
$$


4. **Calculate Hu Moments**:

Derive the seven Hu Moments \${\lambda_i}$ from the covariance matrix \mathbf{C}:

- \${\lambda_1}$ = (\${\eta_{20}}$ + \${\eta_{02}}$)
- \${\lambda_2}$ = (\${\eta_{20}}$ - \${\eta_{02}}$)^2 + 4(\${\eta_{11}}$)^2)
- \${\lambda_3}$ = (\${\eta_{30}}$ - 3(\${\eta_{12}}$)^2 + (3(\${\eta_{21}}$) - \${\eta_{03}}$)^2 \)
- \${\lambda_4}$ = (\${\eta_{03}}$ + \${\eta_{12}}$)^2 + (\${\eta_{21}}$ + \${\eta_{03}}$)^2 \)
- \${\lambda_5}$ = (\${\eta_{03}}$ - 3(\${\eta_{12}}$)(\${\eta_{30}}$ + \${\eta_{12}}$)[(\${\eta_{30}}$ + \${\eta_{12}}$)^2 - 3(\${\eta_{21}}$ + \${\eta_{03}}$)^2] + (3(\${\eta_{21}}$) - \${\eta_{03}}$)(\${\eta_{21}}$ + \${\eta_{03}}$)[3(\${\eta_{30}}$ + \${\eta_{12}}$)^2 - (\${\eta_{21}}$ + \${\eta_{03}}$)^2]
- \${\lambda_6}$ = (\${\eta_{20}}$ - \${\eta_{02}}$)[(\${\eta_{30}}$ + \${\eta_{12}}$)^2 - (\${\eta_{21}}$ + \${\eta_{03}}$)^2] + 4(\${\eta_{11}}$)(\${\eta_{30}}$ + \${\eta_{12}}$)(\${\eta_{21}}$ + \${\eta_{03}}$)
- \${\lambda_7}$ = (3(\${\eta_{21}}$) - \${\eta_{03}}$)(\${\eta_{30}}$ + \${\eta_{12}}$)[(\${\eta_{30}}$ + \${\eta_{12}}$)^2 - 3(\${\eta_{21}}$ + \${\eta_{03}}$)^2] - (\${\eta_{30}}$ - 3(\${\eta_{12}}$))(\${\eta_{21}}$ + \${\eta_{03}}$)[3(\${\eta_{30}}$ + \${\eta_{12}}$)^2 - (\${\eta_{21}}$ + \${\eta_{03}}$)^2] \)

5. **Apply invariance**:

   Use the Hu Moments \( ${\lambda_1}$ ) to \( ${\lambda_7}$ ) to describe and compare shapes of objects in images regardless of their orientation or size.


### Implementation:

#### 1. `cv2.moments(im)`
Computes the moments of a grayscale image `im`. Moments are a set of statistical measures that describe the shape and spatial distribution of intensity in an image.

#### 2. `cv2.HuMoments(moments)`
Calculates the Hu Moments from the moments computed in the previous step. Hu Moments are a set of seven numbers that are invariant to image transformations such as rotation, scale, and translation. They are used for shape recognition and image analysis.

#### 3. `np.sign()` and `np.log10()`
These are NumPy functions used to process the Hu Moments:
- `np.sign(huMoments)`: Computes the sign of each element in `huMoments`.
- `np.abs(huMoments + 0.001)`: Adds a small value (0.001) to `huMoments` and computes its absolute value to avoid issues with zero values.
- `np.log10()`: Computes the base-10 logarithm of the absolute values obtained from the previous step.

The final result, `huMoments`, represents the processed Hu Moments, which are used in various applications such as image recognition and pattern analysis.

```python
moments = cv2.moments(im)
huMoments = cv2.HuMoments(moments)
huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments + 0.001))


### Performance Analysis 

After conducting experiments with KNN using voting over a group of image syllables, it attained an accuracy of `58%` on the validation data. This indicates it might mot be a suitable set of features.


