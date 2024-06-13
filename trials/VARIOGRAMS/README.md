## Fractal Dimension Estimation Method using Variogram

The Variogram method is a Fractal Dimension (FD) estimation technique that quantifies differences in pixel values between pairs of samples with specific relative orientations. The mathematical formulation of the Variogram algorithm is defined as follows:
<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/ee8db218-9ce7-4625-90af-841e9df717f3">
</p>

**Equation (1) and (2):**
- **Equation (1):** Defines the relationship for computing variogram y(h) where h is the lag step (used values: h = 1, 2, ..., 6).
- **Equation (2):** Calculates the Fractal Dimension (FD) using the obtained slope from a linear regression between log[h] and log[y(h)].

**Steps to Estimate FD:**
1. Choose feature extraction directions (e.g., vertical, horizontal, diagonal).
2. Compute differences in pixel values for each direction.
3. Use Equation (1) to compute variogram y(h) for h = 1 to 6.
4. Perform linear regression between log[h] and log[y(h)] to obtain the slope.
5. Compute FD using Equation (2) with the obtained slope.
6. Incorporate the intercept of the linear regression as a secondary feature.
7. Compute FD for each directional dimension, resulting in a 6-dimensional feature vector.
8. Moreover we used the intercept of the linear regression as second feature, whereas 3 directional FD were computed our feature vector is 6D.


### Implementation:

### `getSigma(image, h, direction)`
Calculates the variance (sigma) of differences in pixel values for a given direction and lag distance (h).
Direction of feature extraction:
- 0: Horizontal
- 1: Vertical
- 2: Diagonal
#### Parameters:
```
The input image as a 2D numpy array (grayscale).
h: int
Lag distance.
direction: int
```

#### Returns:
```
Variance (sigma) of differences in pixel values for the specified direction and lag distance.
```
`VARIOGRAM(h, sigma)`
#### Parameters:
```
h: list
    List of lag distances.
sigma: list
    List of variance (sigma) values corresponding to each lag distance in h.
```

#### Returns:
```
list
    List containing the Fractal Dimension (FD) and intercept (c) of the linear regression.

```
`extract_features(image)`
#### Parameters:
```
image: numpy array
    The input grayscale image.
```

#### Returns:
```
llist
    List containing the extracted Fractal Dimensions and intercepts for all three directions (horizontal, vertical, diagonal).

```
### Performance Analysis 

After conducting experiments with KNN using voting over a group of image syllables, it attained an accuracy of `81.4%` on the validation data. This indicates promising potential, suggesting that further refinement and exploration with different classifiers could potentially yield even better results.
