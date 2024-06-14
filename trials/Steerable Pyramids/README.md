# Steerable Pyramids

Steerable pyramids are a versatile image decomposition technique used in image processing and computer vision. They offer a hierarchical representation of images, capturing both spatial structures and orientation information effectively.

## Key Features

- **Multi-Scale Representation:** Decomposes images into multiple scales, enabling detailed feature extraction at different levels.
  
- **Multi-Orientation Analysis:** Analyzes image features by orientation as well as scale, beneficial for tasks like edge detection and texture analysis.
  
- **Rotation-Invariance:** Maintains robustness against image rotations, making them suitable for applications requiring orientation-invariant features.

## Implementation

#### To create a Gaussian kernel and its first derivatives for image processing:

```python
m = 4  # 1-sided filter size
x_index, y_index = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
x_index = x_index.reshape((-1, 1))
y_index = y_index.reshape((-1, 1))

G = lambda x, y: np.exp(-(x**2 + y**2))
G0 = lambda x, y: -2 * x * G(x, y)
G90 = lambda x, y: 2 * y * G(x, y)

# Gaussian Kernel
gk = G(x_index, y_index).reshape((m * 2 + 1, m * 2 + 1))

# Gaussian 1st Derivative with 0-degree orientation
gk0 = G0(x_index, y_index).reshape((m * 2 + 1, m * 2 + 1))

# Gaussian 1st Derivative with 90-degree orientation
gk90 = G90(x_index, y_index).reshape((m * 2 + 1, m * 2 + 1))
```
#### Define angles for steering filters
```python
steering_angles = np.arange(0, 180, 30)
print(steering_angles)  # [  0  30  60  90 120 150]

# Initialize results array (assuming levels[0] is your image)
results = np.zeros((levels[0].shape[0], levels[0].shape[1], len(steering_angles)))

# Generate steerable filters for each angle
gk_theta = []
for angle in steering_angles:
    gk_theta.append(np.cos(np.radians(angle)) * gk0 + np.sin(np.radians(angle)) * gk90)

# Apply steerable filters to the image
R_Theta = []
for g_theta in gk_theta:
    # Assuming cv2 is used for filtering
    R_Theta.append(cv2.filter2D(levels[0], ddepth=-1, kernel=g_theta))
```
