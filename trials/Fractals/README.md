## Box Counting Dimension (BCD)
<p align="center">
  <img src="https://github.com/Omar-Said-4/Arabic_OFR/assets/87082462/f17a76eb-df50-4055-89d7-2f372d5cb3c4">
</p>
The Box Counting Dimension (BCD) is one of the most widely used fractal dimensions. Its popularity is largely due to its relatively simple mathematical calculation and estimation process.

To estimate the BCD, follow these steps:

1. **Divide the Text Block**: According to equation (1), divide the text block into a grid of boxes of size `r`.
2. **Count Non-Empty Boxes**: Count the number of boxes that are not empty.
3. **Repeat for Different `r`**: Repeat the above steps for different values of `r`.
4. **Plot the Graph**: Produce a graph of `log N(r)` versus `log(1/r)` for each size.
5. **Estimate BCD**: Estimate the BCD by performing linear regression between `log N(r)` and `log(1/r)`, as described in equation (1):

`BCD(r) = lim_{r to 0} {log N(r)}/{log(1/r)}`

### Implementation:
#### `box_count(image, box_size)`
```
DESC: Counts the number of non-empty boxes of a specific size covering the image.

```
```
  Args:
    image: A 2D numpy array representing the binary image.
    box_size: The size of the square boxes used for counting.
```
```
  Returns:
    The number of non-empty boxes.
```

## Performance Analysis Module

After conducting experiments with KNN, it attained an accuracy of `45.4%` on the validation data, indicating that the current set of features may not be optimal.
