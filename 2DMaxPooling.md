#### 2D Max Pooling

Implement a 2D max pooling operation for image/feature map downsampling. The program should take an input tensor and produce an output tensor by applying max pooling with specified kernel size, stride, and padding.

<img width="745" height="205" alt="image" src="https://github.com/user-attachments/assets/43f70df1-644f-4d7a-aeac-cdab3b0c622d" />

#### Implementation Requirements
 - External libraries are not permitted
 - The `solve` function signature must remain unchanged
 - The final result must be stored in tensor `output`

### Max Pooling Operation

For each output position (n, c, h_out, w_out), compute the maximum value over the corresponding input window:
`output[n, c, h_out, w_out] = max(input[n, c, h:h+kernel_size, w:w+kernel_size])`
where `h = h_out * stride and w = w_out * stride`

### Example 1:
```
Input:  input = [[[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]]]
        kernel_size = 2
        stride = 1
        padding = 0
Output: output = [[[[5.0, 6.0],
                    [8.0, 9.0]]]]
```
### Example 2:
```
Input:  input = [[[[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0],
                   [21.0, 22.0, 23.0, 24.0, 25.0]]]]
        kernel_size = 3
        stride = 1
        padding = 1
Output: output = [[[[7.0, 8.0, 9.0, 10.0, 10.0],
                    [12.0, 13.0, 14.0, 15.0, 15.0],
                    [17.0, 18.0, 19.0, 20.0, 20.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0]]]]
```
### Constraints
 - 1 ≤ `N` ≤ 100 (batch size)
 - 1 ≤ `C` ≤ 512 (channels)
 - 1 ≤ `H`, `W` ≤ 1024 (height, width)
 - 1 ≤ `kernel_size` ≤ 16
 - 1 ≤ `stride` ≤ 16
 - 0 ≤ `padding` ≤ 16
Input and output tensors use float32 precision
Performance is measured with `N` = 4, `kernel_size` = 3, `stride` = 2
