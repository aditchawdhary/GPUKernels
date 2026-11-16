## Integral Image

Given a `2D` image of shape `H × W` containing 32-bit floating point values, compute its integral image (also known as a summed area table). 
The value at each output position `output[i][j]` is the sum of all input values in the rectangle from the top-left corner `(0, 0)` to position `(i, j)`, inclusive. All values are stored in row-major order.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in output
 - Read exclusively from input and write exclusively to output (do not update input)

## Example
<img width="1110" height="324" alt="image" src="https://github.com/user-attachments/assets/d91b75c3-c632-41b1-a873-91180917907c" />

## Constraints
 - `1` ≤ `H`, `W` ≤ `16,384`
 - Input values are in the range `[-100, 100]`
 - All values are `32-bit` floats
 - Performance is measured with `H = 8,192`, `W = 8,192`
