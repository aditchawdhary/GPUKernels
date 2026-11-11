### Ordinary Least Squares

Solve the Ordinary Least Squares (OLS) regression problem on a GPU. Given a feature matrix `X` of `size n_samples x n_features` and a target vector `y` of size `n_samples`, 
compute the coefficient vector `beta` that minimizes the sum of squared residuals:
The closed-form solution to OLS is:

<img width="1199" height="59" alt="image" src="https://github.com/user-attachments/assets/89deb946-2c9d-4040-b291-3ca837e724bb" />
The closed-form solution to OLS is:
<img width="1191" height="60" alt="image" src="https://github.com/user-attachments/assets/9fda0cfc-1e94-4d85-a6f0-a7cb44f053dd" />


#### Implementation Requirements
 - External libraries are not permitted.
 - The `solve` function signature must remain unchanged.
 - The final coefficients must be stored in the `beta` vector.
 - Assume that the feature matrix `X` is full rank (i.e., `X^T . X` is invertible).

#### Example:

<img width="1047" height="544" alt="image" src="https://github.com/user-attachments/assets/339817fa-811b-4349-bd6b-f790ead40663" />


#### Constraints
 - 1 ≤ `n_samples` ≤ 100,000
 - 1 ≤ `n_features` ≤ 1,000
 - `n_samples` ≥ `n_features`
 - -1000.0 ≤ `values` in `X` and `y` ≤ 1000.0
 - Solutions are tested with absolute tolerance of `1e-2` and relative tolerance of `1e-2`
 - Performance is measured with `n_features = 32`, `n_samples = 32`

```cuda
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}

```
