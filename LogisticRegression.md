#### Logistic Regression

Solve the logistic regression problem on a GPU. Given a feature matrix `X` of size `n_samples x n_features` and a binary target vector `y` of size `n_samples`(containing only 0s and 1s), compute the coefficient vector `\beta` that maximizes the log-likelihood:
<img width="738" height="114" alt="image" src="https://github.com/user-attachments/assets/5dbb734a-50bd-4dc2-95f9-04a38153bc15" />


#### Implementation Requirements
 - External libraries are not permitted
 - The `solve` function signature must remain unchanged
 - The final coefficients must be stored in the `beta` vector
 - The target vector `y` contains only binary values (0 and 1)

#### Example:

<img width="738" height="668" alt="image" src="https://github.com/user-attachments/assets/681ca1b9-1cc9-4640-a398-99fe112bd5c5" />

 

#### Constraints
 - 1 ≤ `n_samples` ≤ 100,000
 - 1 ≤ `n_features` ≤ 1,000
 - `n_samples` ≥ `n_features`
 - -10.0 ≤ `values in X` ≤ 10.0
 - y contains only binary values: 0 or 1
 - Solutions are tested with absolute tolerance of `1e-2` and relative tolerance of `1e-2`
 - Performance is measured with `n_features = 8`, `n_samples = 16`

```
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}

```
