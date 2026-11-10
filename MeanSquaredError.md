## Mean Squared Error

Implement a GPU program to calculate the Mean Squared Error (MSE) between predicted values and target values. Given two arrays of equal `length`, `predictions` and `targets`, compute:

<img width="729" height="77" alt="image" src="https://github.com/user-attachments/assets/5cef29e3-144e-4e3a-86a4-e1dce95c610d" />
 
where N is the number of elements in each array.

### Implementation Requirements
 - External libraries are not permitted.
 - The solve function signature must remain unchanged.
 - The final result must be stored in the mse variable.

### Example 1:
```
  Input:  predictions = [1.0, 2.0, 3.0, 4.0]
          targets = [1.5, 2.5, 3.5, 4.5]
  Output: mse = 0.25
```

### Example 2:
```
  Input:  predictions = [10.0, 20.0, 30.0]
          targets = [12.0, 18.0, 33.0]
  Output: mse = 5.67
```

### Constraints
 - 1 ≤ N ≤ 100,000,000
 - -1000.0 ≤ predictions[i], targets[i] ≤ 1000.0
 - Performance is measured with N = 50,000,000
