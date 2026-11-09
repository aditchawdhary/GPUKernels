#### Distillation KL Divergence

In knowledge distillation, a small student LLM is trained to match the full next-token distribution of a larger teacher. For each of `N` token positions, compute the forward KL divergence between the temperature-softened teacher and student distributions over a vocabulary of `V` entries. 
The inputs `teacher_logits` and `student_logits` are float32 matrices of shape `(N, V)`. Write one float32 value per row to `kl`, a vector of shape `(N,)`.

<img width="760" height="253" alt="image" src="https://github.com/user-attachments/assets/26a98864-b7ae-41f9-8cdb-403c4286f92a" />

<img width="760" height="271" alt="image" src="https://github.com/user-attachments/assets/76d6827d-251f-4f16-b8fd-72549b8731ad" />


#### Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The solve function signature must remain unchanged
 - The final result must be stored in kl
 - kl is not zero-initialized; every element must be written

#### Example 1

 <img width="763" height="183" alt="image" src="https://github.com/user-attachments/assets/71020781-d156-4642-8223-af1139c07d34" />


#### Example 2

<img width="763" height="190" alt="image" src="https://github.com/user-attachments/assets/b6b5657e-349c-4a56-be9f-15650edf1acd" />
 

#### Constraints
 - 1 ≤ `N` ≤ 65,536
 - 1 ≤ `V` ≤ 262,144
 - 0.5 ≤ `temperature` ≤ 10.0
 - All logits are float32 values in [−500, 500]
 - Performance is measured with `N` = 2,048, `V` = 128,256, `temperature` = 2.0
