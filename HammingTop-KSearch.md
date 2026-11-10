### Hamming Top-K Search

Implement a GPU program that finds the k nearest binary codes to a query under Hamming distance. The query is a `256-bit` code stored as `8` `int32 words`. The array codes holds `N` such codes, flattened: code `i` is `codes[8i .. 8i+7]`. 
The distance from the query to code `i` is the number of bit positions at which they differ: the sum over `w = 0 to 7` of the popcount of `query[w] XOR codes[8i + w]`. Every one of the `32 bits` of every word counts, including the sign bit, so a distance is between `0 and 256`.

Write to output the indices of the k codes with the smallest distance, sorted by distance in ascending order. Codes with equal distance are ordered by lower index first. Because of this tie rule the answer is unique, even when many codes share the cutoff distance. 
This is the key-selection step of hash-based sparse attention and of binary-embedding nearest-neighbor search.

### Implementation Requirements
 - External libraries are not permitted
 - The solve function signature must remain unchanged
 - The final result must be stored in array output
 - The words are treated as raw 32-bit patterns; int32 is only the storage type

### Example 1:
```
Input:  query = [1, 2, 3, 0, 0, 0, 0, 0]
        codes = [1, 2, 3, 0, 0, 0, 0, 0,
                 0, 2, 3, 0, 0, 0, 0, 0,
                 7, 0, 3, 0, 0, 0, 0, 0,
                 1, 3, 2, 0, 0, 0, 0, 0,
                 1, 2, 2, 0, 0, 0, 0, 0]
        N = 5
        k = 2
Distances: [0, 1, 3, 2, 1]
Output: output = [0, 1]
Codes 1 and 4 tie at distance 1, and code 1 comes first because its index is lower.
```

### Example 2:
```
Input:  query = [0, 0, 0, 0, 0, 0, 0, 0]
        codes = [1, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0,
                 -1, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 1]
        N = 4
        k = 2
Distances: [2, 0, 32, 2]
Output: output = [1, 0]
Codes 0 and 3 tie at distance 2 at the cutoff, and only the one with the lower index is kept.
```

### Constraints
 - 1 ≤ `N` ≤ 10,000,000
 - 1 ≤ `k` ≤ min(N, 1,024)
 - `codes` has 8 × `N` elements and `query` has 8 elements, all signed `32-bit integers`; all 32 bits count, including the sign bit
 - Performance is measured with `N` = 4,000,000, `k` = 128
