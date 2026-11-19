## Non-Maximum Suppression

Implement Non-Maximum Suppression (NMS), the post-processing pass that every object detector runs after the network forward pass: the model emits thousands of overlapping candidate boxes, and NMS collapses them down to one box per object. 
Given `N` boxes in corner format `(x1, y1, x2, y2)` together with their confidence scores, walk the boxes from the highest score to the lowest; every box that has not been suppressed yet is kept, and it suppresses all lower-scoring boxes whose intersection over union `(IoU)` with it is greater than `iou_threshold`. 
Write a 1 into `keep[i]` for a surviving box and a 0 for a suppressed one.

<img width="1170" height="564" alt="image" src="https://github.com/user-attachments/assets/172d6030-68af-437d-8aba-09ef9e0fbe29" />

The overlap of two boxes `a` and `b` is measured by their intersection over union:

<img width="1166" height="221" alt="image" src="https://github.com/user-attachments/assets/3f5f7002-ff60-40cd-b04d-1c9f54f5b931" />

 
## Algorithm
1. Order the box indices by score, from highest to lowest. Boxes with equal scores are ordered by increasing index.
2. Traverse that order. If the current box has already been suppressed, skip it. Otherwise set its `keep` entry to 1 and suppress every box that appears later in the order whose IoU with it is greater than `iou_threshold`.
3. Every suppressed box gets a `keep` entry of 0.
A suppressed box never suppresses anything itself, so the decisions are chained: whether box `j` survives depends on which higher-scoring boxes survived before it.

## Implementation Requirements
 - Use only native features (external libraries are not permitted)
 - The `solve` function signature must remain unchanged
 - The final result must be stored in the `keep` tensor
 - A box is suppressed only when its IoU is strictly greater than `iou_threshold`
 - `keep` is indexed by the original box order, not by the score-sorted order
 - Every entry of `keep` must be written: 1 for a kept box, 0 for a suppressed box

## Example
Six candidates covering three objects, with `iou_threshold` = 0.5:

<img width="1182" height="308" alt="image" src="https://github.com/user-attachments/assets/66ccece8-76a7-412e-b143-6d34cca77348" />
 
 
 
Box 2 has the highest score, so it is kept and suppresses box 3 <img width="177" height="25" alt="image" src="https://github.com/user-attachments/assets/772440b9-1d32-400e-97db-e21e8fc51b5f" />
Box 0 is kept next and suppresses box 1 by the same overlap, and box 4 is kept and suppresses box 5. Boxes from different objects never overlap here, so all three winners survive.

## Constraints
 - 1 ≤ `N` ≤ 20,000
 - boxes has shape `(N, 4)` and is stored row-major as `(x1, y1, x2, y2)` with `x1 < x2` and `y1 < y2`
 - Box coordinates are integer-valued and lie in `[−10,000, 10,000]`
 - `0 ≤ scores[i] ≤ 1`; equal scores are possible
 - `0 ≤ iou_threshold ≤ 1`
 - `boxes` and scores are `32-bit` floating point, keep is `32-bit` signed integer
 - Performance is measured with `N = 16,384` and `iou_threshold = 0.5`
