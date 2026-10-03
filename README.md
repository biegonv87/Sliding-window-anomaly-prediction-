Here is a cleaner and more professional version of your Markdown, while keeping the technical meaning and results unchanged:

# Nitrate Sliding-Window Anomaly Detection

## 1. Dataset

* **Dataset:** `AG_NO3_fill_cells_remove_NAN.csv`
* **Value column:** `NO3N`
* **Ground-truth label column:** `Student_Flag`
* **Time column:** `Date`

## 2. Detection Method

A **threshold-based sliding-window percentile method** was used to detect upper-tail anomalies in the `NO3N` time series.

For each observation, the detection threshold was calculated from the preceding 500 observations.

## 3. Parameters

| Parameter                                 | Value          |
| ----------------------------------------- | -------------- |
| Window size (W)                           | **500**      |
| Percentile (q)                            | **90%**        |
| Percentile method                         | **Linear**     |
| Detection type                            | **Upper-tail** |
| Step size                                 | **1**          |
| Current observation included in threshold | **No**         |
| First 1,000 observations classified       | **No**         |

## 4. Threshold Calculation

For observation `i`, only the previous 500 observations were used to calculate the detection threshold:

```text
window = data[i-500:i]
threshold = percentile(window, 90)
```

The current observation was **not included** when calculating its threshold.

An observation was classified as an anomaly when its `NO3N` value was **greater than or equal to** the calculated 90th-percentile threshold.

The first 500 observations were not classified because there were not enough previous observations to form a complete window.

## 5. Results

The detector identified **4,760 observations** as anomalies.

| Metric               |     Result |
| -------------------- | ---------: |
| Detected anomalies   |  **4,760** |
| True Positives (TP)  |    **105** |
| True Negatives (TN)  | **25,496** |
| False Positives (FP) |  **4,655** |
| False Negatives (FN) |     **34** |
| Normal Accuracy      | **84.56%** |
| Anomaly Accuracy     | **75.54%** |

## 6. Design Choices

The detector uses a **one-sided upper-tail threshold**, meaning that only unusually high `NO3N` values are considered anomalies.

For every classified observation, the threshold is calculated using **only previous observations**. This prevents the current observation from influencing its own detection threshold.

A window size of **500 observations** was used. Consequently, the first 500 observations were excluded from classification because a complete historical window was not yet available.

The 90th percentile was calculated using NumPy's **`linear` percentile method**.
