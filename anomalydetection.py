import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Fetching the CSV file
CSV_FILE = "AG_NO3_fill_cells_remove_NAN.csv"

# Columns in the dataset
VALUE_COLUMN = "NO3N"
LABEL_COLUMN = "Student_Flag"
TIME_COLUMN = "Date"

# Setting up the sliding window size and the percentile
WINDOW_SIZE = 500
PERCENTILE_Q = 90.0
TWO_SIDED = False
LOWER_PERCENTILE_Q = 5.0

# Creating the output folder

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "anomaly_output"

def load_data(file_path):
    """
    Load the CSV dataset.
    """

    # If the CSV path is relative, look for it beside
    # the Python script.

    csv_path = Path(file_path)

    if not csv_path.is_absolute():
        csv_path = SCRIPT_DIR / csv_path

    if not csv_path.exists():

        raise FileNotFoundError(
            f"\nCSV file could not be found:\n{csv_path}\n\n"
            f"Make sure '{file_path}' is in the same folder "
            f"as this Python script."
        )

    df = pd.read_csv(csv_path)

    return df

  # Processing the data
def prepare_data(df):
    """
    Preparing the nitrate values, ground-truth labels,
    and time column.
    """

    if VALUE_COLUMN not in df.columns:

        raise ValueError(
            f"Value column '{VALUE_COLUMN}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:

        raise ValueError(
            f"Label column '{LABEL_COLUMN}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    values = pd.to_numeric(
        df[VALUE_COLUMN],
        errors="coerce"
    )

    labels = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="coerce"
    )

    valid = (
        values.notna()
        & labels.notna()
    )

    values = values[valid].to_numpy(
        dtype=float
    )

    labels = labels[valid].to_numpy(
        dtype=int
    )


    if TIME_COLUMN in df.columns:

        time_values = df.loc[
            valid,
            TIME_COLUMN
        ].to_numpy()

    else:

        time_values = np.arange(
            len(values)
        )

    return (
        values,
        labels,
        time_values
    )

# Creating the detector itself

def sliding_window_detector(values):
    """
    Previous-observations-only sliding-window anomaly detector.

    For each point i:

        window = values[i-W:i]

    Therefore, the current observation is NOT included
    in the threshold calculation.

    Parameters:
        W = 500
        q = 90%

    Detection rule:

        anomaly if current_value >= threshold
    """

    n = len(values)

    predictions = np.zeros(
        n,
        dtype=int
    )

    thresholds = np.full(
        n,
        np.nan,
        dtype=float
    )

    for i in range(
        WINDOW_SIZE,
        n
    ):

        window = values[
            i - WINDOW_SIZE:i
        ]

        threshold = np.percentile(
            window,
            PERCENTILE_Q,
            method="linear"
        )

        thresholds[i] = threshold

        current_value = values[i]

        if current_value >= threshold:

            predictions[i] = 1

        else:

            predictions[i] = 0

    return (
        predictions,
        thresholds
    )

# Calcul;ating the metrics (TP, TN, FP, FN,Normal Accuracy and Anomaly Accuracy)

def calculate_metrics(
    true_labels,
    predictions
):

    evaluated_indices = np.arange(
        WINDOW_SIZE,
        len(true_labels)
    )

    y_true = true_labels[
        evaluated_indices
    ]

    y_pred = predictions[
        evaluated_indices
    ]

    TP = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    TN = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    FP = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    FN = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    # The Normal Accuracy

    if (TN + FP) > 0:

        normal_accuracy = (
            TN / (TN + FP)
        ) * 100

    else:

        normal_accuracy = 0.0

    # The Anomaly Accuracy

    if (TP + FN) > 0:

        anomaly_accuracy = (
            TP / (TP + FN)
        ) * 100

    else:

        anomaly_accuracy = 0.0

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "normal_accuracy": normal_accuracy,
        "anomaly_accuracy": anomaly_accuracy
    }

# The printed output results

def print_metrics(metrics):

    print("\n")
    print("=" * 65)
    print("These are the Confusion Matrix Results")
    print("=" * 65)

    print(
        f"True Positives (TP) : "
        f"{metrics['TP']:,}"
    )

    print(
        f"True Negatives (TN) : "
        f"{metrics['TN']:,}"
    )

    print(
        f"False Positives (FP): "
        f"{metrics['FP']:,}"
    )

    print(
        f"False Negatives (FN): "
        f"{metrics['FN']:,}"
    )

    print("\nAccuracy results:")

    print(
        f"Normal Accuracy  : "
        f"{metrics['normal_accuracy']:.2f}%"
    )

    print(
        f"Anomaly Accuracy : "
        f"{metrics['anomaly_accuracy']:.2f}%"
    )

    
# Saving the results to a csv file

def save_predictions(
    values,
    labels,
    predictions,
    thresholds,
    time_values
):
    
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = pd.DataFrame({

        "Index":
            np.arange(
                len(values)
            ),

        "Date":
            time_values,

        "NO3N":
            values,

        "Student_Flag":
            labels,

        "Threshold_97":
            thresholds,

        "Predicted_Anomaly":
            predictions
    })

    output_file = (
        OUTPUT_DIR
        / "anomalies.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    return output_file

#Creating the anomaly plot

def create_plot(
    values,
    predictions,
    thresholds
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(15, 7)
    )

    plt.plot(
        values,
        linewidth=1,
        label="NO3N"
    )

    plt.plot(
        thresholds,
        linewidth=1,
        label="90th Percentile Threshold"
    )

    anomaly_indices = np.where(
        predictions == 1
    )[0]

    if len(anomaly_indices) > 0:

        plt.scatter(
            anomaly_indices,
            values[anomaly_indices],
            color="red",
            marker=".",
            s=35,
            label="Detected anomaly"
        )

    plt.xlabel(
        "Index"
    )

    plt.ylabel(
        "NO3N"
    )

    plt.title(
        "Nitrate Anomaly Detection "
        "(W=500, q=90%)"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "anomaly_plot.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return plot_file

def main():

    print("\n")
    print("=" * 65)
    print(
        "Sliding Window Percentile Detector"
    )
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("\n")
    print("=" * 65)
    print("Dataset")
    print("=" * 65)

    df = load_data(
        CSV_FILE
    )

    print(
        f"File: {CSV_FILE}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print("\nAvailable columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    (
        values,
        labels,
        time_values
    ) = prepare_data(
        df
    )

    print("\n")
    print("=" * 65)
    print("Dataset")
    print("=" * 65)

    print(
        f"Value column : "
        f"{VALUE_COLUMN}"
    )

    print(
        f"Label column : "
        f"{LABEL_COLUMN}"
    )

    print(
        f"Time column  : "
        f"{TIME_COLUMN}"
    )

    print(
        f"Rows after cleaning: "
        f"{len(values):,}"
    )

    print(
        f"\nGround-truth anomalies: "
        f"{int(labels.sum()):,}"
    )

    print(
        f"Ground-truth normal observations: "
        f"{int((labels == 0).sum()):,}"
    )

    if len(values) <= WINDOW_SIZE:

        raise ValueError(
            f"\nWindow size W={WINDOW_SIZE} is too "
            f"large for {len(values)} observations."
        )

    print("\n")
    print("=" * 65)
    print("Final Detector Settings")
    print("=" * 65)

    print(
        f"Window size W       : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Percentile q        : "
        f"{PERCENTILE_Q}%"
    )

    print(
        "Percentile method   : "
        "linear"
    )

    print(
        "Detection type      : "
        "Upper-tail"
    )

    print(
        "Current point used in threshold: "
        "NO"
    )

    print(
        "Previous observations used     : "
        "YES"
    )

    print(
        f"Previous observations per window: "
        f"{WINDOW_SIZE}"
    )

    print(
        f"First {WINDOW_SIZE:,} observations "
        f"classified: NO"
    )

    # Running Detector

    print("\n")
    print("=" * 65)
    print("The Final Detector")
    print("=" * 65)

    (
        predictions,
        thresholds
    ) = sliding_window_detector(
        values
    )

    detected_count = int(
        predictions.sum()
    )

    print(
        f"\nDetected anomalies: "
        f"{detected_count:,}"
    )

    metrics = calculate_metrics(
        labels,
        predictions
    )

    print_metrics(
        metrics
    )

if __name__ == "__main__":
    main()
