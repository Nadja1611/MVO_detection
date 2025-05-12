import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os


def boxplot(tp_indices, fn_indices, lesion_volumes, infarct_volumes, data_dir=None):
    # Extract FN and TP lesion volumes
    fn_volumes = lesion_volumes[fn_indices]
    fn_volumes_is = infarct_volumes[fn_indices]

    tp_volumes = lesion_volumes[tp_indices]
    tp_volumes_is = infarct_volumes[tp_indices]

    # Compute means and medians
    mean_fn, median_fn = fn_volumes.mean().item(), fn_volumes.median().item()
    mean_tp, median_tp = tp_volumes.mean().item(), tp_volumes.median().item()

    mean_fn_is, median_fn_is = (
        fn_volumes_is.mean().item(),
        fn_volumes_is.median().item(),
    )
    mean_tp_is, median_tp_is = (
        tp_volumes_is.mean().item(),
        tp_volumes_is.median().item(),
    )

    print("FN Lesion Volumes - Mean:", mean_fn, "Median:", median_fn, flush=True)
    print("TP Lesion Volumes - Mean:", mean_tp, "Median:", median_tp, flush=True)

    print("FN Infarct Volumes - Mean:", mean_fn_is, "Median:", median_fn_is, flush=True)
    print("TP Infarct Volumes - Mean:", mean_tp_is, "Median:", median_tp_is, flush=True)

    # Convert to Pandas DataFrame for Seaborn
    df_lesion = pd.DataFrame(
        {
            "Volume": torch.cat([fn_volumes, tp_volumes]).numpy(),
            "Type": ["FN"] * len(fn_volumes) + ["TP"] * len(tp_volumes),
            "Measure": ["Lesion Volume"] * (len(fn_volumes) + len(tp_volumes)),
        }
    )

    df_infarct = pd.DataFrame(
        {
            "Volume": torch.cat([fn_volumes_is, tp_volumes_is]).numpy(),
            "Type": ["FN"] * len(fn_volumes_is) + ["TP"] * len(tp_volumes_is),
            "Measure": ["Infarct Volume"] * (len(fn_volumes_is) + len(tp_volumes_is)),
        }
    )

    df = pd.concat([df_lesion, df_infarct])

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Lesion Volume Boxplot
    sns.boxplot(
        x="Type",
        y="Volume",
        data=df_lesion,
        palette={"FN": "#A1CAF1", "TP": "#FBC15E"},
        ax=axes[0],
    )
    axes[0].set_title("Lesion Volume Distribution")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Lesion Volume")
    axes[0].grid(True)

    # Add annotations
    axes[0].text(
        0,
        mean_fn,
        f"Mean: {mean_fn:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="blue",
    )
    axes[0].text(
        0,
        median_fn,
        f"Median: {median_fn:.2f}",
        ha="center",
        va="top",
        fontsize=10,
        color="blue",
    )

    axes[0].text(
        1,
        mean_tp,
        f"Mean: {mean_tp:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="darkorange",
    )
    axes[0].text(
        1,
        median_tp,
        f"Median: {median_tp:.2f}",
        ha="center",
        va="top",
        fontsize=10,
        color="darkorange",
    )

    # Infarct Volume Boxplot
    sns.boxplot(
        x="Type",
        y="Volume",
        data=df_infarct,
        palette={"FN": "#A1CAF1", "TP": "#FBC15E"},
        ax=axes[1],
    )
    axes[1].set_title("Infarct Volume Distribution")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Infarct Volume")
    axes[1].grid(True)

    # Add annotations
    axes[1].text(
        0,
        mean_fn_is,
        f"Mean: {mean_fn_is:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="blue",
    )
    axes[1].text(
        0,
        median_fn_is,
        f"Median: {median_fn_is:.2f}",
        ha="center",
        va="top",
        fontsize=10,
        color="blue",
    )

    axes[1].text(
        1,
        mean_tp_is,
        f"Mean: {mean_tp_is:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="darkorange",
    )
    axes[1].text(
        1,
        median_tp_is,
        f"Median: {median_tp_is:.2f}",
        ha="center",
        va="top",
        fontsize=10,
        color="darkorange",
    )

    # Adjust layout and save
    plt.tight_layout()
    if data_dir:
        plt.savefig(f"boxplot_{os.path.basename(data_dir)}.png", dpi=300)
