
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


OUTPUT_PATH = Path("./results")
METHOD_COLORS = {
    "easyocr": "tab:blue",
    "tesseract": "tab:orange",
    "paddleocr": "tab:green",
}


def plot_metrics(results):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    methods = ["easyocr", "tesseract", "paddleocr"]

    experiments = [
        ("jpeg", "Jakość JPEG", "jpg_quality"),
        ("resolution", "Rozdzielczość", "size_ratio"),
        ("noise", "Szum", "noise_std"),
    ]

    metrics = [
        ("average_cer", "CER", "cer"),
        ("average_wer", "WER", "wer"),
        ("ocr_time", "Czas OCR [s]", "time"),
    ]

    for experiment, experiment_name, parameter in experiments:

        experiment_results = [
            r for r in results
            if r["experiment"] == experiment
        ]

        values = sorted(set(
            r[parameter] for r in experiment_results
        ))

        x = np.arange(len(values))
        width = 0.25

        labels = [str(value) for value in values]

        for metric, metric_name, filename in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, method in enumerate(methods):
                y_values = []

                for value in values:
                    matching = [
                        r[metric]
                        for r in experiment_results
                        if r["method"] == method
                        and r[parameter] == value
                    ]

                    y_values.append(
                        np.mean(matching) if matching else np.nan
                    )

                ax.bar(
                    x + (i - 1) * width,
                    y_values,
                    width,
                    label=method,
                    color=METHOD_COLORS[method],
                )

            ax.set_title(
                f"Porównanie metod OCR – {experiment_name} ({metric_name})"
            )
            ax.set_xlabel(experiment_name)
            ax.set_ylabel(metric_name)

            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            ax.grid(axis="y", alpha=0.3)
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                OUTPUT_PATH / f"{filename}_{experiment}.png",
                dpi=150,
                bbox_inches="tight",
            )

            plt.show()
            plt.close(fig)


def plot_metrics_seperate(results):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("jpeg", "Jakość JPEG", "jpg_quality"),
        ("resolution", "Współczynnik skalowania", "size_ratio"),
        ("noise", "Odchylenie standardowe szumu", "noise_std"),
    ]

    metrics = [
        ("average_cer", "CER", "cer"),
        ("average_wer", "WER", "wer"),
        ("ocr_time", "Czas OCR [s]", "time"),
    ]

    for metric, metric_name, filename in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (experiment, parameter_name, parameter) in zip(
            axes, experiments
        ):
            experiment_results = [
                r for r in results
                if r["experiment"] == experiment
            ]

            for method, color in METHOD_COLORS.items():

                grouped = {}

                for result in experiment_results:
                    if result["method"] != method:
                        continue

                    value = result[parameter]
                    grouped.setdefault(value, []).append(result[metric])

                if not grouped:
                    continue

                x_values = sorted(grouped.keys())

                y_values = [
                    np.mean(grouped[x])
                    for x in x_values
                ]

                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    label=method,
                    color=color,
                )

            ax.set_title(parameter_name)
            ax.set_xlabel(parameter_name)
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)

            if ax.get_legend_handles_labels()[0]:
                ax.legend()

        fig.suptitle(f"Porównanie metod OCR – {metric_name}")
        fig.tight_layout()

        fig.savefig(
            OUTPUT_PATH / f"{filename}_separate.png",
            dpi=150,
        )

        plt.show()
        plt.close(fig)


def plot_example(img, evaluations, title):
    """
    evaluations:
        {
            "easyocr": wynik evaluate_ocr_vs_ground_truth(...),
            "tesseract": ...,
            "paddleocr": ...
        }
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    img_rgb = img[:, :, ::-1].copy()

    methods = list(evaluations.keys())
    fig, axes = plt.subplots(
        1, len(methods),
        figsize=(7 * len(methods), 8),
    )

    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        evaluation = evaluations[method]

        ax.imshow(img_rgb)
        ax.set_title(
            f"{method}\n"
            f"CER: {evaluation['average_cer']:.3f} | "
            f"WER: {evaluation['average_wer']:.3f}"
        )
        ax.axis("off")

        for i, detail in enumerate(evaluation["details"]):
            x1, y1, x2, y2 = detail["gt_box"]
            cer = detail["cer"]

            if cer < 0.2:
                color = "green"
            elif cer < 0.6:
                color = "yellow"
            else:
                color = "red"

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.text(
                x1, max(0, y1 - 3),
                f"{cer:.2f}",
                color="black" if color == "yellow" else "white",
                fontsize=8,
                bbox=dict(
                    facecolor=color,
                    alpha=0.4,
                    edgecolor="none",
                    pad=2,
                ),
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATH / f"example_{title.replace(' ', '_')}.png",
        dpi=150,
    )
    plt.show()
    plt.close(fig)