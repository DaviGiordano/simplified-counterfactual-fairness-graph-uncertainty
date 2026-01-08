#!/usr/bin/env python3
"""
Functions to save tuned counterfactual fairness metrics and calculate absolute differences.
Based on plot_tuned_causal_model_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_tuned_cf_data(base_path: Path) -> pd.DataFrame:
    """Load CF evaluation results for all tuned classifiers."""
    classifiers = ["LR", "RF", "GB", "FAIRGBM"]
    rows: list[pd.DataFrame] = []
    for clf in classifiers:
        cf_path = (
            base_path
            / "output"
            / "adult"
            / "med"
            / "tuned_classification_cf"
            / clf
            / "cf_evaluation.csv"
        )
        if not cf_path.exists():
            print(f"⚠️  Missing CF results for {clf}: {cf_path}")
            continue
        try:
            df = pd.read_csv(cf_path)
        except Exception as e:
            print(f"⚠️  Could not read {cf_path}: {e}. Skipping {clf}.")
            continue
        # Ensure required columns exist
        required = [
            "causal_model",
            "Female_to_Male.negative_to_positive_switch_rate",
            "Female_to_Male.positive_to_negative_switch_rate",
            "Male_to_Female.negative_to_positive_switch_rate",
            "Male_to_Female.positive_to_negative_switch_rate",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"⚠️  Missing expected columns for {clf}: {missing}")
        df["classifier"] = clf
        rows.append(df)

    if not rows:
        raise ValueError("No tuned CF evaluation data found.")

    all_df = pd.concat(rows, ignore_index=True)
    print(f"📊 Loaded tuned CF rows: {len(all_df)}")
    return all_df


def rename_causal_models(df: pd.DataFrame) -> pd.DataFrame:
    """Rename causal model names for better display."""
    model_mapping = {
        "linear": "Linear",
        "lgbm": "LGBM",
        "causalflow": "CNF",
        "diffusion": "DCM",
    }
    df["causal_model"] = df["causal_model"].map(lambda x: model_mapping.get(x, x))
    return df


def melt_cf_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape CF rate columns to long format with metric and intervention labels."""
    mapping = {
        "Female_to_Male.negative_to_positive_switch_rate": ("PSR", "Female→Male"),
        "Female_to_Male.positive_to_negative_switch_rate": ("NSR", "Female→Male"),
        "Male_to_Female.negative_to_positive_switch_rate": ("PSR", "Male→Female"),
        "Male_to_Female.positive_to_negative_switch_rate": ("NSR", "Male→Female"),
    }
    rate_cols = list(mapping.keys())
    present_cols = [c for c in rate_cols if c in df.columns]
    m = df.melt(
        id_vars=["classifier", "causal_model"],
        value_vars=present_cols,
        var_name="rate_col",
        value_name="rate",
    )
    m["metric"] = m["rate_col"].map(lambda c: mapping.get(c, ("Unknown", "Unknown"))[0])
    m["intervention"] = m["rate_col"].map(
        lambda c: mapping.get(c, ("Unknown", "Unknown"))[1]
    )
    m.drop(columns=["rate_col"], inplace=True)
    return m


def save_plotted_values(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Save the values that are being plotted on the charts.
    Returns a DataFrame with the plotted values for further processing.
    """
    print("📊 Saving plotted values...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics for each combination
    grouped = df.groupby(["classifier", "causal_model", "metric", "intervention"])[
        "rate"
    ]

    plotted_values = []
    for name, group in grouped:
        if isinstance(name, tuple) and len(name) == 4:
            classifier, causal_model, metric, intervention = name
        else:
            continue

        plotted_values.append(
            {
                "classifier": classifier,
                "causal_model": causal_model,
                "metric": metric,
                "intervention": intervention,
                "mean": group.mean(),
                "std": group.std(),
                "ci_lower": group.quantile(0.025),
                "ci_upper": group.quantile(0.975),
                "count": len(group),
                "min": group.min(),
                "max": group.max(),
            }
        )

    plotted_df = pd.DataFrame(plotted_values)

    # Save to CSV
    output_file = output_dir / "tuned_cf_plotted_values.csv"
    plotted_df.round(3).to_csv(output_file, index=False)
    print(f"✅ Saved plotted values: {output_file}")

    return plotted_df


def calculate_absolute_differences(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Calculate differences between NSR and PSR for male→female MINUS female→male.
    Returns a DataFrame with causal model, classifier, PSR difference, NSR difference.
    """
    print("📊 Calculating differences (male→female MINUS female→male)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pivot the data to get mean values for each combination
    pivot_df = (
        df.groupby(["classifier", "causal_model", "metric", "intervention"])["rate"]
        .mean()
        .reset_index()
    )

    # Create separate DataFrames for each metric and intervention
    nsr_female_male = pivot_df[
        (pivot_df["metric"] == "NSR") & (pivot_df["intervention"] == "Female→Male")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    nsr_male_female = pivot_df[
        (pivot_df["metric"] == "NSR") & (pivot_df["intervention"] == "Male→Female")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    psr_female_male = pivot_df[
        (pivot_df["metric"] == "PSR") & (pivot_df["intervention"] == "Female→Male")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    psr_male_female = pivot_df[
        (pivot_df["metric"] == "PSR") & (pivot_df["intervention"] == "Male→Female")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    # Calculate differences (male→female MINUS female→male)
    nsr_diff = nsr_male_female - nsr_female_male
    psr_diff = psr_male_female - psr_female_male

    # Convert to long format: causal model, classifier, PSR difference, NSR difference
    results = []
    for causal_model in nsr_diff.index:
        for classifier in nsr_diff.columns:
            results.append(
                {
                    "causal_model": causal_model,
                    "classifier": classifier,
                    "PSR_difference": psr_diff.loc[causal_model, classifier],
                    "NSR_difference": nsr_diff.loc[causal_model, classifier],
                }
            )

    combined_results = pd.DataFrame(results)

    # Save to CSV
    output_file = output_dir / "tuned_cf_differences.csv"
    combined_results.round(3).to_csv(output_file, index=False)
    print(f"✅ Saved differences: {output_file}")

    return combined_results


def create_detailed_metrics_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create a detailed table with all metrics for each causal model and classifier combination.
    Each row is a causal model, each column is a classifier with separate columns for NSR and PSR.
    """
    print("📊 Creating detailed metrics table...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pivot the data to get mean values for each combination
    pivot_df = (
        df.groupby(["classifier", "causal_model", "metric", "intervention"])["rate"]
        .mean()
        .reset_index()
    )

    # Create separate DataFrames for each metric and intervention
    nsr_female_male = pivot_df[
        (pivot_df["metric"] == "NSR") & (pivot_df["intervention"] == "Female→Male")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    nsr_male_female = pivot_df[
        (pivot_df["metric"] == "NSR") & (pivot_df["intervention"] == "Male→Female")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    psr_female_male = pivot_df[
        (pivot_df["metric"] == "PSR") & (pivot_df["intervention"] == "Female→Male")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    psr_male_female = pivot_df[
        (pivot_df["metric"] == "PSR") & (pivot_df["intervention"] == "Male→Female")
    ].pivot(index="causal_model", columns="classifier", values="rate")

    # Add descriptive suffixes to column names
    nsr_female_male.columns = [
        f"{col}_NSR_Female→Male" for col in nsr_female_male.columns
    ]
    nsr_male_female.columns = [
        f"{col}_NSR_Male→Female" for col in nsr_male_female.columns
    ]
    psr_female_male.columns = [
        f"{col}_PSR_Female→Male" for col in psr_female_male.columns
    ]
    psr_male_female.columns = [
        f"{col}_PSR_Male→Female" for col in psr_male_female.columns
    ]

    # Combine all metrics
    detailed_table = pd.concat(
        [nsr_female_male, nsr_male_female, psr_female_male, psr_male_female], axis=1
    )

    # Save to CSV
    output_file = output_dir / "tuned_cf_detailed_metrics.csv"
    detailed_table.round(3).to_csv(output_file)
    print(f"✅ Saved detailed metrics table: {output_file}")

    return detailed_table


def main():
    """Main function to run all the analysis and save results."""
    print("🚀 Starting tuned CF metrics analysis...")
    print("=" * 60)

    base_path = Path(__file__).parent
    output_dir = base_path / "adult_tuned_causal_model_metrics"

    try:
        # Load and process data
        raw_df = load_tuned_cf_data(base_path)
        raw_df = rename_causal_models(raw_df)
        long_df = melt_cf_rates(raw_df)

        # Save plotted values
        plotted_values = save_plotted_values(long_df, output_dir)

        # Calculate absolute differences
        abs_differences = calculate_absolute_differences(long_df, output_dir)

        # Create detailed metrics table
        detailed_table = create_detailed_metrics_table(long_df, output_dir)

        print("\n🎉 All tuned CF metrics analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        print("=" * 60)

        # Print summary
        print("\n📊 Summary of generated files:")
        print(f"  - tuned_cf_plotted_values.csv: {len(plotted_values)} rows")
        print(
            f"  - tuned_cf_absolute_differences.csv: {len(abs_differences)} rows × {len(abs_differences.columns)} columns"
        )
        print(
            f"  - tuned_cf_detailed_metrics.csv: {len(detailed_table)} rows × {len(detailed_table.columns)} columns"
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
