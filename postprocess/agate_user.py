import numpy as np
import time
import os
import sys
import json
import anyio
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from datetime import datetime
import json
from collections import defaultdict
try:
    import seaborn as sns
except ImportError:
    sns = None  # or pass, or print a warning
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import csv
from scipy import stats as scipy_stats
import pprint


dir1 = os.path.abspath(__file__)
dir2 = os.path.dirname(dir1)
dir3 = os.path.dirname(dir2)
dir4 = os.path.dirname(dir3)
tests_dir = os.path.dirname(dir4)
sys.path.insert(1, dir2)
try:
    from ecc_pattern_1000 import *
except ImportError:
    print('ecc_pattern_1000 not found - this is optional')

class SocFlags:
    """SOC operation flags"""

    SOC_READY = 100
    OPERATION_COMPLETE = 101
    READY_FOR_NEXT = 3
    READ_START = 5
    READ_END = 6


NUM_IO = 78
NUM_BL = 64
NUM_WL = 1296
WL_PER_CHUNK = 162  # 1296/8 = 162
NUM_CHUNKS = 8
SEED = 0x12345677


"""
Pseudo-Random Number Generator (PRNG)

This implements a Linear Congruential Generator (LCG) with parameters:
- Multiplier (A) = 1664525
- Increment (C) = 1013904223
- Modulus (M) = 2^32 (implemented via & 0xFFFFFFFF)

The algorithm generates the next number in sequence using:
X_(n+1) = (A * X_n + C) mod M

For generating levels 0-3, we use bits 12-13 of the 32-bit number:
- This provides good uniformity
- A sample of 78x64x1296 Produces
- Value 0: 1617439 times (25.0%)
- Value 1: 1617411 times (25.0%)
- Value 2: 1617370 times (25.0%)
- Value 3: 1617412 times (25.0%)
- They work identically on both platforms (X86 and Agate-RISCV)
- producing identical outputs for the same seed
"""


class PRNG:
    # Constants matching C implementation
    # ----- !!!!! DO NOT CHANGE !!!! ------
    PRNG_A = 1664525
    PRNG_C = 1013904223
    PRNG_M = 0xFFFFFFFF  # 2^32

    def __init__(self, seed):
        self.seed = SEED & self.PRNG_M
        self.current = self.seed

    def next(self):
        """Generate next random number"""
        self.current = (self.PRNG_A * self.current + self.PRNG_C) & self.PRNG_M
        return self.current

    def range(self, min_val, max_val):
        """Generate random number in range [min_val, max_val]
        Using bits 12-13 for better randomness with power-of-2 ranges"""
        range_size = max_val - min_val + 1
        val = self.next()
        return min_val + ((val >> 12) % range_size)

    def reset(self):
        """Reset to initial state"""
        self.current = self.seed


def verify_prng_sequence(total_values, save_file=None, chunk_size=78, print_columns=5):
    prng = PRNG(SEED)
    values = []

    for i in range(total_values):
        val = prng.range(0, 3)
        values.append(val)

    print(f"Target values (showing first {print_columns} columns):")
    num_columns = (total_values + chunk_size - 1) // chunk_size
    print(f"Total columns {num_columns}")
    for col in range(min(print_columns, num_columns)):
        print(f"Column {col}: [", end="")
        for row in range(chunk_size):
            idx = col * chunk_size + row
            if idx < total_values:
                print(f"{values[idx]}", end="")
                if row < chunk_size - 1 and (col * chunk_size + row + 1) < total_values:
                    print(", ", end="")
        print("]")
    print("...")

    values_array = np.array(values, dtype=np.uint8)

    if total_values == 78 * 64 * 1296:
        values_array = values_array.reshape(78, 64, 1296)

    if save_file:
        np.save(save_file, values_array)
        print(
            f"Saved PRNG sequence to {save_file} and size of array {values_array.size}"
        )

    return values_array


def get_level_from_adc(adc_value, read_pass_ranges):
    if adc_value <= read_pass_ranges[0]:
        return 0
    elif adc_value < read_pass_ranges[1]:
        return 1
    elif adc_value < read_pass_ranges[2]:
        return 2
    else:
        return 3


def bin_to_array(filename: str) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.uint8)
    result = data.reshape(78, 64, 1296)
    # Save as .npy file
    output_file = filename.replace(".bin", ".npy")
    np.save(output_file, result)
    return result


def verify_forming_yield(read_data, bl_list, wl_list, io_list, level):

    if isinstance(read_data, str):
        if read_data.endswith(".bin"):
            read_values = np.fromfile(read_data, dtype=np.uint8).reshape(78, 64, 1296)
        else:
            read_values = np.load(read_data)
    else:
        read_values = read_data

    # Get dimensions
    num_bl = len(bl_list)
    num_wl = len(wl_list)
    num_io = len(io_list)
    total_cells = num_bl * num_wl * num_io

    bl_indices = np.array(bl_list)[:, np.newaxis, np.newaxis]
    wl_indices = np.array(wl_list)[np.newaxis, :, np.newaxis]
    io_indices = np.array(io_list)[np.newaxis, np.newaxis, :]
    adc_values = read_values[io_indices, bl_indices, wl_indices]

    # Count formed cells (ADC > threshold)
    formed_cells = np.sum(adc_values > level)
    forming_yield = (formed_cells / total_cells) * 100

    print(f"\nForming Yield Analysis")
    print("-" * 40)
    print(f"Target Level: {level}")
    print(f"Total Cells: {total_cells:,}")
    print(f"Successfully Formed: {formed_cells:,}")
    print(f"Forming Yield: {forming_yield:.2f}%")

    return forming_yield


def compare_with_read(
    read_data,
    bl_list,
    wl_list,
    io_list,  # Contains the actual IO indices e.g. [0,5,10]
    seed=SEED,
    prng=True,
    level=0xFF,
    row_bar=False,
    analyze_spatial_ber_flag=False,
    print_cols=15,
):
    read_pass_ranges = [14, 38, 48]
    num_bl = len(bl_list)
    num_wl = len(wl_list)
    num_io = len(io_list)  # Number of enabled IOs

    print("\nPerforming level comparison analysis...")
    print("-" * 60)
    print("LEVEL COMPARISON ANALYSIS")
    print("-" * 60)

    print(f"Compare ADC values from full array with targets")
    print(f"BL count: {num_bl}, WL count: {num_wl}, Enabled IO count: {num_io}")
    print(f"IO indices: {io_list}")

    # Load ADC values
    if isinstance(read_data, str):
        if read_data.endswith(".bin"):
            read_values = np.fromfile(read_data, dtype=np.uint8)
            if read_values.size == 78 * 64 * 1296:
                read_values = read_values.reshape(78, 64, 1296)
        elif read_data.endswith(".npy"):
            read_values = np.load(read_data)
    else:
        read_values = read_data

    # Generate target values exactly like C code - only for enabled IOs
    prng_gen = PRNG(seed)
    target_values = []

    if row_bar is True:
        bl_ranges = [
            (0, 16, 0),  # BL 0-15: Level 0
            (16, 32, 1),  # BL 16-31: Level 1
            (32, 48, 2),  # BL 32-47: Level 2
            (48, 64, 3),  # BL 48-63: Level 3
        ]

    # Generate targets for each BL/WL combination
    # Similar to whats done in C code
    for bl in bl_list:
        # Determine level based on BL range if not using PRNG
        target_level = level
        if row_bar is True:
            if not prng and level == 0xFF:
                # Find which range this BL falls into
                for start, end, range_level in bl_ranges:
                    if start <= bl < end:
                        target_level = range_level
                        break

        for wl in wl_list:
            # Generate num_io values for enabled IOs
            col_targets = []
            for i in range(num_io):  # Generate one value per enabled IO
                if prng and level == 0xFF:
                    target = prng_gen.range(0, 3)
                else:
                    target = target_level
                col_targets.append(target)
            target_values.extend(col_targets)

    print("obtaining ADC values")
    read_values_flat = read_values.flatten()
    bl_indices = np.array(bl_list)[:, np.newaxis, np.newaxis]  # Shape: (num_bl, 1, 1)
    wl_indices = np.array(wl_list)[np.newaxis, :, np.newaxis]  # Shape: (1, num_wl, 1)
    io_indices = np.array(io_list)[np.newaxis, np.newaxis, :]  # Shape: (1, 1, num_io)
    indices = (bl_indices * 1296 + wl_indices) * 78 + io_indices
    indices = indices.ravel()
    adc_values = read_values_flat[indices]

    print(f"Target values size {len(adc_values)}")
    print(f"ADC values size {len(adc_values)}")
    print(f"READ PASS RANGES {read_pass_ranges}")
    read_levels = np.array(
        [get_level_from_adc(x, read_pass_ranges) for x in adc_values]
    )
    target_values = np.array(target_values)

    matches = target_values == read_levels
    match_count = np.sum(matches)
    mismatch_indices = np.where(~matches)[0]

    # Per-level analysis
    level_stats = {}
    for level in range(4):
        level_mask = target_values == level
        total = np.sum(level_mask)
        errors = np.sum(~matches & level_mask)

        # Convert to bit arrays for error calculation
        level_targets = target_values[level_mask]
        level_reads = read_levels[level_mask]
        target_bits = np.column_stack(((level_targets >> 1) & 1, level_targets & 1))
        read_bits = np.column_stack(((level_reads >> 1) & 1, level_reads & 1))
        bit_errors = np.sum(target_bits != read_bits)

        level_stats[level] = {
            "total": total,
            "errors": errors,
            "bit_errors": bit_errors,
        }

    total_cells = len(target_values)
    total_bits = total_cells * 2
    bit_errors = sum(stats["bit_errors"] for stats in level_stats.values())
    ber = bit_errors / total_bits if total_bits > 0 else 0

    print(f"\nComparison Results:")
    print(f"Programmed cells compared: {total_cells}")
    print(f"Matching values: {match_count}")
    print(f"Mismatches: {len(mismatch_indices)}")

    print(f"\nPer-Level Analysis:")
    for level in range(4):
        stats = level_stats[level]
        print(f"\nLevel {level}:")
        print(f"  Total cells: {stats['total']}")
        if stats["total"] > 0:
            print(
                f"  Cell errors: {stats['errors']} ({(stats['errors']/stats['total']*100):.2f}%)"
            )
            print(f"  Bit errors: {stats['bit_errors']}")
            print(f"  Level BER: {(stats['bit_errors']/(stats['total']*2)):.2e}")
        else:
            print("  No cells programmed to this level")

    print(f"\nOverall Bit Error Rate Analysis:")
    print(f"Total bits compared: {total_bits}")
    print(f"Total bit errors: {bit_errors}")
    print(f"Overall BER: {ber:.2e}")


def randomness_test():
    prng = PRNG(SEED)
    values = []
    for i in range(NUM_IO * NUM_BL * NUM_WL):
        val = prng.range(0, 3)
        values.append(val)

    for i in range(0, 5, 78):
        print(f"{i:3d}-{i+9:3d}: ", end="")
        print(" ".join(f"{val}" for val in values[i : i + 10]))

    from collections import Counter

    counts = Counter(values)
    print("\nValue distribution:")
    for val in sorted(counts.keys()):
        print(f"Value {val}: {counts[val]} times ({counts[val]/len(values)*100:.1f}%)")


def plot_adc_analysis(
    io_list,  # List of IO indices to analyze
    bl_list,  # List of bitlines used
    wl_list,  # List of wordlines used
    seed=SEED,  # Seed for PRNG
    pattern_mode=0xFF,  # 0xFF for PRNG, 0xEC for ECC, 0-3 for fixed level
    read_pass_ranges=[14, 40, 58],  # ADC value thresholds
    highlight_io=None,  # Specific IO to highlight (optional)
    save_dir=None,
):
    start_time = time.time()

    # Get dimensions
    num_bl = len(bl_list)
    num_wl = len(wl_list)
    num_io = len(io_list)

    print(f"\nGenerating comprehensive analysis for {num_io} IOs")
    print(f"Total BL/WL combinations: {num_bl * num_wl}")

    # Pre-load all NPY files
    read_values_dict = {}
    for io_index in io_list:
        npy_file = f"IO{io_index}.npy"
        npy_filename = os.path.join(save_dir, npy_file)
        try:
            read_values_dict[io_index] = np.load(npy_filename)
            print(f"Loaded NPY file: {npy_filename}")
        except Exception as e:
            print(f"Error loading NPY file {npy_filename}: {e}")
            io_list = [io for io in io_list if io != io_index]

    if not io_list:
        print("No valid IO files could be loaded. Aborting.")
        return None

    # Initialize pattern counter for ECC patterns
    pattern_counter = 0

    # Generate all targets first
    all_targets = {}
    prng_gen = PRNG(seed)

    print("Generating target values...")
    for bl in bl_list:
        for wl in wl_list:
            bl_wl_targets = []
            if pattern_mode == 0xEC:  # ECC pattern mode
                # (rolling over at 1000)
                pattern_index = pattern_counter % len(ECC_PATTERN_1000)
                pattern_counter += 1
                if pattern_counter >= 1000:
                    pattern_counter = 0
                current_pattern = ECC_PATTERN_1000[pattern_index]
                # Extract 2-bit targets for the specific IOs we're analyzing
                for io_index in io_list:
                    pattern_byte = current_pattern[io_index // 2]
                    target_2bit = (pattern_byte >> ((io_index % 2) * 2)) & 0x3
                    bl_wl_targets.append(target_2bit)
            elif pattern_mode == 0xFF:  # PRNG mode
                for _ in range(len(io_list)):
                    target = prng_gen.range(0, 3)
                    bl_wl_targets.append(target)
            else:
                # Fixed level mode (0, 1, 2, or 3)
                level = pattern_mode
                for _ in range(len(io_list)):
                    bl_wl_targets.append(level)
            all_targets[(bl, wl)] = bl_wl_targets

    # Group ADC values by IO and target level
    io_level_adc_values = {}
    for io_idx, io_index in enumerate(io_list):
        io_level_adc_values[io_index] = {0: [], 1: [], 2: [], 3: []}

    print("Collecting ADC values...")
    # Collect ADC values
    for bl in bl_list:
        for wl in wl_list:
            for io_idx, io_index in enumerate(io_list):
                read_values = read_values_dict[io_index]
                target = all_targets[(bl, wl)][io_idx]
                adc_value = read_values[0, bl, wl]

                # Store ADC value by target level and IO
                io_level_adc_values[io_index][target].append(adc_value)

    # Calculate statistics
    io_level_stats = {}
    for io_index in io_list:
        io_level_stats[io_index] = {}
        for level in range(4):
            values = np.array(io_level_adc_values[io_index][level])

            if len(values) > 0:
                mean = np.mean(values)
                std_dev = np.std(values)
                count = len(values)
                p25 = np.percentile(values, 25) if len(values) >= 4 else mean
                p75 = np.percentile(values, 75) if len(values) >= 4 else mean
            else:
                mean = 0
                std_dev = 0
                count = 0
                p25 = 0
                p75 = 0

            io_level_stats[io_index][level] = {
                "mean": mean,
                "std_dev": std_dev,
                "count": count,
                "p25": p25,
                "p75": p75,
            }

    # Create two separate figures: one for distributions and one for statistics
    # Figure 1: ADC Distributions and Sigma Plot
    fig1 = plt.figure(figsize=(20, 16))
    gs1 = GridSpec(2, 1, height_ratios=[1, 1])  # Equal height for both plots

    # Set seaborn style
    sns.set_style("whitegrid")

    # Colors for different levels and IOs
    level_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]
    level_names = ["Level 0", "Level 1", "Level 2", "Level 3"]

    if len(io_list) <= 10:
        io_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    else:
        cmap = plt.cm.get_cmap("tab20", len(io_list))
        io_colors = [cmap(i) for i in range(len(io_list))]

    # =====================================
    # 1. Plot ADC Distribution (top)
    # =====================================
    ax1 = plt.subplot(gs1[0])

    # First, combine data from all IOs for background distribution
    combined_data = {0: [], 1: [], 2: [], 3: []}
    for io_index in io_list:
        for level in range(4):
            if io_index != highlight_io:  # Skip highlight IO for combined data
                combined_data[level].extend(io_level_adc_values[io_index][level])

    # Plot combined data first (background)
    for level in range(4):
        if len(combined_data[level]) > 10:  # Need enough points for KDE
            sns.kdeplot(
                combined_data[level],
                color=level_colors[level],
                alpha=0.3,
                label=f"{level_names[level]} (All IOs)",
                fill=True,
                ax=ax1,
            )

    # Plot highlighted IO if specified
    if highlight_io is not None and highlight_io in io_list:
        for level in range(4):
            data = io_level_adc_values[highlight_io][level]
            if len(data) > 10:  # Need enough points for KDE
                sns.kdeplot(
                    data,
                    color=level_colors[level],
                    linewidth=3,
                    label=f"{level_names[level]} (IO {highlight_io})",
                    ax=ax1,
                )

    # Add read pass thresholds as vertical lines
    for i, threshold in enumerate(read_pass_ranges):
        ax1.axvline(
            x=threshold,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Read Pass Range: {threshold}",
        )
        # Add region labels
        if i == 0:
            ax1.text(
                threshold / 2,
                ax1.get_ylim()[1] * 0.9,
                "Level 0",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        elif i == len(read_pass_ranges) - 1:
            ax1.text(
                (threshold + 64) / 2,
                ax1.get_ylim()[1] * 0.9,
                "Level 3",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        else:
            ax1.text(
                (threshold + read_pass_ranges[i - 1]) / 2,
                ax1.get_ylim()[1] * 0.9,
                f"Level {i}",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )

    # Add vertical lines at means for each level (all IOs)
    for level in range(4):
        # Calculate overall mean for this level across all IOs
        all_values = []
        for io_index in io_list:
            all_values.extend(io_level_adc_values[io_index][level])

        if all_values:
            mean_val = np.mean(all_values)
            ax1.axvline(
                x=mean_val,
                color=level_colors[level],
                linestyle="-",
                alpha=0.8,
                label=f"Mean {level_names[level]}: {mean_val:.1f}",
            )

    # Set labels for distribution plot
    title1 = "ADC Code Distribution"
    if highlight_io is not None:
        title1 += f" (Highlighting IO {highlight_io})"

    ax1.set_xlabel("ADC Value", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    ax1.set_title(title1, fontsize=16)
    ax1.set_xlim(0, 64)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9)

    # =====================================
    # 2. Plot Sigma by Level (bottom)
    # =====================================
    ax2 = plt.subplot(gs1[1])

    # X-axis will be the levels 0, 1, 2, 3
    x = list(range(4))

    # Plot sigma values for each IO
    legend_elements = []

    for i, io_index in enumerate(io_list):
        color = io_colors[i % len(io_colors)]

        # Get sigma values for each level
        sigmas = [io_level_stats[io_index][level]["std_dev"] for level in range(4)]
        counts = [io_level_stats[io_index][level]["count"] for level in range(4)]

        # Plot sigmas - skip levels with no data
        valid_x = []
        valid_sigmas = []

        for j in range(4):
            if counts[j] > 0:
                valid_x.append(j)
                valid_sigmas.append(sigmas[j])

        if valid_x:
            ax2.scatter(
                valid_x, valid_sigmas, color=color, s=100, label=f"IO {io_index}"
            )
            ax2.plot(valid_x, valid_sigmas, color=color, linestyle="-", marker="o")

        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, marker="o", label=f"IO {io_index}")
        )

    # Set labels for sigma plot
    ax2.set_xlabel("Target Level", fontsize=14)
    ax2.set_ylabel("Standard Deviation (Sigma)", fontsize=14)
    ax2.set_title("Standard Deviation of ADC Measurements", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Level 0", "Level 1", "Level 2", "Level 3"])
    ax2.grid(True, alpha=0.3)

    # Add legend with multiple columns if needed
    if len(io_list) > 10:
        ax2.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            ncol=3,
            fontsize=10,
        )
    else:
        ax2.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=12,
        )

    # Adjust layout for first figure
    plt.tight_layout()

    # Save the first plot
    plot_filename1 = "adc_distribution_analysis"
    if highlight_io is not None:
        plot_filename1 += f"_io{highlight_io}"

    plt.savefig(f"{plot_filename1}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_filename1}.svg", bbox_inches="tight")

    # =====================================
    # Create Figure 3: Normalized Sigma Plot (Zero Scale -4 to +4)
    # =====================================
    fig3 = plt.figure(figsize=(18, 10))

    # Setting up the plot
    ax_sigma = plt.subplot(111)

    # Define level boundaries based on read_pass_ranges
    level_boundaries = [0] + read_pass_ranges + [64]

    # Level regions background colors (light pastel colors)
    level_bg_colors = ["#e6f7ff", "#e6ffe6", "#fff9e6", "#ffe6e6"]

    # Draw vertical lines at level boundaries
    for boundary in level_boundaries:
        ax_sigma.axvline(x=boundary, color="black", linestyle="--", alpha=0.5)

    # Add text labels for each level region
    for i in range(4):
        center = (level_boundaries[i] + level_boundaries[i + 1]) / 2
        ax_sigma.text(
            center,
            4.2,
            f"Level {i}",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            fontsize=12,
        )

    # Draw horizontal lines for sigma bands
    for sigma in range(-4, 5):
        if sigma == 0:
            ax_sigma.axhline(y=sigma, color="black", linestyle="-", linewidth=1.5)
        else:
            ax_sigma.axhline(y=sigma, color="gray", linestyle=":", linewidth=0.8)
            ax_sigma.text(
                0.5,
                sigma + 0.1,
                f"{sigma}σ",
                ha="left",
                va="center",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor=None),
            )

    # Map each IO's performance on the plot for each level
    # Create a dictionary to track which IOs we've already added to the legend
    legend_added = {io_index: False for io_index in io_list}

    for level in range(4):
        # Calculate global mean and std for this level across all IOs
        all_values = []
        for io_index in io_list:
            all_values.extend(io_level_adc_values[io_index][level])

        if not all_values:
            continue

        global_mean = np.mean(all_values)
        global_std = np.std(all_values)

        # If global_std is too small (close to zero), use a minimum value to avoid division issues
        if global_std < 0.1:
            global_std = 0.1

        # Plot each IO's data on normalized sigma scale
        for i, io_index in enumerate(io_list):
            # Get mean for this IO at this level
            io_mean = io_level_stats[io_index][level]["mean"]
            count = io_level_stats[io_index][level]["count"]

            if count > 0:
                # Calculate deviation in sigma units
                sigma_deviation = (io_mean - global_mean) / global_std

                # Plot marker - only add to legend once
                color = io_colors[i % len(io_colors)]
                if not legend_added[io_index]:
                    ax_sigma.scatter(
                        io_mean,
                        sigma_deviation,
                        s=100,
                        color=color,
                        marker="o",
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=1,
                        label=f"IO {io_index}",
                    )
                    legend_added[io_index] = True
                else:
                    ax_sigma.scatter(
                        io_mean,
                        sigma_deviation,
                        s=100,
                        color=color,
                        marker="o",
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=1,
                    )

    # Set plot limits and labels
    ax_sigma.set_xlim(0, 64)
    ax_sigma.set_ylim(-4.5, 4.5)
    ax_sigma.set_xlabel("ADC Value", fontsize=14)
    ax_sigma.set_ylabel("Normalized Deviation (σ)", fontsize=14)
    ax_sigma.set_title("Normalized Plot", fontsize=16)

    # Add grid
    ax_sigma.grid(True, alpha=0.3)

    # Create a proper legend with all IOs
    handles, labels = ax_sigma.get_legend_handles_labels()

    # Group legends - first the region labels, then the IO labels
    region_handles = handles[:4]
    region_labels = labels[:4]

    io_handles = [h for h, l in zip(handles, labels) if "IO" in l]
    io_labels = [l for l in labels if "IO" in l]

    # Combine them but keep them grouped logically
    all_handles = region_handles + io_handles
    all_labels = region_labels + io_labels

    # If there are many IOs, use multiple columns for the legend
    ncol = 1
    if len(io_list) > 10:
        ncol = 2
    if len(io_list) > 20:
        ncol = 3

    ax_sigma.legend(
        all_handles, all_labels, loc="upper left", bbox_to_anchor=(1.01, 1), ncol=ncol
    )

    # Adjust layout
    plt.tight_layout()

    # Save the third plot
    plot_filename3 = "adc_normalized_sigma_analysis"
    if highlight_io is not None:
        plot_filename3 += f"_io{highlight_io}"

    plt.savefig(f"{plot_filename3}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_filename3}.svg", bbox_inches="tight")

    elapsed_time = time.time() - start_time
    print(f"\nProcessing and plotting completed in {elapsed_time:.2f} seconds")
    print(f"Plots saved as:")
    print(f"- '{plot_filename1}.png' and '{plot_filename1}.svg'")
    print(f"- '{plot_filename3}.png' and '{plot_filename3}.svg'")


################### Miao's MatLab script in F2MT-115 #######################


def analyze_error_ranges(test_data_path, prng_targets_path, output_dir=None):
    # Create output directory if needed
    if output_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output_dir = f"error_analysis_{current_time}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    test_data = np.load(test_data_path)
    prng_target = np.load(prng_targets_path)

    # Define ranges
    lv_write_range = np.array([[0, 4], [28, 34], [48, 54], [62, 63]])
    lv_read_range = np.array([14, 42, 60])

    # Initialize error arrays
    error_write_range = np.zeros((78, 64, 1296))
    error_read_range = np.zeros((78, 64, 1296))

    # Calculate errors
    for i in range(78):
        for j in range(64):
            for k in range(1296):
                temp1 = test_data[i, j, k]
                temp2 = prng_target[i, j, k]

                # Write range error check
                if temp1 < lv_write_range[temp2, 0] or temp1 > lv_write_range[temp2, 1]:
                    error_write_range[i, j, k] = 1

                # Read range error check
                if temp2 == 0:
                    if temp1 >= lv_read_range[0]:
                        error_read_range[i, j, k] = 1
                elif temp2 == 1:
                    if temp1 >= lv_read_range[1] or temp1 < lv_read_range[0]:
                        error_read_range[i, j, k] = 1
                elif temp2 == 2:
                    if temp1 >= lv_read_range[2] or temp1 < lv_read_range[1]:
                        error_read_range[i, j, k] = 1
                elif temp2 == 3:
                    if temp1 < lv_read_range[2]:
                        error_read_range[i, j, k] = 1

    # Calculate per IO errors
    error_write_per_IO = np.sum(error_write_range, axis=(1, 2))
    error_read_per_IO = np.sum(error_read_range, axis=(1, 2))

    # Plot 1: Error per IO
    plt.figure(figsize=(10, 6))
    plt.plot(range(78), error_write_per_IO, "-o", label="write range error")
    plt.plot(range(78), error_read_per_IO, "-s", label="read range error")
    plt.legend()
    plt.xlabel("IO index")
    plt.ylabel("Bit error")
    plt.savefig(f"{output_dir}/error_per_io.png")
    plt.close()

    # Plot 2: Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(error_write_per_IO, error_read_per_IO)
    plt.xlabel("Write range error")
    plt.ylabel("Read range error (BER)")
    plt.savefig(f"{output_dir}/error_scatter.png")
    plt.close()

    # Calculate per WL errors
    error_write_per_wl_total = np.zeros(1296)
    error_read_per_wl_total = np.zeros(1296)
    error_write_per_16bl_per_io_store = np.zeros((78, 16))
    error_read_per_16bl_per_io_store = np.zeros((78, 16))

    # Create figure for BL errors
    plt.figure(figsize=(12, 8))

    for i in range(78):
        error_write_range_per_io = error_write_range[i]
        error_read_range_per_io = error_read_range[i]

        error_write_per_wl_per_io = np.sum(error_write_range_per_io, axis=0)
        error_write_per_bl_per_io = np.sum(error_write_range_per_io, axis=1)
        error_read_per_wl_per_io = np.sum(error_read_range_per_io, axis=0)
        error_read_per_bl_per_io = np.sum(error_read_range_per_io, axis=1)

        # Calculate 16BL groupings
        error_write_per_16bl_per_io = np.sum(
            error_write_per_bl_per_io.reshape(4, 16), axis=0
        )
        error_read_per_16bl_per_io = np.sum(
            error_read_per_bl_per_io.reshape(4, 16), axis=0
        )

        error_write_per_wl_total += error_write_per_wl_per_io
        error_read_per_wl_total += error_read_per_wl_per_io

        error_write_per_16bl_per_io_store[i] = error_write_per_16bl_per_io
        error_read_per_16bl_per_io_store[i] = error_read_per_16bl_per_io

        # Plot normalized errors
        plt.plot(
            range(16),
            error_write_per_16bl_per_io - np.mean(error_write_per_16bl_per_io),
            label=f"IO {i}",
        )

    plt.xlabel("BL index")
    plt.ylabel("Bit error write range (norm)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bl_errors.png")
    plt.close()

    # Plot total errors per WL
    plt.figure(figsize=(12, 6))
    plt.plot(range(1296), error_write_per_wl_total, label="Write errors")
    plt.plot(range(1296), error_read_per_wl_total, label="Read errors")
    plt.xlabel("WL index")
    plt.ylabel("Write/Read BER per WL")
    plt.legend()
    plt.savefig(f"{output_dir}/total_wl_errors.png")
    plt.close()

    # Calculate and plot similarity matrices
    def plot_similarity_matrix(data, metric_name, filename):
        plt.figure(figsize=(10, 8))
        plt.imshow(data)
        plt.colorbar()
        plt.title(f"{metric_name} Similarity Matrix")
        plt.savefig(filename)
        plt.close()

    # Cosine similarity for write errors
    cosine_matrix_write = np.zeros((78, 78))
    for i in range(78):
        for j in range(78):
            vec1 = error_write_per_16bl_per_io_store[i]
            vec2 = error_write_per_16bl_per_io_store[j]
            cosine_matrix_write[i, j] = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

    plot_similarity_matrix(
        cosine_matrix_write, "Cosine", f"{output_dir}/cosine_similarity_write.png"
    )

    # Cosine similarity for read errors
    cosine_matrix_read = np.zeros((78, 78))
    for i in range(78):
        for j in range(78):
            vec1 = error_read_per_16bl_per_io_store[i]
            vec2 = error_read_per_16bl_per_io_store[j]
            cosine_matrix_read[i, j] = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

    plot_similarity_matrix(
        cosine_matrix_read, "Cosine", f"{output_dir}/cosine_similarity_read.png"
    )

    # Correlation matrix
    correlation_matrix = np.zeros((78, 78))
    for i in range(78):
        for j in range(78):
            correlation_matrix[i, j] = np.corrcoef(
                error_write_per_16bl_per_io_store[i],
                error_write_per_16bl_per_io_store[j],
            )[0, 1]

    plot_similarity_matrix(
        correlation_matrix, "Correlation", f"{output_dir}/correlation_matrix.png"
    )

    # Create subplots for individual IO patterns
    for i in range(0, 78, 10):
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.ravel()

        for j in range(10):
            if i + j < 78:
                data = error_write_per_16bl_per_io_store[i + j]
                axs[j].plot(data - np.mean(data))
                axs[j].set_xlabel("BL index")
                axs[j].set_ylabel("Write range bit error (norm)")
                axs[j].set_title(f"IO {i + j}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/io_patterns_{i:02d}-{min(i+9,77):02d}.png")
        plt.close()

    return error_write_range, error_read_range


def elapsed(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_struct = time.gmtime(elapsed_time)
    formatted_time = time.strftime(
        "%H hours, %M minutes, and %S seconds", elapsed_time_struct
    )
    print(f"Test completed : {formatted_time}")


async def wait_for_soc_flag(
    backend_driver, expected_flag: int, timeout: int = 30
) -> int:
    """Wait for specific SoC flag"""
    start_time = time.time()
    while True:
        if (time.time() - start_time) > timeout:
            raise TimeoutError(f"Timeout waiting for SoC flag {expected_flag}")

        flag = backend_driver.spi.soc_status_get(
            backend_driver.spi.DUMMY_DELAY_TIME
        )
        if flag == expected_flag:
            return flag
        await anyio.sleep(0.001)


async def signal_soc(backend_driver):
    """Signal operation complete with flag 3"""
    status = await backend_driver.spi.soc_status_set_safe(
        SocFlags.READY_FOR_NEXT, backend_driver.spi.DUMMY_DELAY_TIME
    )
    if status != SocFlags.READY_FOR_NEXT:
        raise RuntimeError(f"Failed to set completion flag. Got {status}")


async def execute_operation(
    backend_driver,
    test_id: int,
    params: np.ndarray,
    read_response: bool = True,
    response_size: int = 8,
    read_dst_addr: Optional[int] = None,
) -> Optional[np.ndarray]:

    vars_start = 0x10000000
    try:

        await wait_for_soc_flag(backend_driver, SocFlags.SOC_READY)
        backend_driver.spi.spi_burst_write(
            vars_start,
            data=params,
            delay=backend_driver.spi.SPI_BURST_WRITE_DELAY,
            data_type="u8",
        )

        await signal_soc(backend_driver)
        await wait_for_soc_flag(backend_driver, SocFlags.OPERATION_COMPLETE)

        if read_response:
            read_addr = (
                read_dst_addr if read_dst_addr is not None else vars_start + 0x10000
            )
            response = backend_driver.spi.spi_burst_read(
                read_addr,
                data_size=response_size,
                delay=backend_driver.spi.DUMMY_DELAY_TIME,
                data_type="u32",
            )
            return response
        return None
    finally:
        pass


async def send_agate_soc_set_command_with_retry(backend_driver, value, timeout=10):
    command = backend_driver.spi.SET_STATUS_CMD
    ack_command = backend_driver.spi.SET_STATUS_ACK_CMD
    start_time = time.time()
    status = 0

    while time.time() - start_time < timeout:
        status = backend_driver.spi.agate_soc_status_set_safe(
            value, backend_driver.spi.DUMMY_DELAY_TIME
        )

        # check if response when not None contains set value
        if status is not None:
            if status == value:
                return status
            else:
                # bad value retry
                print(f"{status}!={value} Retrying Set->{value}")
                backend_driver.spi.reset_device()
                await anyio.sleep(0.3)  # Changed from time.sleep
                continue
        # bad status
        else:
            print(f"bad {status} Retrying Set->{value}")
            backend_driver.spi.reset_device()
            await anyio.sleep(0.1)  # Changed from time.sleep
            continue

    return status


async def send_agate_soc_get_command_with_retry(backend_driver, value, timeout=10):
    command = backend_driver.spi.GET_STATUS_CMD
    ack_command = backend_driver.spi.GET_STATUS_ACK_CMD
    start_time = time.time()
    status = 0

    while time.time() - start_time < timeout:
        # command is a get, so ensure we got a good status
        # back from the ack
        status = backend_driver.spi.soc_status_get(
            backend_driver.spi.DUMMY_DELAY_TIME
        )

        # check if response contains value
        if status is None:
            print(f"Retrying Get <- {time.time() - start_time} {timeout}")
            # bad status or retry
            backend_driver.spi.reset_device()
            await anyio.sleep(0.3)  # Changed from time.sleep to anyio.sleep
            continue
        else:
            return status

    return status


async def wait_for_status_reg_flag_handshake(backend_driver, timeout, wait=None):
    SoC_flag = 3
    start = time.time()

    while SoC_flag == 3:
        if (time.time() - start) < timeout:
            SoC_flag = await send_agate_soc_get_command_with_retry(
                backend_driver, 0, timeout
            )

            if SoC_flag is None:
                raise Exception("Soc_Flag is None")

            if wait is None:
                await anyio.sleep(0.001)  # Using anyio.sleep instead
            else:
                wait()  # Make sure this wait function is async if it involves I/O
        else:
            raise Exception("timeout")

    return SoC_flag


async def switch_to_user_flow(backend_driver):

    # switch to user-flow
    ###########################################################################
    backend_driver.host_soc_cmd.pass_control_to_soc()
    send_pkt = np.zeros(2, dtype=np.uint32)
    send_pkt[0] = backend_driver.host_soc_cmd.PY_TO_SOC_USER_FLOW_OP & 0xFFFF
    backend_driver.host_soc_cmd.write_mem(
        backend_driver.host_soc_cmd.PY_TO_SOC_CMD_ADDR, send_pkt
    )
    backend_driver.host_soc_cmd.wait_for_soc_to_respond(
        timeout=100,
        done_cmd=backend_driver.host_soc_cmd.SOC_TO_PY_USER_FLOW_OP_ACK,
        debug=False,
    )
    ###########################################################################


async def switch_to_test_flow(backend_driver):

    ###########################################################################
    await signal_soc(backend_driver)
    ###########################################################################


def set_blref_calibration_group(backend_driver, ios, blref_value, macro):
    for io in ios:
        backend_driver.agate_reg.set_calbits_sequence_by_name(
            macro, "BLREF_CAL", blref_value, debug_print=False
        )


def determine_base_vcmidac_group(vcm_idac_input, vcmidac_basegroup):
    # Determine delta between vcm_idac_input and the highest vcmidac in the basegroup
    delta = vcm_idac_input - vcmidac_basegroup[1]
    offset = 2 * int(delta / 2)

    # Offset from the highest vcmidac in the basegroup
    bucketed_vcmidac = vcmidac_basegroup[1] + offset
    return bucketed_vcmidac


def get_io_groups(filename, io_test_list=None, enable_vcm_grouping=False):
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        grouped_blref = defaultdict(list)
        # grouped_vcmidac = defaultdict(list)
        vcmidac_list = []
        for key, value in data.items():
            if "BLREF_CAL_BL_FB_io" in key and "voltage" not in key:
                io_num = int(key.split("io")[-1])
                grouped_blref[value].append(io_num)
            if "WINCAL_VCM_IDAC_" in key and enable_vcm_grouping:
                # io_num = int(key.split("io")[-1])
                # grouped_vcmidac[value].append(io_num)
                vcmidac_list.append(value)

        if enable_vcm_grouping:
            # Find the mean VCM setting and take the floor and ceiling integers as the base group
            vcmidac_list.sort()
            vcmidac_list = np.array(vcmidac_list)
            mean_vcmidac = vcmidac_list.mean()
            vcmidac_basegroup = (int(mean_vcmidac), int(mean_vcmidac + 1))

            # For each BLREF group bucket out each group by VCM
            grouped_blref_vcmidac = defaultdict(list)
            for blref, io_sublist in grouped_blref.items():
                for io in io_sublist:
                    wincal_vcmidac = data.get(f"WINCAL_VCM_IDAC_io{io}", -1)
                    # If window calibration data exists for this IO bucket accordingly
                    if wincal_vcmidac >= 0:
                        # Determine the correct base group for each io
                        vcmidac = determine_base_vcmidac_group(
                            wincal_vcmidac, vcmidac_basegroup
                        )
                    # Else use the default VCM (higher VCM_IDAC of the basegroup)
                    else:
                        vcmidac = vcmidac_list[1]
                    # Save IO in dictionary with tupple of (BLREF, VCM_IDAC)
                    grouped_blref_vcmidac[(blref, vcmidac)].append(io)

            # Convert from dictionary to list format
            result_allios = []
            for (blref_value, vcmidac_value), ios in grouped_blref_vcmidac.items():
                ios.sort()
                result_allios.append([ios, (blref_value, vcmidac_value)])

            result_allios.sort(key=lambda x: len(x[0]), reverse=True)

            # Only keep io groups with IOs that are to be tested
            if io_test_list is not None:
                result = []
                for i, (io_group, (blref_value, vcmidac_value)) in enumerate(
                    result_allios
                ):
                    intersection = list(set(io_group) & set(io_test_list))
                    intersection.sort()
                    if len(intersection) > 0:
                        result.append([intersection, (blref_value, vcmidac_value)])
            else:
                result = result_allios

        # No VCM_IDAC grouping
        else:
            result_allios = []
            for blref_value, ios in grouped_blref.items():
                ios.sort()
                result_allios.append([ios, blref_value])

            result_allios.sort(key=lambda x: len(x[0]), reverse=True)

            # Only keep io groups with IOs that are to be tested
            if io_test_list is not None:
                result = []
                for i, (io_group, blref_value) in enumerate(result_allios):
                    intersection = list(set(io_group) & set(io_test_list))
                    intersection.sort()
                    if len(intersection) > 0:
                        result.append([intersection, (blref_value, -1)])
            else:
                result = result_allios

        return result

    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {filename}: {e}")
        return None
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None


#########################################################################################################
# Invoked from test_automation_user.py as well as pytest
#########################################################################################################


async def read_io_dma(
    backend_driver,
    io_index,
    macro_index,
    filename=None,
):
    vars_start = 0x10000000
    vars_dst = 0x10080000
    READ_SIZE = 64 * 1296
    SPI_READ_SIZE = READ_SIZE // 4  # For uint32 reads
    SoC_flag = None
    sync_timeout = 10
    np.set_printoptions(threshold=np.inf)

    io_read_linear = np.zeros(READ_SIZE, dtype=np.uint8)
    io_read_full = np.zeros((1, 64, 1296), dtype=np.uint8)
    SoC_flag = await wait_for_status_reg_flag_handshake(backend_driver, sync_timeout)

    print(f"Reading macro {macro_index} io {io_index}")
    SoC_flag = None
    try:
        while True:
            if SoC_flag is not None:
                await signal_soc(backend_driver)

            SoC_flag = await wait_for_status_reg_flag_handshake(
                backend_driver, sync_timeout
            )

            if SoC_flag == SocFlags.SOC_READY:
                print(
                    f"Ready begin reading macro {macro_index&0xFF} io {io_index&0xFF}"
                )
                params = [111, 0, io_index & 0xFF, macro_index & 0xFF]
                params[1] = len(params) - 2
                backend_driver.spi.spi_burst_write(
                    vars_start,
                    data=np.array(params, dtype=np.uint8),
                    delay=backend_driver.spi.SPI_BURST_WRITE_DELAY,
                    data_type="u8",
                )
                start_time = time.time()
                continue

            if SoC_flag == SocFlags.OPERATION_COMPLETE:
                print(f"Rx soc_flag {SoC_flag}")
                current_read = backend_driver.spi.spi_burst_read(
                    vars_dst,
                    data_size=SPI_READ_SIZE,
                    delay=backend_driver.spi.DUMMY_DELAY_TIME,
                    data_type="u32",
                )
                current_read_bytes = current_read.view(np.uint8)
                if len(current_read_bytes) == READ_SIZE:
                    io_read_linear = current_read_bytes
                else:
                    print(
                        f"Size mismatch: expected {READ_SIZE}, got {len(current_read_bytes)}"
                    )
                    break

                print(f"Read completed: {READ_SIZE} bytes")
                elapsed(start_time)

                if filename:
                    with open(filename, "wb") as f:
                        f.write(io_read_linear.tobytes())

                io_read_full = io_read_linear.reshape(1, 64, 1296)
                print(f"Final array shape: {io_read_full.shape}")
                return io_read_full

    except Exception as e:
        print(f"Test failed with error: {str(e)}")


async def get_ft_history():

    with open(
        os.path.join(
            tests_dir,
            "agate",
            "dev-settings",
            "json",
            "soc_op_params",
            "setv_forming_parameters.json",
        ),
        "r",
    ) as file:
        ft_params = json.load(file)

    if not ft_params:
        raise ValueError("Failed to load setv_forming_parameters.json")

    enable_ft_history = ft_params.get("ft_history", True)

    return enable_ft_history


async def get_ft_history_setv():

    with open(
        os.path.join(
            tests_dir,
            "agate",
            "dev-settings",
            "json",
            "soc_op_params",
            "setv_finetune_parameters.json",
        ),
        "r",
    ) as file:
        ft_params = json.load(file)

    if not ft_params:
        raise ValueError("Failed to load setv_forming_parameters.json")

    enable_ft_history = ft_params.get("ft_history", True)

    return enable_ft_history


async def run_parallel_form(
    backend_driver,
    io_indices: List[int],
    bl_list: List[int],
    wl_list: List[int],
    macro_index: int,
    level: int,
    total_target_values: int = 0,
    nn_model: bool = False,
    pll=191,
    delayed_reset=0,
    read_dst_addr: Optional[int] = None,
) -> np.ndarray:
    """Run parallel formingwith setv_forming parameters from JSON"""

    # Load fine-tune parameters
    with open(
        os.path.join(
            tests_dir,
            "agate",
            "dev-settings",
            "json",
            "soc_op_params",
            "setv_forming_parameters.json",
        ),
        "r",
    ) as file:
        ft_params = json.load(file)

    if not ft_params:
        raise ValueError("Failed to load setv_forming_parameters.json")

    # Save new JSON file
    # Build configuration to match SetvFinetuneParams struct
    config = {
        # Operation parameters first
        "operation": {
            "io_indices": io_indices,
            "bl_list": bl_list,
            "wl_list": wl_list,
            "macro_index": macro_index,
            "level": level,
            "total_target_values": total_target_values,
            "pll_freq_code": pll,
            # this parameter controls reset behavior after n cells
            # it is a uint8_t. setting to 255 means form 256 WL then reset
            "delayed_reset": delayed_reset,
        },
        "finetune_params": ft_params,
    }

    # Convert configuration to JSON string
    config_json = json.dumps(config)
    config_bytes = config_json.encode("utf-8")
    config_len = len(config_bytes)

    """
         Byte Layout:
        - byte[0]: test_id (110/111) for routing
        - byte[1]: format flag (0x1)
        - bytes[2-5]: JSON length (32-bit LE)
        - bytes[6+]: JSON data
        """

    # Construct parameter array
    params = []

    # First byte is always test_id for routing
    test_id = 112
    params = [test_id]

    # Add JSON flag and length
    params.extend(
        [
            0x1,  # JSON flag to indicate JSON format follows
            config_len & 0xFF,
            (config_len >> 8) & 0xFF,
            (config_len >> 16) & 0xFF,
            (config_len >> 24) & 0xFF,
        ]
    )

    # Add JSON bytes
    params.extend(list(config_bytes))

    # For debugging
    if os.environ.get("DEBUG"):
        print("Sending configuration:")
        print(params)

    # Execute operation
    await execute_operation(
        backend_driver,
        test_id=params[0],
        params=np.array(params, dtype=np.uint8),
        read_response=True,
        response_size=20,
        read_dst_addr=read_dst_addr,
    )


async def run_parallel_finetune(
    backend_driver,
    io_indices: List[int],
    bl_list: List[int],
    wl_list: List[int],
    macro_index: int,
    level: int,
    total_target_values: int = 0,
    nn_model: bool = False,
    pll=192,
    read_dst_addr: Optional[int] = None,
    param_json_file: Optional[str] = None,
    param_overrides: Optional[dict] = None,
    delayed_reset: int = 0,
) -> np.ndarray:
    """Run parallel fine-tuning with setv_finetune parameters from JSON"""

    # Load fine-tune parameters
    if param_json_file is None:
        param_json_file = "setv_finetune_parameters.json"
    else:
        print(f" - - Loading this json file for finetune params: {param_json_file}")

    with open(
        os.path.join(
            tests_dir,
            "agate",
            "dev-settings",
            "json",
            "soc_op_params",
            param_json_file,
        ),
        "r",
    ) as file:
        ft_params = json.load(file)

    if not ft_params:
        raise ValueError(f"Failed to load {param_json_file}")

    # DO manual ft param overrides
    if param_overrides is not None:
        for param_key, param_value in param_overrides.items():
            if param_key not in ft_params.keys():
                raise Exception(f"{param_key} not a ft parameter")
            else:
                ft_params[param_key] = param_value
                print(f" - - Overriding {param_key} with {param_value}")

    # Save new JSON file
    # Build configuration to match SetvFinetuneParams struct
    config = {
        # Operation parameters first
        "operation": {
            "io_indices": io_indices,
            "bl_list": bl_list,
            "wl_list": wl_list,
            "macro_index": macro_index,
            "level": level,
            "total_target_values": total_target_values,
            "pll_freq_code": pll,
            "delayed_reset": delayed_reset,
        },
        "finetune_params": ft_params,
    }

    # Convert configuration to JSON string
    config_json = json.dumps(config)
    config_bytes = config_json.encode("utf-8")
    config_len = len(config_bytes)

    """
         Byte Layout:
        - byte[0]: test_id (110/111) for routing
        - byte[1]: format flag (0x1)
        - bytes[2-5]: JSON length (32-bit LE)
        - bytes[6+]: JSON data
        """

    # Construct parameter array
    params = []

    # First byte is always test_id for routing
    test_id = 112
    params = [test_id]

    # Add JSON flag and length
    params.extend(
        [
            0x1,  # JSON flag to indicate JSON format follows
            config_len & 0xFF,
            (config_len >> 8) & 0xFF,
            (config_len >> 16) & 0xFF,
            (config_len >> 24) & 0xFF,
        ]
    )

    # Add JSON bytes
    params.extend(list(config_bytes))

    # For debugging
    if os.environ.get("DEBUG"):
        print("Sending configuration:")
        print(params)

    # Execute operation
    await execute_operation(
        backend_driver,
        test_id=params[0],
        params=np.array(params, dtype=np.uint8),
        read_response=True,
        response_size=20,
        read_dst_addr=read_dst_addr,
    )


async def run_parallel_set_reset(
    backend_driver,
    io_indices: List[int],
    bl_list: List[int],
    wl_list: List[int],
    macro_index: int,
    flag: int,
    reset_sl: int,
    reset_pw: int,
    set_wl: int,
    set_bl: int,
    set_pw: int,
    read_dst_addr: Optional[int] = None,
) -> np.ndarray:
    """Run parallel set_reset"""

    # Save new JSON file
    # Build configuration to match SetvFinetuneParams struct
    config = {
        # Operation parameters first
        "operation": {
            "io_indices": io_indices,
            "bl_list": bl_list,
            "wl_list": wl_list,
            "macro_index": macro_index,
            "flag": flag,
            "reset_sl": reset_sl,
            "reset_pw": reset_pw,
            "set_wl": set_wl,
            "set_bl": set_bl,
            "set_pw": set_pw,
        },
    }

    # Convert configuration to JSON string
    config_json = json.dumps(config)
    config_bytes = config_json.encode("utf-8")
    config_len = len(config_bytes)

    """
         Byte Layout:
        - byte[0]: test_id (110/111) for routing
        - byte[1]: format flag (0x1)
        - bytes[2-5]: JSON length (32-bit LE)
        - bytes[6+]: JSON data
        """

    # Construct parameter array
    params = []

    # First byte is always test_id for routing
    test_id = 115
    params = [test_id]

    # Add JSON flag and length
    params.extend(
        [
            0x1,  # JSON flag to indicate JSON format follows
            config_len & 0xFF,
            (config_len >> 8) & 0xFF,
            (config_len >> 16) & 0xFF,
            (config_len >> 24) & 0xFF,
        ]
    )

    # Add JSON bytes
    params.extend(list(config_bytes))

    # For debugging
    if os.environ.get("DEBUG"):
        print("Sending configuration:")
        print(params)

    # Execute operation
    await execute_operation(
        backend_driver,
        test_id=params[0],
        params=np.array(params, dtype=np.uint8),
        read_response=True,
        response_size=20,
        read_dst_addr=read_dst_addr,
    )


async def run_read_subarray_dma(
    backend_driver,
    io_indices: List[int],
    bl_list: List[int],
    wl_list: List[int],
    macro_index: int,
    read_dst_addr: Optional[int] = None,
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    plot: Optional[bool] = False,
) -> np.ndarray:
    """Run read subarrays of specifc IO/BL/WL"""

    # Save new JSON file
    config = {
        # Operation parameters first
        "operation": {
            "io_indices": io_indices,
            "bl_list": bl_list,
            "wl_list": wl_list,
            "macro_index": macro_index,
        },
    }

    # Convert configuration to JSON string
    config_json = json.dumps(config)
    config_bytes = config_json.encode("utf-8")
    config_len = len(config_bytes)

    """
         Byte Layout:
        - byte[0]: test_id (110/111) for routing
        - byte[1]: format flag (0x1)
        - bytes[2-5]: JSON length (32-bit LE)
        - bytes[6+]: JSON data
        """

    # Construct parameter array
    params = []

    # First byte is always test_id for routing
    test_id = 114
    params = [test_id]

    # Add JSON flag and length
    params.extend(
        [
            0x1,  # JSON flag to indicate JSON format follows
            config_len & 0xFF,
            (config_len >> 8) & 0xFF,
            (config_len >> 16) & 0xFF,
            (config_len >> 24) & 0xFF,
        ]
    )

    params.extend(list(config_bytes))

    vars_start = 0x10000000
    vars_dst = 0x10080000
    READ_SIZE = 64 * 1296
    SPI_READ_SIZE = READ_SIZE // 4  # For uint32 reads
    SoC_flag = None
    sync_timeout = 10
    np.set_printoptions(threshold=np.inf)

    io_read_linear = np.zeros(READ_SIZE, dtype=np.uint8)
    io_read_full = np.zeros((1, 64, 1296), dtype=np.uint8)
    SoC_flag = await wait_for_status_reg_flag_handshake(backend_driver, sync_timeout)

    print(f"Reading macro {macro_index} io {io_indices}")
    SoC_flag = None
    # save_dir = None
    index = 0
    saved_files = []
    saved_dirs = []
    lock_not_acquired = False
    try:
        while True:
            if SoC_flag is not None:
                try:
                    await signal_soc(backend_driver)
                except:
                    print("Could not acquire lock!!")
                    if SoC_flag in [SocFlags.OPERATION_COMPLETE]:
                        lock_not_acquired = True
                        break

            SoC_flag = await wait_for_status_reg_flag_handshake(
                backend_driver, sync_timeout
            )

            if SoC_flag == SocFlags.SOC_READY:
                print(f"Rx soc_flag {SoC_flag}")
                print(f"Ready begin reading macro {macro_index&0xFF}")
                backend_driver.spi.spi_burst_write(
                    vars_start,
                    data=np.array(params, dtype=np.uint8),
                    delay=backend_driver.spi.SPI_BURST_WRITE_DELAY,
                    data_type="u8",
                )
                start_time = time.time()
                continue

            if SoC_flag == SocFlags.READ_START:
                print(f"Rx soc_flag {SoC_flag}")
                current_read = backend_driver.spi.spi_burst_read(
                    vars_dst,
                    data_size=SPI_READ_SIZE,
                    delay=backend_driver.spi.DUMMY_DELAY_TIME,
                    data_type="u32",
                )
                current_read_bytes = current_read.view(np.uint8)
                if len(current_read_bytes) == READ_SIZE:
                    io_read_linear = current_read_bytes
                else:
                    print(
                        f"Size mismatch: expected {READ_SIZE}, got {len(current_read_bytes)}"
                    )
                    break

                elapsed(start_time)
                io_index = io_indices[index]
                print(f"Read completed for IO {io_index}: {READ_SIZE} bytes")
                io_read_full = io_read_linear.reshape(1, 64, 1296)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if file_name is None:
                    npy_filename = f"IO{io_index}.npy"
                else:
                    npy_filename = file_name.replace('IOIDX',str(io_index)) + '.npy'
                full_path = os.path.join(save_dir, npy_filename)
                np.save(full_path, io_read_full)
                # Store information for later plotting
                saved_files.append(
                    {
                        "io_index": io_index,
                        "npy_filename": npy_filename,
                        "full_path": full_path,
                        "io_read_full": io_read_full,
                    }
                )
                index += 1

            if SoC_flag == SocFlags.OPERATION_COMPLETE:
                print("subarray read completed")
                break

    except Exception as e:
        print(f"Test failed with error: {str(e)}")

    if plot:
        for saved_data in saved_files:
            io_index = saved_data["io_index"]
            npy_filename = saved_data["npy_filename"]
            io_read_full = saved_data["io_read_full"]

            save_dir = backend_driver.plot_ops.plot_adc_io_subarray(
                io_read_full,
                bl_list,
                wl_list,
                save_directory=save_dir,  # This might need to be defined outside the loop
                macro=macro_index,
                mode=6,
                filename=npy_filename,
            )

            saved_dirs.append(save_dir)
            print(f"Plotted IO {io_index}, saved to directory: {save_dir}")


def dump_calbit_registers(backend_driver, macro):
    calbit_names = [
        "CWRN_OVERRIDE",
        "ADC_VCM_IDAC",
        "ADC_VCM_RDAC",
        "ADC_VREF_IDAC",
        "ADC_VREF_RDAC",
        "BLDRV_BL_R",
        "BLREF_CAL",
        "LDO_VREF",
    ]

    print("Dumping calbit register values after test:")

    for calbit_name in calbit_names:
        try:
            out_num = backend_driver.agate_reg.read_calbits_sequence_by_name(
                macro, calbit_name, debug_print=False
            )
            print(f"READING CALBIT {calbit_name} = {out_num}")
        except Exception as e:
            print(f"Failed to read {calbit_name}: {e}")
