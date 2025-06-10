# evaluate_emotion_compare.py
import os
import parselmouth
import numpy as np
import matplotlib.pyplot as plt

def extract_stats(wav_path):
    snd = parselmouth.Sound(wav_path)

    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values[f0_values == 0] = np.nan

    intensity = snd.to_intensity()
    energy_values = intensity.values.T.squeeze()

    f0_stats = {
        "mean": np.nanmean(f0_values),
        "std": np.nanstd(f0_values),
        "min": np.nanmin(f0_values),
        "max": np.nanmax(f0_values)
    }
    energy_stats = {
        "mean": np.nanmean(energy_values),
        "std": np.nanstd(energy_values),
        "min": np.nanmin(energy_values),
        "max": np.nanmax(energy_values)
    }
    return f0_stats, energy_stats

def plot_comparison(steps, f0_means, f0_stds, energy_means, energy_stds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # F0 mean/std
    plt.figure()
    plt.errorbar(steps, f0_means, yerr=f0_stds, fmt='-o', capsize=5, label="F0 (Hz)")
    plt.title("F0 Mean and Std over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f0_over_steps.png"))
    plt.close()

    # Energy mean/std
    plt.figure()
    plt.errorbar(steps, energy_means, yerr=energy_stds, fmt='-o', capsize=5, color="purple", label="Energy")
    plt.title("Energy Mean and Std over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_over_steps.png"))
    plt.close()

if __name__ == "__main__":
    steps = [200002,20000]
    wav_files = [f"{step}.wav" for step in steps]
    output_dir = "emotion_eval_compare"

    f0_means, f0_stds = [], []
    energy_means, energy_stds = [], []

    for wav_file, step in zip(wav_files, steps):
        if not os.path.exists(wav_file):
            print(f"[WARNING] {wav_file} not found, skipping.")
            continue
        f0_stats, energy_stats = extract_stats(wav_file)
        print(f"\n📌 Step {step}")
        print(f"📊 F0: {f0_stats}")
        print(f"📊 Energy: {energy_stats}")
        f0_means.append(f0_stats["mean"])
        f0_stds.append(f0_stats["std"])
        energy_means.append(energy_stats["mean"])
        energy_stds.append(energy_stats["std"])

    plot_comparison(steps, f0_means, f0_stds, energy_means, energy_stds, output_dir)