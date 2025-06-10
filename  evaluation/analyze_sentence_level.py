# evaluate_emotion.py
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_audio(wav_path, output_dir):
    snd = parselmouth.Sound(wav_path)

    # F0 (pitch)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values[f0_values == 0] = np.nan

    # Energy (intensity)
    intensity = snd.to_intensity()
    energy_values = intensity.values.T.squeeze()

    # Statistics
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

    print(f"📊 F0 Stats: {f0_stats}")
    print(f"📊 Energy Stats: {energy_stats}")

    # Plots
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(f0_values)
    plt.title("F0 Contour")
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f0_contour.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(energy_values, color="purple")
    plt.title("Energy Contour")
    plt.xlabel("Frame")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_contour.png"))
    plt.close()

if __name__ == "__main__":
    wav_path = "200000.wav"
    output_dir = "emotion_eval_200000"
    analyze_audio(wav_path, output_dir)