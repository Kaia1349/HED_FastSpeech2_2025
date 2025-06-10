import os
import numpy as np
import pandas as pd
import parselmouth
import matplotlib.pyplot as plt
from textgrid import TextGrid
from functools import reduce

def get_phoneme_segments(textgrid_path):
    tg = TextGrid.fromFile(textgrid_path)
    phone_tier = [t for t in tg.tiers if 'phone' in t.name.lower()][0]
    segments = []
    for interval in phone_tier:
        phoneme = interval.mark.strip().lower()
        if phoneme in ['', '<eps>', 'sil']:
            continue
        segments.append({
            'phoneme': phoneme,
            'start': interval.minTime,
            'end': interval.maxTime
        })
    return segments

def extract_f0(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=0.005)
    f0_values = []
    times = []
    for i in range(pitch.get_number_of_frames()):
        f0 = pitch.get_value_in_frame(i)
        time = pitch.get_time_from_frame_number(i + 1)
        if f0 is not None:
            f0_values.append(f0)
            times.append(time)
    return np.array(times), np.array(f0_values)

def get_f0_stats_per_phoneme(times, f0_values, segments, label):
    stats = []
    for seg in segments:
        mask = (times >= seg['start']) & (times <= seg['end'])
        f0_seg = f0_values[mask]
        if len(f0_seg) > 0:
            f0_mean = np.mean(f0_seg)
            f0_std = np.std(f0_seg)
        else:
            f0_mean = f0_std = np.nan
        stats.append({
            'phoneme': seg['phoneme'],
            'start': seg['start'],
            'end': seg['end'],
            f'f0_mean_{label}': f0_mean,
            f'f0_std_{label}': f0_std
        })
    return pd.DataFrame(stats)

def draw_f0_plot(times, f0_values, segments, label, save_path):
    if len(f0_values) == 0 or np.all(np.isnan(f0_values)):
        print(f"[Warning] Skipping plot for {label} — all F0 values are NaN.")
        return
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0_values, label="F0", color='blue')
    for seg in segments:
        try:
            plt.axvspan(seg['start'], seg['end'], alpha=0.2, color='orange')
            mid = (seg['start'] + seg['end']) / 2
            plt.text(mid, np.nanmax(f0_values) * 0.95, seg['phoneme'], ha='center', va='top', fontsize=8)
        except Exception:
            continue
    plt.title(f"F0 Curve with Phoneme Boundaries: {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_multiple_conditions(wav_paths, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []
    for wav_path, label in zip(wav_paths, labels):
        textgrid_path = wav_path.replace(".wav", ".TextGrid")
        if not os.path.exists(textgrid_path):
            print(f"[Error] Missing TextGrid file: {textgrid_path}")
            continue
        segments = get_phoneme_segments(textgrid_path)
        times, f0_values = extract_f0(wav_path)
        df = get_f0_stats_per_phoneme(times, f0_values, segments, label)
        all_dfs.append(df)
        draw_f0_plot(times, f0_values, segments, label, os.path.join(output_dir, f"F0_{label}.png"))

    if all_dfs:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=["phoneme", "start", "end"], how="outer"), all_dfs)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        merged_df.to_csv(os.path.join(output_dir, "multi_condition_f0_analysis.csv"), index=False)
        print("✅ 分析完成，结果已保存到：multi_condition_f0_analysis.csv")

        # Phoneme-level mean comparison
        f0_mean_cols = [col for col in merged_df.columns if col.startswith("f0_mean_")]
        summary = merged_df.groupby("phoneme")[f0_mean_cols].mean().reset_index()
        summary.to_csv(os.path.join(output_dir, "phoneme_level_f0_comparison.csv"), index=False)
        print("📊 每个音素的F0均值对比结果保存至：phoneme_level_f0_comparison.csv")

# ===== 用法示例 =====
if __name__ == "__main__":
    wav_files = ["20000.wav","40000.wav","100000.wav", "150000.wav","200000.wav"]
    labels = ["s_20000","s_40000","s_100000", "s_150000", "s_200000"]
    output_folder = "multi_f0_results"

    process_multiple_conditions(wav_files, labels, output_folder)