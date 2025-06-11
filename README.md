# 🎙️ Fine-Grained Emotional Speech Synthesis with FastSpeech 2 + HED

This repository contains the official implementation of the master's thesis:

**"Towards Fine-Grained Emotional Modulation in FastSpeech 2 with Hierarchical Emotion Distributions"**

This work investigates phoneme-level controllability of emotional prosody in speech synthesis. A 12-dimensional Hierarchical Emotion Distribution (HED) vector is introduced and injected into a modified FastSpeech 2 architecture to enable fine-grained, interpretable, and consistent emotional modulation.

---

## 📄 Thesis

- **Title**: Towards Fine-Grained Emotional Modulation in FastSpeech 2 with Hierarchical Emotion Distributions  
- **Author**: Qiyan Huang
- **Institution**: University of Groningen
- **Date**: June 2025

---

## 📁 Project Structure

```
hed-fastspeech2/
├── configs/              # YAML configuration files
├── data/                 # Example HED vectors (for inference tests)
├── evaluation/           # Prosody analysis & BWS scoring scripts
├── logs/                 # Training and validation logs
├── model/                # Modified FastSpeech 2 model definition
├── parameters/           # Pretrained SVM and scaler models for HED
├── scripts/              # All training, synthesis, and extraction scripts
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT license
└── README.md             # You are here
```

---

## 🧠 Key Features

- ✅ Phoneme-level controllable emotional synthesis via 12-dim HED vectors  
- ✅ Modified FastSpeech 2 with mid-encoder HED injection  
- ✅ Stepwise emotional conditioning during training  
- ✅ Evaluation of both sentence- and phoneme-level prosody  
- ✅ Subjective perceptual evaluation using Best-Worst Scaling (BWS)  

---

## 📦 Installation

```bash
[git clone https://github.com/kaia1349/HED_FastSpeech2_2025.git](https://github.com/Kaia1349/HED_FastSpeech2_2025.git)
cd HED_FastSpeech2_2025
pip install -r requirements.txt
```

⚠️ *Note:This project requires OpenSMILE, Montreal Forced Aligner (MFA) (https://montreal-forced-aligner.readthedocs.io)

---

## 🗃️ Dataset

The project uses the English subset of the Emotional Speech Dataset (ESD).  
Download the dataset separately and organize it under `data/raw/` for preprocessing.

This repository does not contain or distribute any audio files due to licensing restrictions.

---

## 🔍 HED Feature Extraction

To extract utterance-, word-, and phone-level features and compute 12-dim HED vectors:

```bash
python scripts/extract_esd_features.py
python scripts/get_hierarchical_ed_svm.py
```

Required resources:
- `parameters/linearsvm_OpenSMILE.pkl`  
- `parameters/scaler_OpenSMILE.pkl`

➡️ The `eGeMAPSv02.conf` config file must be downloaded from the [official OpenSMILE repository](https://audeering.github.io/opensmile/).

---

## 🏋️‍♂️ Training

Baseline FastSpeech 2 (0–15000 steps):

```bash
python scripts/train.py --config configs/train.yaml
```

HED-enhanced fine-tuning (from 15000):

```bash
python scripts/train.py --config configs/train.yaml --resume_path checkpoints/baseline/15000.pth.tar
```

---

## 🗣️ Inference

Example inference with HED vector control:

```bash
python3 synthesize.py \
  --restore_step 200000 \
  --mode single \
  --text "all the way to China is home" \
  --speaker_id 2 \
  -p config/HED/preprocess.yaml \
  -m config/HED/model.yaml \
  -t config/HED/train.yaml \
  --hed_vector 0.3895 0.3990 0.4661 0.4545 0.3298 0.3498 0.4751 0.4199 0.2574 0.3931 0.4649 0.4198
```

---

## 📊 Evaluation

Scripts for prosody analysis and subjective testing:
- `evaluation/analyze_phoneme_level.py`: per-phoneme F0 analysis
- `evaluation/analyze_sentence_level.py`: utterance-level acoustic curves
- `evaluation/BWS_results_summary.pdf`:  BWS survey results 

---

## 📥 Pretrained Resources

- ✅ 15k-step baseline checkpoint: https://drive.google.com/file/d/1M4dd7YI_dZr9IC1JSrl97cmOT7trkF2z/view?usp=sharing
- ✅ Checkpoints used for analysis：https://drive.google.com/drive/folders/1wqZe1ofbq9CbfR3EpCl0vfc5-G6QmX1y?usp=sharing
- ✅ Example normalized HED vectors: `data/hed_vectors/*.npy`  
- ✅ Trained SVM and scaler models: `parameters/*.pkl`  

---

## 📚 Acknowledgments

This repository is adapted from and built upon:
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2): The original FastSpeech 2 implementation (MIT License)  
- [shinshoji01/Summary-Hierarchical-ED](https://github.com/shinshoji01/Summary-Hierarchical-ED): The HED feature extraction logic, using OpenSMILE and SVM

All external components are used in accordance with their licenses.

---

## 📜 License

This project is licensed under the MIT License.

---

