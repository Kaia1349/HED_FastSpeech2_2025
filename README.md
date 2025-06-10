# 🎙️ Fine-Grained Emotional Speech Synthesis with FastSpeech 2 + HED

This repository contains the official implementation of the master's thesis:

**"Towards Fine-Grained Emotional Modulation in FastSpeech 2 with Hierarchical Emotion Distributions"**

This work investigates phoneme-level controllability of emotional prosody in speech synthesis. A 12-dimensional Hierarchical Emotion Distribution (HED) vector is introduced and injected into a modified FastSpeech 2 architecture to enable fine-grained, interpretable, and consistent emotional modulation.

---

## 📄 Thesis

- **Title**: Towards Fine-Grained Emotional Modulation in FastSpeech 2 with Hierarchical Emotion Distributions  
- **Author**: Qiyan  
- **Institution**: [Your University Name]  
- **Date**: June 2025

📄 PDF: *[Add link or upload if available]*

---

## 📁 Project Structure

```
hed-fastspeech2/
├── configs/              # YAML configuration files
├── checkpoints/          # Pretrained baseline checkpoint at 15000 steps
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
git clone https://github.com/your-username/hed-fastspeech2.git
cd hed-fastspeech2
pip install -r requirements.txt
```

⚠️ *Note: This project requires OpenSMILE and textgrid Python bindings.*

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
python scripts/synthesize.py \
  --text "I am going to back home" \
  --hed_vector "0.3 0.2 0.1 0.4 0.5 0.2 0.3 0.1 0.3 0.2 0.3 0.4"
```

---

## 📊 Evaluation

Scripts for prosody analysis and subjective testing:
- `evaluation/phoneme_level.py`: per-phoneme F0 / Energy analysis
- `evaluation/evaluate_emotion.py`: utterance-level acoustic curves
- `evaluation/example_bws_results.xlsx`: annotated BWS score sheet

---

## 📥 Pretrained Resources

- ✅ 15k-step baseline checkpoint: `checkpoints/baseline/15000.pth.tar`  
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

## 📬 Contact

For questions or suggestions:  
**Qiyan**  
📧 Email: [your-email@example.com]  
🌐 GitHub: [https://github.com/your-username](https://github.com/your-username)
