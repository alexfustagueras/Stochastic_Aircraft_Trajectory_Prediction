# VT1 - Stochastic Aircraft Trajectory Prediction

This repository contains the full implementation for the specialization project **"Stochastic Aircraft Trajectory Prediction"**, developed at the **Zurich University of Applied Sciences (ZHAW)**, Center for Aviation (ZAV). The project explores **short-term aircraft trajectory forecasting in en-route airspace** using a **Bidirectional LSTM with Mixture Density Network (BLSTM-MDN)** architecture trained on historical ADS-B data.

---

## 🚀 Project Overview

With the rise of **Free Route Airspace (FRA)** in Europe, accurately predicting aircraft positions — along with their uncertainty — is critical for conflict detection and traffic safety. This project develops a deep learning model that outputs **probabilistic forecasts** of future trajectories, using:

- Historical ADS-B trajectories from Swiss airspace (FRA-LSAS).
- A BLSTM encoder-decoder framework.
- A Mixture Density Network (MDN) output layer to model uncertainty.

---

## 📁 Project Structure

```
VT_1/
│
├── 01_preprocessing.ipynb         # Raw data loading and filtering
├── 02a_filtering.ipynb            # En-route trajectory extraction
├── 02b_transforming.ipynb         # Feature engineering
├── 02c_generating_samples.ipynb   # Sample generation (input/output pairs)
├── 03_model_training.py           # Script version of model training
├── 04_inference.ipynb             # Inference and uncertainty visualization
│
├── traffic.yml                    # Conda environment for preprocessing
├── tf-mdn.yml                     # Conda environment for model training
├── README.md
```

---

## 📦 Environments

Two Conda environments are used:

* `traffic.yml` — for data preprocessing using the [`traffic`](https://traffic-viz.github.io/) library.
* `tf-mdn.yml` — for model development and training with TensorFlow.

To recreate them:

```bash
conda env create -f traffic.yml
conda env create -f tf-mdn.yml
```

---

## 📚 Citation

> **Alex Fustagueras**.
> *Stochastic Aircraft Trajectory Prediction*.

> Specialization Project, ZHAW Centre for Aviation, July 2025