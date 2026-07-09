# CSIRO Biomass – projekt kursowy TersorFlow (ML-II, UW)

Projekt zrealizowany w ramach kursu uczenia maszynowego na Uniwersytecie Warszawskim.  
Opiera się na wyzwaniu Kaggle **CSIRO Biomass Challenge** – przewidywanie 5 składowych suchej biomasy pastwiska na podstawie zdjęć ortofoto i metadanych.

## Wykorzystane biblioteki
- `pandas`, `numpy` – analiza danych,
- `TensorFlow / Keras` – budowa i uczenie modeli,
- `matplotlib`, `seaborn` – wizualizacje.

## Główne założenia
- Obrazy dzielone na patche 256×256, przetwarzane w przestrzeni **HSV**.
- Dwuetapowe uczenie:
  1. **Autoenkoder** (pretraining) – uczy się odtwarzać patche.
  2. **Model właściwy** – zamrożony enkoder + pozycyjne kodowanie 2D + atencja (Multi‑Head) + MLP.
- Model przewiduje tylko 3 zmienne (`Green`, `GDM`, `Total`) w skali logarytmicznej (`log1p`); `Clover` i `Dead` wyprowadzane matematycznie.
- Straty: ważona MAE (0.1, 0.3, 0.6); ocena: ważony R² dla 5 zmennych (zgodny z regułami Kaggle).

Walidacja (miesiąc 10): **R² ≈ 0.35**  
Test (miesiąc 11): **R² ≈ 0.50**
