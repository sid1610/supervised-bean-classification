# Dry Bean Classification App

This repository contains a Streamlit app for predicting dry bean varieties from 16 numeric bean shape features.
The interface uses a mission-control style dashboard with a clickable orbital viewport that captures targeted cursor points.

## UI Highlights

- Clickable target viewport for capturing cursor coordinates.
- Live target X/Y display with estimated latitude, longitude, sector, and lock confidence.
- Target acquisition log that stores the latest clicked points.
- Advanced classifier control panel grouped by feature categories.
- SVC decision-score table and styled prediction output.

## Project Files

- `app.py` - Streamlit web app with target capture and model predictions.
- `bean_classifier_model.pkl` - trained SVC classification model.
- `bean_scaler.pkl` - fitted `StandardScaler` used before prediction.
- `label_encoder.pkl` - fitted label encoder for converting model output to bean class names.
- `requirements.txt` - Python dependencies needed to run the app.
- `runtime.txt` - Python runtime hint for Streamlit Community Cloud.

## Bean Classes

The app predicts one of these classes:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

## Features Expected by the Model

The model expects these 16 features in this exact order:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

Use the same feature scale/preprocessing that was used when the model was trained.

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Click inside the orbital viewport to capture a target point. The left panel updates with the cursor X/Y position, estimated map coordinates, sector, and lock confidence.

## Push to GitHub

If this folder is not already a Git repository, initialize it:

```bash
git init
git branch -M main
git add .
git commit -m "Add dry bean classifier app"
```

Create an empty repository on GitHub, then connect and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub username and repository name.
