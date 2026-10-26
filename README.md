# Cheminformatics Virtual Laboratory

A Streamlit web app for educational cheminformatics exploration.

## Features
- SMILES input and built-in example molecules
- 2D structure depiction using RDKit
- Molecular descriptors (formula, molecular weight, LogP, TPSA, H-bond donors/acceptors, rings, bonds, etc.)
- Interactive 3D viewer using py3Dmol
- CSV export and PDF report generation

## Run locally
Use Python 3.10 or 3.11 for the easiest dependency compatibility.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud and create a new app from that repository.
4. Select `app.py` as the main file and deploy.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Notes
- The app is a fresh implementation inspired by the functionality in the code shared in chat, not a copy of an inaccessible hosted app.
- Molecular descriptors are calculated values, not experimental measurements.
- Sigma/pi bond counting for aromatic systems is convention-dependent and is labelled as approximate.
