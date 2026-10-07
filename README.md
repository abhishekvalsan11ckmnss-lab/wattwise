# WATTWISE — Working AI NILM Prototype

This version upgrades the original UI prototype into a software-only **working NILM demonstration**.

## What is actually working

- Streamlit user-side dashboard
- Demo login / registration flow
- Synthetic smart-meter (aggregate power) generator
- Lightweight neural-network NILM model using scikit-learn MLP
- AI appliance ON/OFF estimation from the aggregate signal
- Appliance-level energy estimation (kWh)
- Estimated electricity cost
- Live Usage page
- AI Model page explaining the pipeline
- Appliance Usage page
- History and alerts
- Button to generate a new smart-meter simulation

## AI pipeline

```text
Aggregate household power
          ↓
7-sample sliding window
          ↓
Feature scaling
          ↓
MLP neural network
64 neurons → 32 neurons
          ↓
6 appliance ON/OFF predictions
          ↓
Power × time
          ↓
Appliance kWh + ₹ cost
```

## Run on Windows

1. Install Python 3.10 or newer.
2. Open Command Prompt inside this folder.
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` after installing Python.

## Demo login

Email: `demo@wattwise.app`

Password: `WattWise@123`

## Important academic/project note

The neural network is trained on **synthetic data** so the prototype works immediately without ESP32 hardware. It demonstrates the NILM concept, but its accuracy must NOT be presented as real-world model accuracy.

For the final hardware version, replace the synthetic meter generator with ESP32 + current/energy sensor readings and retrain the model using measured appliance datasets.
