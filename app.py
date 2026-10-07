import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="WATTWISE - AI NILM", page_icon="⚡", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
APPLIANCES = ["Air Conditioner", "Refrigerator", "Washing Machine", "Kettle", "TV", "Lights"]
POWER = np.array([1200, 150, 500, 1800, 120, 180], dtype=float)
BASE_LOAD = 55.0
RATE = 8.0
MONTHS = ["May", "June", "July", "August", "September", "October"]

# -----------------------------
# Synthetic NILM data generator
# -----------------------------
def make_windows(n=9000, seed=42, window=7):
    rng = np.random.default_rng(seed)
    states = rng.binomial(1, [0.18, 0.72, 0.07, 0.035, 0.20, 0.45], size=(n, len(APPLIANCES)))
    # Add realistic persistence: appliances tend to remain ON for several samples.
    for j in range(states.shape[1]):
        for i in range(1, n):
            if rng.random() < [0.88, 0.97, 0.90, 0.82, 0.91, 0.94][j]:
                states[i, j] = states[i - 1, j]
    # Force occasional appliance events.
    for _ in range(max(50, n // 60)):
        t = rng.integers(0, n)
        a = rng.integers(0, len(APPLIANCES))
        duration = int(rng.integers(2, 18))
        states[t:min(n, t + duration), a] = 1
    aggregate = BASE_LOAD + states @ POWER + rng.normal(0, 25, n)
    aggregate = np.clip(aggregate, 0, None)

    X, y = [], []
    half = window // 2
    for i in range(half, n - half):
        w = aggregate[i-half:i+half+1]
        # Features are intentionally based only on the aggregate meter signal.
        X.append(np.r_[w, w.mean(), w.std(), w.max() - w.min()])
        y.append(states[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), aggregate, states

@st.cache_resource(show_spinner="Training AI NILM model...")
def train_nilm_model():
    X, y, _, _ = make_windows()
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=80,
                      early_stopping=True, random_state=42, batch_size=256)
    )
    model.fit(X, y)
    return model

MODEL = train_nilm_model()

# -----------------------------
# Simulation / inference helpers
# -----------------------------
def simulated_day(seed=10):
    rng = np.random.default_rng(seed)
    samples = 24 * 12  # every 5 minutes
    t = np.arange(samples)
    hour = t / 12
    states = np.zeros((samples, len(APPLIANCES)), dtype=int)

    # Refrigerator cycles throughout the day.
    states[:, 1] = ((np.sin(2*np.pi*(hour % 1.7)/1.7) > 0.1)).astype(int)
    states[:, 1] &= rng.random(samples) > 0.16
    # AC mostly evening/night + some afternoon use.
    states[:, 0] = (((hour >= 13) & (hour <= 16)) | (hour >= 18) | (hour <= 6)).astype(int)
    states[:, 0] &= rng.random(samples) > 0.28
    # Washing machine: two sessions.
    states[:, 2] = (((hour >= 8) & (hour < 9.5)) | ((hour >= 18) & (hour < 19))).astype(int)
    # Kettle: short bursts.
    states[:, 3] = (((hour > 7.5) & (hour < 7.75)) | ((hour > 16.5) & (hour < 16.75)) | ((hour > 20) & (hour < 20.25))).astype(int)
    # TV evening.
    states[:, 4] = ((hour >= 18.5) & (hour <= 22.5)).astype(int)
    # Lights morning/evening/night.
    states[:, 5] = ((hour < 6.5) | (hour >= 18)).astype(int)
    states = np.where(states & (rng.random(states.shape) > 0.08), 1, 0)

    aggregate = BASE_LOAD + states @ POWER + rng.normal(0, 22, samples)
    aggregate = np.clip(aggregate, 0, None)
    return pd.DataFrame({"Time": pd.date_range("2026-10-07", periods=samples, freq="5min"),
                         "Aggregate_W": aggregate}), states

def feature_from_window(values):
    values = np.asarray(values, dtype=float)
    return np.r_[values, values.mean(), values.std(), values.max() - values.min()].reshape(1, -1)

def estimate_states(model, aggregate):
    vals = np.asarray(aggregate, dtype=float)
    window = 7
    half = window // 2
    out = np.zeros((len(vals), len(APPLIANCES)), dtype=int)
    for i in range(half, len(vals) - half):
        out[i] = model.predict(feature_from_window(vals[i-half:i+half+1]))[0]
    out[:half] = out[half]
    out[-half:] = out[-half-1]
    return out

# -----------------------------
# Session / authentication
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "simulation_seed" not in st.session_state:
    st.session_state.simulation_seed = 10


def login_page():
    st.markdown("# ⚡ WATTWISE")
    st.markdown("### AI-powered household energy monitoring")
    st.caption("One main meter + NILM AI → appliance-level energy insights")
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            if email == "demo@wattwise.app" and password == "WattWise@123":
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid login. Demo: demo@wattwise.app / WattWise@123")
    with tab2:
        name = st.text_input("Full name")
        email2 = st.text_input("Email address")
        p1 = st.text_input("Create password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        if st.button("Create Account", use_container_width=True):
            if not name or not email2 or not p1:
                st.warning("Please complete all fields.")
            elif p1 != p2:
                st.error("Passwords do not match.")
            elif len(p1) < 8:
                st.error("Password should contain at least 8 characters.")
            else:
                st.success("Prototype account created. Use the demo login to enter the application.")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# -----------------------------
# Generate current smart-meter data + AI disaggregation
# -----------------------------
meter_df, true_states = simulated_day(st.session_state.simulation_seed)
pred_states = estimate_states(MODEL, meter_df["Aggregate_W"].values)
meter_df["AI_Estimated_Appliance_W"] = pred_states @ POWER
meter_df["Other_W"] = np.maximum(meter_df["Aggregate_W"] - meter_df["AI_Estimated_Appliance_W"], 0)

# Appliance energy from AI predictions. Samples are 5 minutes.
interval_h = 5 / 60
energy_kwh = pred_states.sum(axis=0) * POWER * interval_h / 1000
costs = energy_kwh * RATE
usage_hours = pred_states.sum(axis=0) * interval_h
app_df = pd.DataFrame({
    "Appliance": APPLIANCES,
    "Energy (kWh)": energy_kwh,
    "Cost (₹)": costs,
    "Usage (hours)": usage_hours
})
total_kwh = float(app_df["Energy (kWh)"].sum())
total_cost = float(app_df["Cost (₹)"].sum())
peak_power = float(meter_df["Aggregate_W"].max())
current_power = float(meter_df["Aggregate_W"].iloc[-1])
highest = app_df.loc[app_df["Energy (kWh)"].idxmax(), "Appliance"]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚡ WATTWISE")
    st.caption(f"Signed in as {st.session_state.user_email}")
    st.success("AI NILM model: ONLINE")
    page = st.radio("Menu", ["Dashboard", "Live Usage", "Appliance Usage", "AI Model", "History", "Alerts", "Profile"])
    st.divider()
    st.caption("Simulation controls")
    if st.button("🔄 Generate New Smart-Meter Data", use_container_width=True):
        st.session_state.simulation_seed += 1
        st.rerun()
    st.caption(f"Simulation run: #{st.session_state.simulation_seed}")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    st.title("WATTWISE Dashboard")
    st.caption("October 2026 • AI-based Non-Intrusive Load Monitoring")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today's Energy", f"{total_kwh:.1f} kWh")
    c2.metric("Estimated Cost", f"₹{total_cost:,.0f}")
    c3.metric("Highest Consumer", highest)
    c4.metric("Peak Power", f"{peak_power:,.0f} W")

    import plotly.express as px
    st.subheader("Aggregate Smart-Meter Signal")
    fig = px.line(meter_df, x="Time", y="Aggregate_W", title="Main-line power reading")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("AI Appliance Disaggregation")
        fig2 = px.bar(app_df, x="Appliance", y="Energy (kWh)", title="AI-estimated appliance energy")
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.subheader("Estimated Cost")
        fig3 = px.pie(app_df, names="Appliance", values="Cost (₹)", title="Cost contribution")
        st.plotly_chart(fig3, use_container_width=True)

    st.info("AI pipeline: aggregate meter signal → sliding-window features → neural-network NILM model → appliance ON/OFF states → kWh and cost.")

elif page == "Live Usage":
    st.title("Live Usage")
    st.caption("Simulated smart-meter stream. Replace this source with ESP32/current-sensor readings in the hardware version.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Power", f"{current_power:,.0f} W")
    c2.metric("Peak Today", f"{peak_power:,.0f} W")
    c3.metric("Today's Energy", f"{total_kwh:.1f} kWh")
    c4.metric("Today's Cost", f"₹{total_cost:,.0f}")

    import plotly.express as px
    fig = px.line(meter_df.tail(72), x="Time", y=["Aggregate_W", "AI_Estimated_Appliance_W"],
                  title="Latest aggregate reading vs AI-disaggregated load")
    st.plotly_chart(fig, use_container_width=True)

    active = pred_states[-1]
    active_names = [APPLIANCES[i] for i, x in enumerate(active) if x]
    if active_names:
        st.success("AI currently detects: " + ", ".join(active_names))
    else:
        st.info("AI currently detects no major appliance load.")

elif page == "Appliance Usage":
    st.title("Appliance Usage")
    st.caption("These values are inferred from one aggregate power signal by the NILM neural network.")
    st.dataframe(app_df.round(2), use_container_width=True, hide_index=True)
    selected = st.selectbox("Select appliance", APPLIANCES)
    row = app_df[app_df["Appliance"] == selected].iloc[0]
    a, b, c = st.columns(3)
    a.metric("AI-estimated Energy", f"{row['Energy (kWh)']:.2f} kWh")
    b.metric("Estimated Cost", f"₹{row['Cost (₹)']:.0f}")
    c.metric("Estimated Usage", f"{row['Usage (hours)']:.1f} hrs")

elif page == "AI Model":
    st.title("AI / NILM Model")
    st.caption("This prototype trains a lightweight multi-output neural network on synthetic aggregate household signals.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "MLP Neural Network")
    c2.metric("Hidden Layers", "64 → 32")
    c3.metric("Outputs", f"{len(APPLIANCES)} appliances")
    st.markdown("### How it works")
    st.code("""Aggregate Power (W)\n        ↓\n7-point sliding window + statistics\n        ↓\nStandardScaler\n        ↓\nMLP Neural Network (64 → 32, ReLU)\n        ↓\n6 appliance ON/OFF predictions\n        ↓\nPower × time → kWh → ₹ cost""", language="text")
    st.markdown("### Important prototype note")
    st.warning("The training data is synthetic so the prototype can run without hardware. For a real project, train/validate the model using measured appliance-level datasets or your own smart-meter + appliance measurements.")
    st.markdown("### Current detected state")
    state_df = pd.DataFrame({"Appliance": APPLIANCES, "AI State": ["ON" if x else "OFF" for x in pred_states[-1]], "Rated Power (W)": POWER.astype(int)})
    st.dataframe(state_df, use_container_width=True, hide_index=True)

elif page == "History":
    st.title("Consumption History")
    # Scale a realistic historical curve around the current AI estimate.
    historical = pd.DataFrame({"Month": MONTHS, "Energy (kWh)": [max(1, total_kwh*x) for x in [0.72, 0.81, 0.93, 0.88, 0.96, 1.0]]})
    historical["Cost (₹)"] = historical["Energy (kWh)"] * RATE
    st.dataframe(historical.round(1), use_container_width=True, hide_index=True)
    import plotly.express as px
    fig = px.line(historical, x="Month", y="Energy (kWh)", markers=True, title="Monthly AI-estimated energy trend")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Alerts":
    st.title("Alerts & AI Insights")
    if peak_power > 2000:
        st.error(f"⚠️ High peak load detected: {peak_power:,.0f} W")
    else:
        st.success(f"✓ Peak load is within the prototype threshold: {peak_power:,.0f} W")
    st.warning(f"⚡ {highest} is the highest estimated energy consumer today.")
    if current_power > 2000:
        st.warning("High current demand right now. Consider switching off non-essential loads.")
    else:
        st.info("💡 Current demand is moderate. AI is continuously estimating appliance states.")
    st.subheader("Recommendations")
    st.write("• Reduce long AC operation during high-demand periods.")
    st.write("• Avoid overlapping high-power appliances such as kettle and washing machine.")
    st.write("• Use the History page to identify increasing consumption trends.")

elif page == "Profile":
    st.title("Profile & Settings")
    st.write(f"**Account:** {st.session_state.user_email}")
    st.text_input("Name", value="Demo User")
    st.number_input("Electricity tariff (₹/kWh)", value=RATE, min_value=1.0, step=0.5)
    st.selectbox("Preferred report", ["Monthly", "Weekly", "Daily"])
    st.button("Save Settings")
    st.caption("Production deployment should use secure authentication, persistent user data, and an actual sensor/API backend.")

st.divider()
st.caption(f"WATTWISE AI NILM Prototype • Generated {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
