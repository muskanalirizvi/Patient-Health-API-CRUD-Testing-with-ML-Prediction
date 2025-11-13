"""
Streamlit frontend for Patient Record & Diabetes Prediction API

Changes in this version:
- Removed Health Check from the Dashboard page (no API /health call on Dashboard).
- Removed the editable API Base URL input from the sidebar and fixed API_BASE to "http://localhost:8000".
  (Sidebar still informs the user which API URL is used.)
- Kept the safe reload_page() helper and previous fixes (forms, address height, full-update payloads).
- Run with:
    pip install streamlit requests pandas plotly
    streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

st.set_page_config(
    page_title="Patient Records & Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# ---------------------------
# Configuration
# ---------------------------
# Fixed API base URL (removed editable input from sidebar per request)
API_BASE = "http://localhost:8000"

# ---------------------------
# Helpers
# ---------------------------
def api_get(path: str, params: dict = None):
    try:
        r = requests.get(API_BASE + path, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None

def api_post(path: str, json_data: dict = None):
    try:
        r = requests.post(API_BASE + path, json=json_data or {}, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        try:
            return {"_error": r.json().get("detail", str(r.text))}
        except Exception:
            return {"_error": r.text}
    except Exception as e:
        return {"_error": str(e)}

def api_put(path: str, json_data: dict):
    try:
        r = requests.put(API_BASE + path, json=json_data, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        try:
            return {"_error": r.json().get("detail", str(r.text))}
        except Exception:
            return {"_error": r.text}
    except Exception as e:
        return {"_error": str(e)}

def api_delete(path: str):
    try:
        r = requests.delete(API_BASE + path, timeout=8)
        r.raise_for_status()
        return {"ok": True}
    except requests.HTTPError:
        try:
            return {"_error": r.json().get("detail", str(r.text))}
        except Exception:
            return {"_error": r.text}
    except Exception as e:
        return {"_error": str(e)}

def pretty_json(d):
    import json
    return json.dumps(d, indent=2, default=str)

def reload_page():
    """
    Cross-version safe rerun:
    - Try st.experimental_rerun() if present.
    - Otherwise update st.query_params to force a rerun, then st.stop().
    - Otherwise toggle session_state key and st.stop().
    - Otherwise instruct manual refresh.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    try:
        st.query_params = {"_": int(time.time())}
        st.stop()
        return
    except Exception:
        pass

    try:
        st.session_state["_reload_ts"] = int(time.time())
        st.stop()
        return
    except Exception:
        pass

    st.info("Please refresh the page manually to see updates.")

# ---------------------------
# Sidebar / Navigation
# ---------------------------
st.sidebar.title("Patient ML Control")
st.sidebar.markdown(f"API Base: `{API_BASE}`  \n(Static — to change, edit streamlit_app.py)")
st.sidebar.caption("Make sure your FastAPI server (main.py) is running on the host above.")
nav = st.sidebar.radio("Navigation", ["Dashboard", "Add Patient", "Patients", "Health"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Jamshed")

# ---------------------------
# Dashboard View (no health check here)
# ---------------------------
if nav == "Dashboard":
    st.title("🩺 Patient Records & Diabetes Prediction — Dashboard")
    st.subheader("Overview")
    patients = api_get("/patients")
    if patients is None:
        st.info("No data (or API unreachable).")
        patients = []
    df = pd.DataFrame(patients) if patients else pd.DataFrame()
    total = len(df)
    positives = df[df["outcome"] == "Positive"].shape[0] if not df.empty and "outcome" in df.columns else 0
    pending = df[df["outcome"] == "Pending"].shape[0] if not df.empty and "outcome" in df.columns else (total - positives)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", total, delta=None)
    c2.metric("Positive Predictions", positives, delta=None)
    c3.metric("Pending / Unknown", pending, delta=None)

    st.markdown("### Outcome Distribution")
    if not df.empty and "outcome" in df.columns:
        fig = px.pie(df, names="outcome", title="Outcomes", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No outcome data yet. Run predictions on patients.")

    st.markdown("### Recent Patients")
    if not df.empty and "date_of_visit" in df.columns:
        try:
            recent = df.sort_values(by="date_of_visit", ascending=False).head(8)
            st.dataframe(recent[["id", "name", "age", "bmi", "outcome", "date_of_visit"]].reset_index(drop=True))
        except Exception:
            st.dataframe(df.head(8))
    else:
        st.info("No patients found. Add one in Add Patient.")

# ---------------------------
# Add Patient View
# ---------------------------
elif nav == "Add Patient":
    st.title("➕ Add New Patient")
    st.markdown("Fill the form below to create a new patient record. Computed fields (BMI, glucose, ... ) are generated by the API.")

    with st.form("add_patient", clear_on_submit=True):
        cols = st.columns([2, 2, 1])
        id_val = cols[0].text_input("Patient ID", placeholder="P001", max_chars=10)
        name = cols[1].text_input("Full name")
        email = cols[2].text_input("Email")

        cols = st.columns([1, 1, 1, 1])
        gender = cols[0].selectbox("Gender", options=["male", "female"])
        contact_number = cols[1].text_input("Contact #")
        age = cols[2].number_input("Age", min_value=1, step=1, value=30)
        pregnancies = cols[3].number_input("Pregnancies", min_value=0, step=1, value=0)

        cols = st.columns([1, 1, 1])
        height = cols[0].number_input("Height (cm)", min_value=30.0, value=170.0, format="%.1f")
        weight = cols[1].number_input("Weight (kg)", min_value=2.0, value=70.0, format="%.1f")
        activity_level = cols[2].selectbox("Activity Level", options=["low", "medium", "high"])

        family_history = st.checkbox("Family history of diabetes", value=False)
        address = st.text_area("Address", height=80)

        submitted = st.form_submit_button("Create Patient")
        if submitted:
            if not id_val or not name or not email:
                st.warning("ID, name and email are required.")
            else:
                payload = {
                    "id": id_val.strip(),
                    "name": name.strip(),
                    "email": email.strip(),
                    "gender": gender,
                    "contact_number": contact_number.strip(),
                    "address": address.strip(),
                    "age": int(age),
                    "height": float(height),
                    "weight": float(weight),
                    "pregnancies": int(pregnancies),
                    "family_history": bool(family_history),
                    "activity_level": activity_level
                }
                result = api_post("/patients", payload)
                if isinstance(result, dict) and result.get("_error"):
                    st.error(f"Error creating patient: {result['_error']}")
                else:
                    st.success(f"Status Code: 200, Patient {payload['id']} created.")
                    reload_page()

# ---------------------------
# Patients List / Details View
# ---------------------------
elif nav == "Patients":
    st.title("📋 Patients")
    search = st.text_input("Search by name or ID")
    limit = st.number_input("Page size", min_value=5, max_value=200, value=50, step=5)

    patients = api_get("/patients")
    if patients is None:
        patients = []
    df = pd.DataFrame(patients) if patients else pd.DataFrame()

    if not df.empty:
        if search:
            mask = df["name"].str.contains(search, case=False, na=False) | df["id"].str.contains(search, na=False)
            df = df[mask]
        # protect against missing columns
        display_cols = [c for c in ["id", "name", "age", "bmi", "outcome", "date_of_visit"] if c in df.columns]
        df_display = df[display_cols].sort_values(by="date_of_visit", ascending=False) if "date_of_visit" in df.columns else df[display_cols]
        st.dataframe(df_display.head(limit).reset_index(drop=True))

        st.markdown("---")
        st.subheader("Patient Detail")
        col_left, col_right = st.columns([2, 1])
        with col_left:
            pid = st.selectbox("Choose patient", options=df["id"].tolist())
            patient = df[df["id"] == pid].iloc[0].to_dict()

            st.markdown(f"### {patient.get('name', '')}  -  {patient.get('id')}")
            st.write(f"Email: {patient.get('email')}")
            st.write(f"Contact: {patient.get('contact_number')}")
            st.write(f"Address: {patient.get('address')}")
            st.write(f"Visited: {patient.get('date_of_visit')}")
            st.write("")

            # Health metrics
            metrics_row = st.columns(4)
            metrics_row[0].metric("BMI", patient.get("bmi", "—"))
            metrics_row[1].metric("Glucose", patient.get("glucose", "—"))
            metrics_row[2].metric("Blood Pressure", patient.get("blood_pressure", "—"))
            metrics_row[3].metric("Insulin", patient.get("insulin", "—"))

            st.markdown("#### Full record")
            st.json(patient)

        with col_right:
            st.markdown("### Actions")
            if st.button("Predict Diabetes", key=f"predict_{pid}"):
                with st.spinner("Running prediction..."):
                    pred = api_post(f"/predict_disease/{pid}", None)
                    if isinstance(pred, dict) and pred.get("_error"):
                        st.error(f"Prediction error: {pred['_error']}")
                    else:
                        prob = pred.get("probability", 0.0)
                        label = pred.get("prediction", "Unknown")
                        st.success(f"Prediction: {label} ({prob*100:.1f}%)")
                        st.progress(min(max(prob, 0.0), 1.0))
                        st.write(pretty_json(pred))

            with st.expander("Edit Patient"):
                with st.form(f"edit_patient_form_{pid}"):
                    edit_name = st.text_input("Full name", value=patient.get("name", ""))
                    edit_email = st.text_input("Email", value=patient.get("email", ""))
                    edit_contact = st.text_input("Contact", value=patient.get("contact_number", ""))
                    edit_address = st.text_area("Address", value=patient.get("address", ""), height=80)
                    ecols = st.columns(3)
                    edit_age = ecols[0].number_input("Age", min_value=1, value=int(patient.get("age", 30)))
                    edit_height = ecols[1].number_input("Height (cm)", min_value=30.0, value=float(patient.get("height", 170.0)), format="%.1f")
                    edit_weight = ecols[2].number_input("Weight (kg)", min_value=2.0, value=float(patient.get("weight", 70.0)), format="%.1f")

                    edit_gender = st.selectbox("Gender", options=["male", "female"], index=["male","female"].index(patient.get("gender", "male")))
                    edit_activity = st.selectbox("Activity level", options=["low", "medium", "high"], index=["low", "medium", "high"].index(patient.get("activity_level", "medium")))
                    edit_family = st.checkbox("Family history", value=bool(patient.get("family_history", False)))
                    edit_preg = st.number_input("Pregnancies", min_value=0, value=int(patient.get("pregnancies", 0)))

                    save_clicked = st.form_submit_button("Save changes")
                    if save_clicked:
                        payload = {
                            "id": pid,
                            "name": edit_name,
                            "email": edit_email,
                            "gender": edit_gender,
                            "contact_number": edit_contact,
                            "address": edit_address,
                            "age": int(edit_age),
                            "height": float(edit_height),
                            "weight": float(edit_weight),
                            "pregnancies": int(edit_preg),
                            "family_history": bool(edit_family),
                            "activity_level": edit_activity
                        }
                        res = api_put(f"/patients/{pid}", payload)
                        if isinstance(res, dict) and res.get("_error"):
                            st.error(f"Update error: {res['_error']}")
                        else:
                            st.success("Status code: 200, Patient updated.")
                            reload_page()

            st.markdown("#### Danger Zone")
            confirm_del = st.checkbox("I confirm that I want to delete this patient", key=f"confirm_del_{pid}")
            if st.button("Delete Patient", key=f"del_{pid}"):
                if not confirm_del:
                    st.warning("Please check the confirmation box before deleting.")
                else:
                    res = api_delete(f"/patients/{pid}")
                    if isinstance(res, dict) and res.get("_error"):
                        st.error(f"Delete error: {res['_error']}")
                    else:
                        st.success("Patient deleted.")
                        reload_page()

    else:
        st.info("No patients yet. Add a new patient from the Add Patient section.")

# ---------------------------
# Health view (kept separate)
# ---------------------------
elif nav == "Health":
    st.title("🔍 Health Check")
    st.markdown("Checks model, scaler and database status from FastAPI /health endpoint.")
    health = api_get("/health")
    if health:
        if health.get("status") == "healthy":
            st.success("Service is healthy ✅")
        else:
            st.error("Service reported issues ❌")
        st.json(health)
    else:
        st.error("Health check failed — is the API running?")

# ---------------------------
# Footer
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Tips:\n- Use the Dashboard to see quick stats\n- Add patients before running predictions\n- If your API runs on a different host/port, update API_BASE in the script")