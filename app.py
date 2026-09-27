"""
Road Accident Severity Prediction
A Streamlit web application for predicting road accident severity
using a trained Machine Learning classification model.

Project: B.Tech Data Science IDP
Author:  <Your Name>
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Add project root to sys.path so we can import from /src
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.feature_engineering import derive_time_features, build_input_dataframe
from src.prediction import (
    load_model,
    load_preprocessing_pipeline,
    run_prediction,
    MODEL_PATH,
    PIPELINE_PATH,
)
from src.config import (
    CATEGORICAL_OPTIONS,
    NUMERIC_FIELDS,
    SEVERITY_CLASSES,
    SEVERITY_COLORS,
)


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------
CSS_PATH = ROOT_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🚦 Road Accident Severity Prediction</h1>
        <h3>Machine Learning Based Accident Severity Classification</h3>
        <p>Enter the accident details below to predict the probable severity of the accident.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar - Model status / About
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown(
        """
        This application uses a trained Machine Learning model to classify
        road accident severity into three categories:

        - 🟢 **Slight Injury**
        - 🟠 **Serious Injury**
        - 🔴 **Fatal injury**
        """
    )

    st.markdown("---")
    st.markdown("### 📦 Model Status")

    model_exists = MODEL_PATH.exists()
    pipeline_exists = PIPELINE_PATH.exists()

    st.markdown(
        f"**Model file:** {'✅ Found' if model_exists else '❌ Not found'}<br>"
        f"**Pipeline file:** {'✅ Found' if pipeline_exists else '❌ Not found'}",
        unsafe_allow_html=True,
    )

    if not (model_exists and pipeline_exists):
        st.info(
            "Place your trained files in the `models/` folder:\n\n"
            "- `accident_severity_model.pkl`\n"
            "- `preprocessing_pipeline.pkl`"
        )

    st.markdown("---")
    st.caption("B.Tech Data Science — IDP Project")


# -----------------------------------------------------------------------------
# Helper for section headers
# -----------------------------------------------------------------------------
def section_header(icon, title, subtitle=None):
    html = f"""
    <div class="section-header">
        <h2>{icon} {title}</h2>
    </div>
    """
    if subtitle:
        # Fixed the quotes here: using single quotes on the outside
        html += f'<p class="section-subtitle">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

with st.form("accident_input_form"):

    # ---------------------------------------------------------------
    # Section 1 — Time & Accident Context
    # ---------------------------------------------------------------
    section_header(
        "🕒", "Time & Accident Context",
        "Details about when and where the accident occurred."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        time_input = st.time_input(
            "Time of Accident",
            value=pd.to_datetime("14:30").time(),
            help="Engineered features (Hour, Minute, Minutes_since_midnight, Time_period) "
                 "will be derived automatically.",
        )
    with col_b:
        day_of_week = st.selectbox("Day of Week", CATEGORICAL_OPTIONS["Day_of_week"])
    with col_c:
        area_accident = st.selectbox(
            "Area Accident Occurred", CATEGORICAL_OPTIONS["Area_accident_occured"]
        )
    with col_d:
        junction_type = st.selectbox(
            "Types of Junction", CATEGORICAL_OPTIONS["Types_of_Junction"]
        )

    col_e, col_f, col_g, col_h = st.columns(4)
    with col_e:
        light_conditions = st.selectbox(
            "Light Conditions", CATEGORICAL_OPTIONS["Light_conditions"]
        )
    with col_f:
        weather_conditions = st.selectbox(
            "Weather Conditions", CATEGORICAL_OPTIONS["Weather_conditions"]
        )
    with col_g:
        road_surface_conditions = st.selectbox(
            "Road Surface Conditions", CATEGORICAL_OPTIONS["Road_surface_conditions"]
        )
    with col_h:
        road_surface_type = st.selectbox(
            "Road Surface Type", CATEGORICAL_OPTIONS["Road_surface_type"]
        )

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Section 2 — Driver Information
    # ---------------------------------------------------------------
    section_header(
        "👤", "Driver Information",
        "Details about the driver involved in the accident."
    )

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        age_band_driver = st.selectbox(
            "Age Band of Driver", CATEGORICAL_OPTIONS["Age_band_of_driver"]
        )
    with col_b:
        sex_of_driver = st.selectbox(
            "Sex of Driver", CATEGORICAL_OPTIONS["Sex_of_driver"]
        )
    with col_c:
        educational_level = st.selectbox(
            "Educational Level", CATEGORICAL_OPTIONS["Educational_level"]
        )
    with col_d:
        vehicle_driver_relation = st.selectbox(
            "Vehicle Driver Relation", CATEGORICAL_OPTIONS["Vehicle_driver_relation"]
        )
    with col_e:
        driving_experience = st.selectbox(
            "Driving Experience", CATEGORICAL_OPTIONS["Driving_experience"]
        )

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Section 3 — Vehicle Information
    # ---------------------------------------------------------------
    section_header(
        "🚗", "Vehicle Information",
        "Details about the vehicle(s) involved in the accident."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        type_of_vehicle = st.selectbox(
            "Type of Vehicle", CATEGORICAL_OPTIONS["Type_of_vehicle"]
        )
    with col_b:
        owner_of_vehicle = st.selectbox(
            "Owner of Vehicle", CATEGORICAL_OPTIONS["Owner_of_vehicle"]
        )
    with col_c:
        service_year = st.selectbox(
            "Service Year of Vehicle", CATEGORICAL_OPTIONS["Service_year_of_vehicle"]
        )
    with col_d:
        defect_of_vehicle = st.selectbox(
            "Defect of Vehicle", CATEGORICAL_OPTIONS["Defect_of_vehicle"]
        )

    col_e, col_f = st.columns(2)
    with col_e:
        num_vehicles_involved = st.number_input(
            "Number of Vehicles Involved",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Total number of vehicles involved in the accident.",
        )
    with col_f:
        vehicle_movement = st.selectbox(
            "Vehicle Movement", CATEGORICAL_OPTIONS["Vehicle_movement"]
        )

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Section 4 — Road & Collision Information
    # ---------------------------------------------------------------
    section_header(
        "🛣️", "Road & Collision Information",
        "Details about road geometry and collision type."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        lanes_or_medians = st.selectbox(
            "Lanes or Medians", CATEGORICAL_OPTIONS["Lanes_or_Medians"]
        )
    with col_b:
        road_allignment = st.selectbox(
            "Road Alignment", CATEGORICAL_OPTIONS["Road_allignment"]
        )
    with col_c:
        type_of_collision = st.selectbox(
            "Type of Collision", CATEGORICAL_OPTIONS["Type_of_collision"]
        )
    with col_d:
        cause_of_accident = st.selectbox(
            "Cause of Accident", CATEGORICAL_OPTIONS["Cause_of_accident"]
        )

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Section 5 — Casualty Information
    # ---------------------------------------------------------------
    section_header(
        "🩹", "Casualty Information",
        "Details about the casualty involved in the accident."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        num_casualties = st.number_input(
            "Number of Casualties",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Total number of casualties resulting from the accident.",
        )
    with col_b:
        casualty_class = st.selectbox(
            "Casualty Class", CATEGORICAL_OPTIONS["Casualty_class"]
        )

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        sex_of_casualty = st.selectbox(
            "Sex of Casualty", CATEGORICAL_OPTIONS["Sex_of_casualty"]
        )
    with col_d:
        age_band_casualty = st.selectbox(
            "Age Band of Casualty", CATEGORICAL_OPTIONS["Age_band_of_casualty"]
        )
    with col_e:
        casualty_severity = st.selectbox(
            "Casualty Severity", CATEGORICAL_OPTIONS["Casualty_severity"]
        )

    col_f, col_g, col_h = st.columns(3)
    with col_f:
        work_of_casuality = st.selectbox(
            "Work of Casualty", CATEGORICAL_OPTIONS["Work_of_casuality"]
        )
    with col_g:
        fitness_of_casuality = st.selectbox(
            "Fitness of Casualty", CATEGORICAL_OPTIONS["Fitness_of_casuality"]
        )
    with col_h:
        pedestrian_movement = st.selectbox(
            "Pedestrian Movement", CATEGORICAL_OPTIONS["Pedestrian_movement"]
        )

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Submit button
    # ---------------------------------------------------------------
    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True,
        type="primary",
    )

st.markdown("</div>", unsafe_allow_html=True)  # close input-container


# -----------------------------------------------------------------------------
# Build raw input dictionary from form values
# -----------------------------------------------------------------------------
raw_input = {
    # Section 1
    "Time": time_input.strftime("%H:%M"),
    "Day_of_week": day_of_week,
    "Area_accident_occured": area_accident,
    "Types_of_Junction": junction_type,
    "Light_conditions": light_conditions,
    "Weather_conditions": weather_conditions,
    "Road_surface_conditions": road_surface_conditions,
    "Road_surface_type": road_surface_type,
    # Section 2
    "Age_band_of_driver": age_band_driver,
    "Sex_of_driver": sex_of_driver,
    "Educational_level": educational_level,
    "Vehicle_driver_relation": vehicle_driver_relation,
    "Driving_experience": driving_experience,
    # Section 3
    "Type_of_vehicle": type_of_vehicle,
    "Owner_of_vehicle": owner_of_vehicle,
    "Service_year_of_vehicle": service_year,
    "Defect_of_vehicle": defect_of_vehicle,
    "Number_of_vehicles_involved": int(num_vehicles_involved),
    "Vehicle_movement": vehicle_movement,
    # Section 4
    "Lanes_or_Medians": lanes_or_medians,
    "Road_allignment": road_allignment,
    "Type_of_collision": type_of_collision,
    "Cause_of_accident": cause_of_accident,
    # Section 5
    "Number_of_casualties": int(num_casualties),
    "Casualty_class": casualty_class,
    "Sex_of_casualty": sex_of_casualty,
    "Age_band_of_casualty": age_band_casualty,
    "Casualty_severity": casualty_severity,
    "Work_of_casuality": work_of_casuality,
    "Fitness_of_casuality": fitness_of_casuality,
    "Pedestrian_movement": pedestrian_movement,
}


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
if submitted:

    st.markdown("<div class='result-container'>", unsafe_allow_html=True)

    try:
        # --------------------------------------------------------------
        # 1. Derive engineered time features
        # --------------------------------------------------------------
        time_features = derive_time_features(raw_input["Time"])

        # --------------------------------------------------------------
        # 2. Build a single-row DataFrame with the expected schema
        #    (raw categorical + numeric values; pipeline handles encoding)
        # --------------------------------------------------------------
        input_df = build_input_dataframe(raw_input, time_features)

        # --------------------------------------------------------------
        # 3. Load model + preprocessing pipeline
        # --------------------------------------------------------------
        model = load_model()
        pipeline = load_preprocessing_pipeline()

        # --------------------------------------------------------------
        # 4. Run prediction
        # --------------------------------------------------------------
        result = run_prediction(model, pipeline, input_df)

        # --------------------------------------------------------------
        # 5. Display result
        # --------------------------------------------------------------
        predicted_class = result["predicted_class"]
        probabilities = result.get("probabilities")

        color = SEVERITY_COLORS.get(predicted_class, "#3498db")

        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 6px solid {color};">
                <div class="prediction-label">Predicted Accident Severity</div>
                <div class="prediction-value" style="color: {color};">
                    {predicted_class}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------------
        # 6. Probability chart (if supported)
        # --------------------------------------------------------------
        if probabilities:
            st.markdown("### 📊 Class Probabilities")

            prob_df = pd.DataFrame(
                {
                    "Severity": list(probabilities.keys()),
                    "Probability": list(probabilities.values()),
                }
            )

            import plotly.express as px

            fig = px.bar(
                prob_df,
                x="Probability",
                y="Severity",
                orientation="h",
                text=prob_df["Probability"].apply(lambda v: f"{v*100:.2f}%"),
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Probability",
                yaxis_title="",
                xaxis_range=[0, 1],
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Also show numeric breakdown
            cols = st.columns(len(probabilities))
            for col, (cls, prob) in zip(cols, probabilities.items()):
                col.metric(
                    label=cls,
                    value=f"{prob*100:.2f}%",
                )

        # --------------------------------------------------------------
        # 7. Show derived time features (transparency)
        # --------------------------------------------------------------
        with st.expander("🔍 View Derived Time Features"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time (raw)": raw_input["Time"],
                            "Hour": time_features["Hour"],
                            "Minute": time_features["Minute"],
                            "Minutes_since_midnight": time_features["Minutes_since_midnight"],
                            "Time_period": time_features["Time_period"],
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🧾 View Final Input Sent to Model"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except FileNotFoundError as e:
        st.error(f"📁 File not found: {e}")
        st.info(
            "Make sure your trained model and preprocessing pipeline are placed "
            "in the `models/` folder as `.pkl` files."
        )

    except ValueError as e:
        st.error(f"⚠️ Invalid input or schema mismatch: {e}")
        st.info(
            "Verify that the preprocessing pipeline expects the same features "
            "that the application is sending. Check the column names and order."
        )

    except KeyError as e:
        st.error(f"🔑 Missing expected feature: {e}")
        st.info(
            "The trained pipeline expects a feature that the application did not provide. "
            "Please retrain the pipeline with the current schema or update `src/config.py`."
        )

    except Exception as e:  # last-resort safety net
        st.error(f"❌ Prediction failed: {e}")
        st.info(
            "An unexpected error occurred during prediction. Please check the "
            "application logs and ensure your model artifacts are compatible."
        )

    st.markdown("</div>", unsafe_allow_html=True)
