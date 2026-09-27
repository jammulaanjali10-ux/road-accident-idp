"""
Central configuration for the Road Accident Severity Prediction app.

This file holds:
  - The exact feature names expected by the trained model
  - The dropdown options shown in the Streamlit UI
  - The class labels produced by the classifier
  - The color palette used for the result visualisation
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (relative to project root, Cloud-deployable)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

MODEL_PATH = MODELS_DIR / "accident_severity_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"

# -----------------------------------------------------------------------------
# Expected prediction classes (must match the label encoder used in training)
# -----------------------------------------------------------------------------
SEVERITY_CLASSES = ["Slight Injury", "Serious Injury", "Fatal injury"]

SEVERITY_COLORS = {
    "Slight Injury": "#2ecc71",   # green
    "Serious Injury": "#f39c12",  # orange
    "Fatal injury": "#e74c3c",    # red
}

# -----------------------------------------------------------------------------
# The exact feature columns the trained model expects (in training order).
# Used to validate the DataFrame before prediction.
#
# NOTE: Hour / Minute / Minutes_since_midnight / Time_period are engineered
#       automatically from the `Time` field and MUST NOT be asked from user.
# -----------------------------------------------------------------------------
EXPECTED_FEATURES = [
    # Time-derived (engineered)
    "Hour",
    "Minute",
    "Minutes_since_midnight",
    "Time_period",

    # Raw user inputs — Section 1
    "Day_of_week",
    "Area_accident_occured",
    "Types_of_Junction",
    "Light_conditions",
    "Weather_conditions",
    "Road_surface_conditions",
    "Road_surface_type",

    # Section 2 — Driver
    "Age_band_of_driver",
    "Sex_of_driver",
    "Educational_level",
    "Vehicle_driver_relation",
    "Driving_experience",

    # Section 3 — Vehicle
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Number_of_vehicles_involved",
    "Vehicle_movement",

    # Section 4 — Road & Collision
    "Lanes_or_Medians",
    "Road_allignment",
    "Type_of_collision",
    "Cause_of_accident",

    # Section 5 — Casualty
    "Number_of_casualties",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
]

# Numeric fields the user enters directly (everything else is categorical)
NUMERIC_FIELDS = [
    "Number_of_vehicles_involved",
    "Number_of_casualties",
]

# -----------------------------------------------------------------------------
# Dropdown options — adjust to match your dataset's unique values
# -----------------------------------------------------------------------------
CATEGORICAL_OPTIONS = {
    "Day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ],
    "Area_accident_occured": [
        "Residential areas", "Office areas", "Recreational areas",
        "Industrial areas", "Church areas", "Market areas",
        "Rural village areas", "Outside rural areas",
        "Hospital areas", "School areas", "Unknown",
    ],
    "Types_of_Junction": [
        "No junction", "Y Shape", "Crossroad", "T Shape",
        "O Shape", "Other", "Roundabout", "U Turn",
    ],
    "Light_conditions": [
        "Daylight", "Darkness - lights lit", "Darkness - no lighting",
        "Darkness - lights unlit", "Darkness - unknown lighting",
    ],
    "Weather_conditions": [
        "Normal", "Raining", "Raining and Windy", "Cloudy",
        "Other", "Windy", "Snow", "Unknown", "Fog or mist",
    ],
    "Road_surface_conditions": [
        "Dry", "Wet or damp", "Snow", "Flood over 3cm. deep", "Unknown",
    ],
    "Road_surface_type": [
        "Asphalt roads", "Earth roads", "Gravel roads",
        "Other", "Asphalt roads with some distress", "Unknown",
    ],

    # Driver
    "Age_band_of_driver": [
        "18-30", "31-50", "Over 51", "Unknown", "Under 18",
    ],
    "Sex_of_driver": ["Male", "Female", "Unknown"],
    "Educational_level": [
        "Above high school", "Junior high school", "Elementary school",
        "High school", "Unknown", "Illiterate", "Writing & reading",
    ],
    "Vehicle_driver_relation": [
        "Employee", "Owner", "Other", "Unknown",
    ],
    "Driving_experience": [
        "5-10yr", "2-5yr", "Above 10yr", "1-2yr", "Below 1yr",
        "No Licence", "unknown",
    ],

    # Vehicle
    "Type_of_vehicle": [
        "Automobile", "Lorry (41?100Q)", "Pick up upto 10Q",
        "Turismo", "Motorcycle", "Public (12? seats)",
        "Public (13?45 seats)", "Long lorry", "Public (> 45 seats)",
        "Stationwagen", "Ridden horse", "Other", "Bajaj", "Bicycle",
        "Unknown",
    ],
    "Owner_of_vehicle": [
        "Owner", "Governmental", "Organization", "Other", "Unknown",
    ],
    "Service_year_of_vehicle": [
        "2-5yrs", "Above 10yr", "5-10yrs", "1-2yr", "Below 1yr",
        "Unknown",
    ],
    "Defect_of_vehicle": [
        "No defect", "7", "5", "Unknown",
    ],
    "Vehicle_movement": [
        "Going straight", "U-Turn", "Moving Backward", "Turnover",
        "Waiting to move", "Getting off", "Parked", "Stopping",
        "Entering a junction", "Other", "Reversing", "Unknown",
    ],

    # Road & Collision
    "Lanes_or_Medians": [
        "Two-way (divided with broken lines for overtaking)",
        "Undivided Two way", "Double carriageway (median)",
        "other", "One way", "Two-way (divided with solid lines for overtaking)",
        "Unknown",
    ],
    "Road_allignment": [
        "Tangent road with flat terrain", "Tangent road with mild grade and flat terrain",
        "Gentle horizontal curve", "Tangent road with rolling terrain",
        "Sharp reverse curve", "Steep grade downward with mountainous terrain and/or road alignment",
        "Steep grade upward with mountainous terrain and/or road alignment",
        "Other", "Unknown",
    ],
    "Type_of_collision": [
        "Vehicle with vehicle collision", "Collision with roadside objects",
        "Collision with pedestrians", "Rollover", "Fall from vehicle",
        "Collision with animals", "Unknown", "Drifted off road",
        "With Train", "Other",
    ],
    "Cause_of_accident": [
        "No distancing", "Changing lane to the right", "Changing lane to the left",
        "Driving carelessly", "No priority to vehicle", "Moving Backward",
        "No priority to pedestrian", "Other", "Overturning", "Driving at high speed",
        "Drunk driving", "Driving to the left", "Unknown", "Overloading",
        "Driving sleepy", "Improper parking",
    ],

    # Casualty
    "Casualty_class": [
        "Driver or rider", "na", "Pedestrian", "Passenger",
    ],
    "Sex_of_casualty": ["Male", "Female", "na"],
    "Age_band_of_casualty": [
        "18-30", "31-50", "Over 51", "Under 18", "5", "na", "Unknown",
    ],
    "Casualty_severity": [
        "Slight Injury", "Serious Injury", "Fatal injury", "na",
    ],
    "Work_of_casuality": [
        "Driver", "Self-employed", "Employee", "Student", "Unemployed",
        "Unknown", "na", "Other",
    ],
    "Fitness_of_casuality": [
        "Normal", "Deaf", "Other", "Blind", "na", "Unknown",
    ],
    "Pedestrian_movement": [
        "Not a Pedestrian", "Crossing from driver's side", "Crossing from passenger's sides",
        "Unknown", "Walking along in carriageway", "Standing or sitting in carriageway",
        "In carriageway, lying down", "na", "Other",
    ],
}

# -----------------------------------------------------------------------------
# Time-period bucketing logic (used for Time_period feature)
# -----------------------------------------------------------------------------
def time_period_from_hour(hour: int) -> str:
    """
    Map a 24h hour value to a human-readable time period.

    Adjust these thresholds to match whatever bucketing logic was used
    when the model was trained.
    """
    if 0 <= hour < 5:
        return "Late Night"
    elif 5 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 20:
        return "Evening"
    elif 20 <= hour < 24:
        return "Night"
    else:
        return "Unknown"
