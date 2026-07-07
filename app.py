import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

prediction = None
confidence = 0
# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="❤️ CardioVision AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# Load Model
# ---------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------------------------------------
# Custom CSS
# ---------------------------------------------
st.markdown("""
<style>

/* Google Font */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}

/* Background */

.stApp{

background:linear-gradient(-45deg,
#070b1f,
#121b3a,
#081c3d,
#111827);

background-size:400% 400%;

animation:gradientBG 15s ease infinite;

color:white;

}

/* Animated Background */

@keyframes gradientBG{

0%{background-position:0% 50%;}

50%{background-position:100% 50%;}

100%{background-position:0% 50%;}

}

/* Hide Streamlit */

header{

visibility:hidden;

}

footer{

visibility:hidden;

}

#MainMenu{

visibility:hidden;

}

/* Title */

.mainTitle{

font-size:58px;

font-weight:800;

text-align:center;

background:linear-gradient(90deg,#ff4b4b,#ff9800,#ffd700);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

animation:pulse 2s infinite;

}

/* Subtitle */

.subTitle{

text-align:center;

font-size:22px;

color:#d6d6d6;

margin-top:-10px;

margin-bottom:35px;

}

/* Pulse */

@keyframes pulse{

0%{

transform:scale(1);

}

50%{

transform:scale(1.04);

}

100%{

transform:scale(1);

}

}

/* Glass Card */

.glass{

background:rgba(255,255,255,.06);

border-radius:22px;

padding:22px;

backdrop-filter:blur(15px);

border:1px solid rgba(255,255,255,.15);

box-shadow:0px 10px 35px rgba(0,0,0,.4);

transition:.4s;

}

.glass:hover{

transform:translateY(-8px);

box-shadow:0px 15px 45px rgba(255,80,80,.5);

}

/* Metric */

[data-testid="metric-container"]{

background:rgba(255,255,255,.08);

border-radius:18px;

padding:18px;

border:1px solid rgba(255,255,255,.1);

box-shadow:0px 5px 18px rgba(255,0,0,.2);

}

/* Labels */

label{

color:#FFD700 !important;

font-size:18px !important;

font-weight:bold !important;

}

/* Inputs */

.stSelectbox div[data-baseweb="select"]{

background:#1d2438;

border-radius:10px;

color:white;

}

.stNumberInput input{

background:#1d2438;

color:white;

}

.stSlider{

padding-top:10px;

}

/* Button */

.stButton>button{

width:100%;

height:60px;

font-size:22px;

font-weight:bold;

border-radius:18px;

border:none;

color:white;

background:linear-gradient(90deg,#ff0000,#ff512f);

box-shadow:0px 12px 35px rgba(255,0,0,.45);

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.03);

background:linear-gradient(90deg,#ff512f,#dd2476);

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#0b132b;

}

section[data-testid="stSidebar"] *{

color:white;

}

/* HR */

hr{

border:1px solid rgba(255,255,255,.1);

}

</style>

""", unsafe_allow_html=True)

# ---------------------------------------------
# HERO
# ---------------------------------------------

st.markdown("""

<div class="mainTitle">

❤️ CardioVision AI

</div>

<div class="subTitle">

AI Powered Heart Disease Prediction & Medical Analytics Dashboard

</div>

""", unsafe_allow_html=True)

# ---------------------------------------------
# Dashboard Metrics
# ---------------------------------------------

m1,m2,m3,m4 = st.columns(4)

with m1:
    st.metric("🤖 Model","Random Forest")

with m2:
    st.metric("🎯 Accuracy","87.5%")

with m3:
    st.metric("📊 Features","15")

with m4:
    st.metric("⚡ Status","ONLINE")

st.divider()

# ---------------------------------------------
# Sidebar
# ---------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/240/heart-with-pulse.png",
        width=120
    )

    st.title("❤️ CardioVision AI")

    st.success("Machine Learning Dashboard")

    st.markdown("---")

    st.subheader("🩺 Model")

    st.write("Random Forest Classifier")

    st.write("Accuracy : **87.5%**")

    st.markdown("---")

    st.subheader("⚙ Technology")

    st.write("• Python")

    st.write("• Scikit-Learn")

    st.write("• Streamlit")

    st.write("• Pandas")

    st.write("• Plotly")

    st.markdown("---")

    st.subheader("📅 Today")

    st.info(datetime.now().strftime("%d %B %Y"))

    st.markdown("---")

    st.success("🟢 System Ready")

st.markdown("---")
# ==========================================================
# PATIENT INFORMATION
# ==========================================================

st.markdown("""
<div class="glass">

<h2 style='text-align:center;color:#FFD700;'>
👤 Patient Information
</h2>

<p style='text-align:center;color:#CFCFCF;'>
Enter the patient's medical details below.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

left,right = st.columns([1,1],gap="large")

# ==========================================================
# LEFT COLUMN
# ==========================================================

with left:

    st.markdown("### 🧍 Personal Details")

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    dataset = st.selectbox(
        "Hospital Dataset",
        [
            "Cleveland",
            "Hungary",
            "Switzerland",
            "VA"
        ]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        80,
        220,
        120
    )

    chol = st.slider(
        "Serum Cholesterol",
        100,
        600,
        200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [
            "TRUE",
            "FALSE"
        ]
    )

# ==========================================================
# RIGHT COLUMN
# ==========================================================

with right:

    st.markdown("### ❤️ Cardiac Examination")

    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "st-t abnormality",
            "lv hypertrophy"
        ]
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [
            "TRUE",
            "FALSE"
        ]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        6.0,
        1.0
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [
            "upsloping",
            "flat",
            "downsloping"
        ]
    )

    ca = st.slider(
        "Major Vessels",
        0,
        4,
        0
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "normal",
            "fixed defect",
            "reversable defect"
        ]
    )

st.write("")

# ==========================================================
# QUICK SUMMARY
# ==========================================================

st.markdown("""
<div class="glass">

<h3 style="color:#FFD700;">
📋 Patient Summary
</h3>

</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "👤 Age",
    age
)

c2.metric(
    "🧬 Gender",
    sex
)

c3.metric(
    "❤️ Blood Pressure",
    trestbps
)

c4.metric(
    "🩸 Cholesterol",
    chol
)

st.write("")

# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.button("Analyze", width="stretch")
predict = st.button(
    "❤️ Analyze Patient",
    width="stretch"
)
import time

if predict:

    with st.spinner("🧠 AI is analyzing patient data..."):
        time.sleep(1.5)

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0].max()

    sample = pd.DataFrame({

        "id":[1],
        "age":[age],
        "sex":[sex],
        "dataset":[dataset],
        "cp":[cp],
        "trestbps":[trestbps],
        "chol":[chol],
        "fbs":[fbs=="TRUE"],
        "restecg":[restecg],
        "thalch":[thalach],
        "exang":[exang=="TRUE"],
        "oldpeak":[oldpeak],
        "slope":[slope],
        "ca":[ca],
        "thal":[thal]

    })

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0].max()

    confidence = round(probability * 100,2)

    st.markdown("---")

    st.header("🩺 AI Diagnosis Report")

    left,right = st.columns([1,1])

    # -----------------------------------------
    # LEFT RESULT CARD
    # -----------------------------------------

    with left:

        if prediction==1:

            st.error("## ⚠️ HIGH RISK")

            st.markdown("""
### Recommendation

🔴 Consult a Cardiologist

🔴 Maintain Healthy Diet

🔴 Daily Exercise

🔴 Monitor Blood Pressure

🔴 Reduce Cholesterol

""")

        else:

            st.success("## ✅ LOW RISK")

            st.markdown("""
### Recommendation

🟢 Continue Healthy Lifestyle

🟢 Regular Exercise

🟢 Annual Checkup

🟢 Balanced Diet

🟢 Stay Active

""")

    # -----------------------------------------
    # RIGHT CONFIDENCE GAUGE
    # -----------------------------------------

    with right:

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=confidence,

            title={'text':"AI Confidence (%)"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':"#00FFAA"},

                'steps':[

                    {'range':[0,40],'color':"#8B0000"},

                    {'range':[40,70],'color':"#FFA500"},

                    {'range':[70,100],'color':"#228B22"}

                ]

            }

        ))

        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=350
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("📊 Patient Health Analysis")

    c1,c2 = st.columns(2)

    # -----------------------------------------
    # BAR CHART
    # -----------------------------------------

    with c1:

        values = {

            "Blood Pressure":trestbps,

            "Cholesterol":chol,

            "Heart Rate":thalach

        }

        fig = px.bar(

            x=list(values.keys()),

            y=list(values.values()),

            color=list(values.values()),

            title="Health Parameters"

        )

        fig.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font_color="white"

        )

        st.plotly_chart(fig, width="stretch")

    # -----------------------------------------
    # DONUT CHART
    # -----------------------------------------

    with c2:

        fig = go.Figure(

            data=[

                go.Pie(

                    labels=["Confidence","Remaining"],

                    values=[confidence,100-confidence],

                    hole=.70

                )

            ]

        )

        fig.update_layout(

            title="Prediction Confidence",

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            height=400

        )

        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("📋 Patient Summary")

    table = pd.DataFrame({
    "Parameter":[
        "Age",
        "Gender",
        "Blood Pressure",
        "Cholesterol",
        "Heart Rate",
        "Chest Pain"
    ],
    "Value":[
        str(age),
        str(sex),
        str(trestbps),
        str(chol),
        str(thalach),
        str(cp)
    ]
})

    st.dataframe(df, width="stretch")

    st.success("✔ AI Analysis Completed Successfully")

    st.balloons()
    st.markdown("---")

st.subheader("🤖 AI Doctor Assistant")

if prediction == 1:

    st.warning("""
### ⚠ AI Recommendation

Your health parameters indicate an increased risk.

Recommended Actions

✔ Schedule a cardiologist appointment.

✔ Reduce saturated fats.

✔ Walk at least 30 minutes daily.

✔ Avoid smoking.

✔ Reduce stress.

✔ Monitor blood pressure weekly.

✔ Maintain healthy body weight.

""")

else:

    st.success("""
### ✅ AI Recommendation

Excellent!

Current values indicate a lower heart disease risk.

Suggestions

✔ Continue regular exercise.

✔ Healthy diet.

✔ Annual heart checkup.

✔ Stay hydrated.

✔ Good sleep.

✔ Maintain cholesterol.

✔ Keep active.

""")
st.markdown("### ❤️ Risk Meter")

risk = confidence

st.progress(risk/100)

if risk < 40:

    st.success("🟢 Low Risk")

elif risk < 70:

    st.warning("🟠 Moderate Risk")

else:

    st.error("🔴 High Risk")
report = f"""
CARDIOVISION AI REPORT

Prediction:
{"HIGH RISK" if prediction==1 else "LOW RISK"}

Confidence:
{confidence} %

Age:
{age}

Gender:
{sex}

Blood Pressure:
{trestbps}

Cholesterol:
{chol}

Heart Rate:
{thalach}
"""

st.download_button(

label="📄 Download Report",

data=report,

file_name="CardioVision_Report.txt",

mime="text/plain"

)
st.markdown("---")

st.subheader("💡 Daily Health Tips")

tips=[

"🥗 Eat more vegetables",

"🚶 Walk 30 minutes daily",

"💧 Drink enough water",

"😴 Sleep 7-8 hours",

"❤️ Reduce cholesterol intake",

"🧘 Practice meditation",

"🚭 Avoid smoking"

]

for tip in tips:

    st.info(tip)
st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:25px;
background:rgba(255,255,255,.08);
border-radius:20px;
margin-top:40px;
">

<h2>❤️ CardioVision AI</h2>

<p>Developed for CodeAlpha Machine Learning Internship</p>

<p>Python • Streamlit • Scikit-Learn • Plotly</p>

</div>
""", unsafe_allow_html=True)