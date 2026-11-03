import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import hashlib
import hmac
import secrets
import os
import time
from datetime import datetime
from pathlib import Path

# =========================================================
# DiaNova AI | The Wellness Journal
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
CSS_PATH = BASE_DIR / "frontend" / "style.css"
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
DB_PATH = BASE_DIR / "dianova_users.db"

st.set_page_config(
    page_title="DiaNova AI | Smarter Insights",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Extra styling for the new editorial / Pinterest-inspired layout.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700;800&display=swap');

:root {
  --forest:#294634; --sage:#547a61; --cream:#faf8f2;
  --peach:#f4c7ae; --butter:#f5e8ad; --mint:#e5eee2;
}
.stApp { background:#faf8f2; }
.block-container { max-width:1420px; padding-top:1.5rem; }
h1,h2,h3 { color:#294634 !important; }
[data-testid="stSidebar"] { background:#eaf0e7; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding: .45rem .65rem; border-radius:12px; margin:.15rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background:#dce9dd;
}
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
  background:#547a61; color:white !important; border:0;
  border-radius:24px; font-weight:700; padding:.65rem 1.2rem;
}
.stButton>button:hover, .stFormSubmitButton>button:hover,
.stDownloadButton>button:hover { background:#355642; transform:translateY(-2px); }
[data-testid="stMetric"] {
  background:white; border:1px solid #e7eae1; border-radius:20px;
  padding:18px 20px; box-shadow:0 7px 22px rgba(45,68,49,.045);
}
[data-testid="stMetricValue"] { color:#355642 !important; }
[data-testid="stFileUploader"] {
  background:#f0f5ed; border:1.5px dashed #91ac94;
  border-radius:20px; padding:16px;
}
[data-testid="stDataFrame"] { border:1px solid #e7eae1; border-radius:16px; }
[data-testid="stExpander"] { background:white; border:1px solid #e7eae1; border-radius:15px; }
hr { border-color:#e4e8de; }
.journal-hero {
  background:linear-gradient(115deg,#dcebdd 0%,#f7edda 58%,#f5d9c8 100%);
  border:1px solid #e8e6d9; border-radius:30px; padding:38px 42px;
  min-height:260px; position:relative; overflow:hidden; margin-bottom:22px;
}
.journal-hero:after {
  content:"✳"; position:absolute; right:45px; top:10px;
  font-size:145px; color:rgba(84,122,97,.13); transform:rotate(15deg);
}
.eyebrow {
  color:#547a61; text-transform:uppercase; letter-spacing:2px;
  font-size:11px; font-weight:800; margin-bottom:12px;
}
.journal-title {
  font-family:'Playfair Display',serif; font-size:clamp(35px,4vw,58px);
  line-height:1.05; color:#294634; margin:0 0 12px;
}
.journal-copy { color:#526c59; font-size:15px; max-width:650px; line-height:1.8; }
.pill {
  display:inline-block; background:#fff9; color:#355642; border:1px solid #d8e2d5;
  border-radius:30px; padding:7px 13px; font-size:11px; font-weight:800;
  letter-spacing:.5px; margin:5px 5px 0 0;
}
.section-kicker {
  color:#78907a; text-transform:uppercase; letter-spacing:2px;
  font-size:10px; font-weight:800; margin:8px 0;
}
.pin-card {
  background:#fff; border:1px solid #e9ebe3; border-radius:22px;
  padding:23px; min-height:155px; box-shadow:0 8px 24px rgba(45,68,49,.045);
}
.pin-card h3 { font-family:'Playfair Display',serif; font-size:23px; margin:5px 0 8px; }
.pin-card p { color:#69786c; font-size:13px; line-height:1.65; }
.pin-icon { font-size:25px; }
.pin-peach { background:#f8e5d9; }
.pin-sage { background:#e3eee1; }
.pin-butter { background:#f6efcf; }
.pin-lilac { background:#eee8f1; }
.note-card {
  background:#f1f4ec; border-left:4px solid #78977b; border-radius:5px 16px 16px 5px;
  padding:16px 20px; color:#526c59; font-size:13px; line-height:1.7;
}
.result-card {
  background:linear-gradient(135deg,#e3eee1,#f8eee0); border:1px solid #dce7d8;
  border-radius:24px; padding:25px; margin-top:18px;
}
.result-card.positive { background:linear-gradient(135deg,#f8e5d9,#f6efcf); }
.big-number { font-family:'Playfair Display',serif; font-size:48px; color:#294634; }
.small-muted { color:#77867a; font-size:12px; }
.page-heading { font-family:'Playfair Display',serif; font-size:38px; color:#294634; }
.dianova-footer { text-align:center; padding:30px 10px; color:#879386; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# Load the user's CSS theme too, if present.
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# FINAL FORM-CONTROL OVERRIDES
# Keep this block AFTER style.css so older theme rules cannot make typed text
# dark-on-dark or leave password fields with mismatched inner/outer backgrounds.
st.markdown("""
<style>
/* Text, email, password and number inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background-color: #ffffff !important;
    color: #243b2d !important;
    -webkit-text-fill-color: #243b2d !important;
    caret-color: #243b2d !important;
    border: 1px solid #b8c8b8 !important;
    border-radius: 12px !important;
    opacity: 1 !important;
}

/* The BaseWeb outer input wrappers */
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    border: 1px solid #b8c8b8 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* Focus state */
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border: 2px solid #547a61 !important;
    box-shadow: 0 0 0 3px rgba(84, 122, 97, 0.14) !important;
}

/* Placeholder and labels */
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder {
    color: #78877b !important;
    -webkit-text-fill-color: #78877b !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    color: #294634 !important;
    font-weight: 600 !important;
}
div[data-testid="InputInstructions"],
div[data-testid="stTextInput"] [data-testid="InputInstructions"] {
    color: #718074 !important;
}

/* Password reveal button */
div[data-testid="stTextInput"] button {
    color: #355642 !important;
    background: #ffffff !important;
    border: 0 !important;
}

/* Prevent browser autofill from painting a dark background */
div[data-testid="stTextInput"] input:-webkit-autofill,
div[data-testid="stTextInput"] input:-webkit-autofill:hover,
div[data-testid="stTextInput"] input:-webkit-autofill:focus {
    -webkit-text-fill-color: #243b2d !important;
    -webkit-box-shadow: 0 0 0 1000px #ffffff inset !important;
    box-shadow: 0 0 0 1000px #ffffff inset !important;
    transition: background-color 9999s ease-out 0s;
}
</style>
""", unsafe_allow_html=True)




# =========================================================
# DiaNova account and personal-history database
# Passwords are stored as salted PBKDF2 hashes, never plaintext.
# =========================================================
def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Profile-only fields. These ALTER statements safely preserve existing accounts.
        user_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        profile_columns = {
            "age": "INTEGER",
            "dietary_preference": "TEXT",
            "activity_level": "TEXT",
            "wellness_goal": "TEXT",
        }
        for column_name, column_type in profile_columns.items():
            if column_name not in user_columns:
                connection.execute(
                    f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"
                )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                source TEXT NOT NULL,
                file_name TEXT,
                record_count INTEGER NOT NULL DEFAULT 1,
                positive_count INTEGER,
                negative_count INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Safely migrate existing databases without deleting saved history.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(prediction_history)"
            ).fetchall()
        }

        if "positive_count" not in existing_columns:
            connection.execute(
                "ALTER TABLE prediction_history ADD COLUMN positive_count INTEGER"
            )

        if "negative_count" not in existing_columns:
            connection.execute(
                "ALTER TABLE prediction_history ADD COLUMN negative_count INTEGER"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS assistant_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


def password_digest(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310000)
    return salt.hex() + "$" + digest.hex()


def verify_password(password, stored_hash):
    try:
        salt_hex, expected_hex = stored_hash.split("$", 1)
        actual = password_digest(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        return hmac.compare_digest(actual, expected_hex)
    except (ValueError, TypeError):
        return False


def register_user(full_name, email, password):
    try:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO users(full_name,email,password_hash,created_at) VALUES(?,?,?,?)",
                (full_name.strip(), email.strip().lower(), password_digest(password),
                 datetime.now().isoformat(timespec="seconds"))
            )
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(email, password):
    with get_connection() as connection:
        user = connection.execute(
            "SELECT id,full_name,email,password_hash FROM users WHERE email=?",
            (email.strip().lower(),)
        ).fetchone()
    if user and verify_password(password, user["password_hash"]):
        return {"id": user["id"], "name": user["full_name"], "email": user["email"]}
    return None


def save_prediction(
    user_id,
    prediction,
    probability,
    source,
    file_name=None,
    record_count=1,
    positive_count=None,
    negative_count=None
):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO prediction_history (
                user_id, prediction, probability, source, file_name,
                record_count, positive_count, negative_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            int(prediction),
            float(probability),
            source,
            file_name,
            int(record_count),
            None if positive_count is None else int(positive_count),
            None if negative_count is None else int(negative_count),
            datetime.now().isoformat(timespec="seconds")
        ))


def get_user_history(user_id):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                prediction, probability, source, file_name, record_count,
                positive_count, negative_count, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def save_chat_message(user_id, role, message):
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO assistant_messages(user_id,role,message,created_at) VALUES(?,?,?,?)",
            (user_id, role, message, datetime.now().isoformat(timespec="seconds"))
        )


def get_chat_history(user_id):
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT role,message FROM assistant_messages WHERE user_id=? ORDER BY id",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


initialise_database()


def authentication_screen():
    st.markdown("""
    <div class="journal-hero">
      <div class="eyebrow">WELCOME TO YOUR PERSONAL WELLNESS SPACE</div>
      <div class="journal-title">A little more insight.<br>A lot more you.</div>
      <div class="journal-copy">Sign in to keep your DiaNova AI activity and insights private to your account.</div>
    </div>
    """, unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("dianova_login"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in  →", use_container_width=True)
        if submitted:
            user = authenticate_user(email, password)
            if user:
                st.session_state["dianova_user"] = user
                st.session_state.pop("dianova_chat", None)
                st.rerun()
            else:
                st.error("Email or password is incorrect.")
    with register_tab:
        with st.form("dianova_register"):
            name = st.text_input("Full name")
            new_email = st.text_input("Email address", key="register_email")
            new_password = st.text_input("Create password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm password", type="password")
            agreed = st.checkbox("I understand DiaNova AI is an educational tool, not a medical service.")
            submitted = st.form_submit_button("Create my account  →", use_container_width=True)
        if submitted:
            if not name.strip() or not new_email.strip() or not new_password:
                st.error("Please complete all fields.")
            elif "@" not in new_email or "." not in new_email.split("@")[-1]:
                st.error("Enter a valid email address.")
            elif len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            elif not agreed:
                st.error("Please confirm the educational-use notice.")
            else:
                ok, message = register_user(name, new_email, new_password)
                (st.success if ok else st.error)(message)


# Require authentication before showing any application pages.
if "dianova_user" not in st.session_state:
    authentication_screen()
    st.stop()

CURRENT_USER = st.session_state["dianova_user"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info("Check that models/best_model.pkl exists and that the virtual environment has the required packages.")
    st.stop()


FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

VALID_RANGES = {
    "HighBP": (0, 1), "HighChol": (0, 1), "CholCheck": (0, 1),
    "BMI": (12, 98), "Smoker": (0, 1), "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1), "PhysActivity": (0, 1),
    "Fruits": (0, 1), "Veggies": (0, 1), "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1), "NoDocbcCost": (0, 1), "GenHlth": (1, 5),
    "MentHlth": (0, 30), "PhysHlth": (0, 30), "DiffWalk": (0, 1),
    "Sex": (0, 1), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)
}

BINARY_LABELS = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in last 5 years", "Smoker": "Smoked 100+ cigarettes",
    "Stroke": "History of stroke", "HeartDiseaseorAttack": "Heart disease or heart attack",
    "PhysActivity": "Physical activity", "Fruits": "Consumes fruit daily",
    "Veggies": "Consumes vegetables daily", "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Healthcare coverage", "NoDocbcCost": "Could not see doctor due to cost",
    "DiffWalk": "Difficulty walking"
}


def make_prediction(input_data):
    frame = pd.DataFrame([input_data], columns=FEATURES)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return prediction, probability


def validate_dataset(dataframe):
    missing = [column for column in FEATURES if column not in dataframe.columns]
    if missing:
        return False, "Missing required columns: " + ", ".join(missing)

    frame = dataframe[FEATURES].copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.empty:
        return False, "The uploaded dataset has no records."

    if frame.isna().any().any():
        return False, "Required feature columns contain missing or non-numeric values."

    for column, (low, high) in VALID_RANGES.items():
        if not frame[column].between(low, high).all():
            return False, f"'{column}' contains values outside the allowed range ({low} to {high})."

    return True, frame


def page_header(eyebrow, title, description):
    st.markdown(
        f'<div class="section-kicker">{eyebrow}</div>'
        f'<div class="page-heading">{title}</div>'
        f'<p style="color:#718074;margin-top:4px">{description}</p>',
        unsafe_allow_html=True
    )


def result_display(prediction, probability):
    positive = prediction == 1
    label = "Positive category" if positive else "Negative category"
    explanation = (
        "The model placed this record in the positive diabetes-related indicator category."
        if positive else
        "The model placed this record in the negative diabetes-related indicator category."
    )
    style = "positive" if positive else ""
    st.markdown(
        f"""
        <div class="result-card {style}">
          <div class="section-kicker">YOUR MODEL RESULT</div>
          <div class="big-number">{'01' if positive else '00'}</div>
          <h2 style="margin:0">{label}</h2>
          <p style="color:#526c59">{explanation}</p>
          <div class="small-muted">Model-estimated positive-class score</div>
          <h2 style="margin:3px 0 0;color:#355642">{probability*100:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("This is a machine-learning output, not a diagnosis or a clinically validated estimate of future diabetes risk.")


def show_dataframe_charts(dataframe, prefix=""):
    left, right = st.columns(2)
    with left:
        st.markdown("**BMI spread**")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(data=dataframe, x="BMI", bins=24, color="#78977b", ax=ax)
        ax.set_xlabel("BMI"); ax.set_ylabel("Records")
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**Age-category mix**")
        age_labels = {
            1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
            5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
            9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79",
            13: "80+"
        }
        counts = dataframe["Age"].value_counts().sort_index()
        chart_labels = [age_labels.get(int(value), str(value)) for value in counts.index]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(chart_labels, counts.values, color="#e9ad91")
        ax.set_xlabel("Age range (years)"); ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=45)
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# =========================================================
# Navigation — deliberately styled as a journal menu.
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 5px 22px">
          <div style="font-size:30px">🌿</div>
          <div style="font-family:'Playfair Display',serif;font-size:27px;font-weight:700;color:#294634">DiaNova</div>
          <div style="font-size:10px;letter-spacing:2px;color:#78907a;font-weight:800">THE WELLNESS JOURNAL</div>
          <p style="font-size:12px;color:#718074;margin-top:12px">Smarter insights. Healthier futures.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="note-card">Welcome back, <b>{CURRENT_USER["name"]}</b> 🌿</div>', unsafe_allow_html=True)
    if st.button("Log out", use_container_width=True):
        st.session_state.pop("dianova_user", None)
        st.session_state.pop("dianova_chat", None)
        st.rerun()
    st.markdown('<div class="section-kicker">YOUR SPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["The Journal", "Health Check-in", "Dataset Atelier", "My Insights", "DiaNova Companion", "Profile"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div class="note-card">A little reminder: data can offer insight, but your healthcare professional offers care.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br><div class='small-muted'>MADE WITH CARE · DIANOVA AI</div>", unsafe_allow_html=True)


# =========================================================
# 1. HOME / JOURNAL
# =========================================================
if page == "The Journal":
    st.markdown(
        """
        <div class="journal-hero">
          <div class="eyebrow">A FRESH PERSPECTIVE ON HEALTH DATA</div>
          <div class="journal-title">Your health story,<br>with a little more insight.</div>
          <div class="journal-copy">
            Welcome to DiaNova AI — an inviting space to explore health indicators,
            understand patterns and experiment with machine-learning insights.
          </div>
          <div style="margin-top:18px">
            <span class="pill">✳ HEALTH ANALYTICS</span>
            <span class="pill">✳ MACHINE LEARNING</span>
            <span class="pill">✳ DATA EXPLORATION</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-kicker">THE PROJECT AT A GLANCE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Survey records", "253,680")
    c2.metric("Input indicators", "21")
    c3.metric("Algorithms compared", "03")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">PICK YOUR NEXT STOP</div>', unsafe_allow_html=True)
    a, b, c = st.columns([1.1, 1, 1])
    with a:
        st.markdown('<div class="pin-card pin-sage"><div class="pin-icon">🪴</div><h3>Health check-in</h3><p>Explore a single health profile and see how the trained model classifies its indicators.</p><span class="pill">01 · EXPLORE</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="pin-card pin-peach"><div class="pin-icon">🗂️</div><h3>Dataset atelier</h3><p>Bring a compatible CSV or Excel dataset, inspect it and create batch predictions.</p><span class="pill">02 · UPLOAD</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="pin-card pin-butter"><div class="pin-icon">📈</div><h3>My insights</h3><p>Explore your own saved prediction history and personal analytics.</p><span class="pill">03 · DISCOVER</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="note-card"><b>THE DATA NOTE</b><br>'
        'The dataset uses self-reported survey indicators. Its positive class combines reported prediabetes and diabetes; it is not a confirmed diagnosis.</div>',
        unsafe_allow_html=True
    )

    with st.expander("About this project & responsible use"):
        st.markdown("**Built with:** Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn")
        st.markdown("**Selected model:** Logistic Regression (chosen for the highest F1 score in the held-out evaluation).")
        st.caption("Evaluation: accuracy 73.15% · precision 31.07% · recall 76.11% · F1 score 44.13%. Accuracy alone can be misleading because the dataset is imbalanced.")
        st.markdown(
            '<div class="note-card"><b>IMPORTANT HEALTH DISCLAIMER</b><br>'
            'DiaNova AI is an educational machine-learning project based on survey-reported indicators. '
            'Its output is not a medical diagnosis and is not a clinically validated estimate of future diabetes risk. '
            'Always consult a qualified healthcare professional for medical advice.</div>',
            unsafe_allow_html=True
        )


# =========================================================
# 2. INDIVIDUAL PREDICTION
# =========================================================
elif page == "Health Check-in":
    page_header("A PERSONAL DATA CHECK-IN", "Let's look at the indicators.", "Complete the sections below. The model will classify the information you enter.")
    st.markdown("<div class=\"note-card\">Age, education and income are shown as familiar ranges/labels. DiaNova converts them to the dataset\'s numeric codes automatically.</div>", unsafe_allow_html=True)

    with st.form("dianova_checkin"):
        st.markdown("### 01 · The basics")
        x1, x2, x3 = st.columns(3)
        with x1:
            bmi = st.number_input("BMI", min_value=12, max_value=98, value=25)
        with x2:
            age_options = {
                1: "18–24 years", 2: "25–29 years", 3: "30–34 years",
                4: "35–39 years", 5: "40–44 years", 6: "45–49 years",
                7: "50–54 years", 8: "55–59 years", 9: "60–64 years",
                10: "65–69 years", 11: "70–74 years", 12: "75–79 years",
                13: "80 years or older"
            }
            selected_age = st.selectbox(
                "Age group",
                options=list(age_options.keys()),
                format_func=lambda value: age_options[value],
                help="Choose the age range that applies. DiaNova converts it to the dataset's original code automatically."
            )
            age = selected_age
        with x3:
            sex = st.selectbox("Sex", ["Female", "Male"],
                               help="This dataset records sex using a two-category survey field.")

        st.markdown("---")
        st.markdown("### 02 · Health indicators")
        st.caption("For yes/no questions, select the answer that best matches the survey definition shown in each label.")
        binary_values = {}
        binary_fields = [
            ("HighBP", "High blood pressure?"),
            ("HighChol", "High cholesterol?"),
            ("CholCheck", "Cholesterol check in last 5 years?"),
            ("Stroke", "History of stroke?"),
            ("HeartDiseaseorAttack", "Heart disease or heart attack?"),
            ("Smoker", "Have you smoked 100+ cigarettes in your life?"),
            ("DiffWalk", "Difficulty walking?"),
            ("AnyHealthcare", "Any healthcare coverage?"),
            ("NoDocbcCost", "Could not see a doctor due to cost in the past 12 months?")
        ]
        cols = st.columns(3)
        for index, (key, label) in enumerate(binary_fields):
            with cols[index % 3]:
                binary_values[key] = st.selectbox(label, ["No", "Yes"], key=f"person_{key}")

        x1, x2, x3 = st.columns(3)
        with x1:
            genhlth = st.selectbox("General health", [1, 2, 3, 4, 5],
                                   format_func=lambda v: {1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}[v])
        with x2:
            menthlth = st.slider("Poor mental health days (0–30)", 0, 30, 0)
        with x3:
            physhlth = st.slider("Poor physical health days (0–30)", 0, 30, 0)

        st.markdown("---")
        st.markdown("### 03 · Everyday habits")
        lifestyle_fields = [
            ("PhysActivity", "Physical activity?"),
            ("Fruits", "Eat fruit at least once per day?"),
            ("Veggies", "Eat vegetables at least once per day?"),
            ("HvyAlcoholConsump", "Heavy alcohol consumption (survey definition)?")
        ]
        cols = st.columns(4)
        for index, (key, label) in enumerate(lifestyle_fields):
            with cols[index]:
                binary_values[key] = st.selectbox(label, ["No", "Yes"], key=f"person_{key}")

        education_options = {
            1: "Never attended school / Kindergarten",
            2: "Grades 1–8",
            3: "Grades 9–11",
            4: "Grade 12 / High school graduate or GED",
            5: "College 1–3 years",
            6: "College 4+ years"
        }
        # INR display labels are approximate conversions of the CDC dataset's
        # original USD income bands (using roughly ₹95 per USD).
        # The selected category code (1–8) remains unchanged for the model.
        income_options = {
            1: "Below ₹9.5 lakh",
            2: "₹9.5 lakh – ₹14.2 lakh",
            3: "₹14.2 lakh – ₹19 lakh",
            4: "₹19 lakh – ₹23.7 lakh",
            5: "₹23.7 lakh – ₹33.2 lakh",
            6: "₹33.2 lakh – ₹47.5 lakh",
            7: "₹47.5 lakh – ₹71.2 lakh",
            8: "₹71.2 lakh or more"
        }

        x1, x2 = st.columns(2)
        with x1:
            selected_education = st.selectbox(
                "Highest education completed",
                options=list(education_options.keys()),
                format_func=lambda value: education_options[value],
                help="Choose the highest education category that applies."
            )
            education = selected_education
        with x2:
            selected_income = st.selectbox(
                "Annual household income (approx. INR)",
                options=list(income_options.keys()),
                format_func=lambda value: income_options[value],
                help="These rupee ranges are approximate conversions of the CDC dataset’s original US-dollar income categories. They are display labels only; the model still receives the original category code."
            )
            income = selected_income

        st.caption("Income ranges are approximate INR equivalents of the original CDC US-dollar bands (about ₹95 per US$1). They are not India-specific income brackets. The original category codes are sent to the model.")


        submitted = st.form_submit_button("Generate my model result  →", use_container_width=True)

    if submitted:
        input_data = {key: int(value == "Yes") for key, value in binary_values.items()}
        input_data.update({
            "BMI": bmi, "GenHlth": genhlth, "MentHlth": menthlth,
            "PhysHlth": physhlth, "Sex": int(sex == "Male"),
            "Age": age, "Education": education, "Income": income
        })
        prediction, probability = make_prediction(input_data)
        save_prediction(CURRENT_USER["id"], prediction, probability, "Health Check-in")
        result_display(prediction, probability)
        st.success("Your result has been saved to My Insights.")


# =========================================================
# 3. DATASET STUDIO
# =========================================================
elif page == "Dataset Atelier":
    page_header("YOUR DATA, YOUR DISCOVERY", "The dataset atelier.", "Upload a compatible health-indicator file, preview its structure and create predictions.")
    st.markdown('<div class="pin-card pin-sage"><div class="pin-icon">📎</div><h3>Drop your dataset here</h3><p>CSV and Excel (.xlsx) files are supported. Your file must contain all 21 model input features.</p></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.success(f"Loaded {uploaded_file.name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Records", f"{len(uploaded_df):,}")
            m2.metric("Columns", len(uploaded_df.columns))
            m3.metric("Missing cells", f"{int(uploaded_df.isna().sum().sum()):,}")

            with st.expander("Preview uploaded records", expanded=True):
                st.dataframe(uploaded_df.head(12), use_container_width=True)

            valid, validated = validate_dataset(uploaded_df)
            if not valid:
                st.error(validated)
                st.caption("Required feature names and category ranges must match the CDC dataset format.")
            else:
                st.markdown("### A quick look at your data")
                show_dataframe_charts(validated)

                if st.button("Analyze dataset & generate predictions  →", use_container_width=True):
                    predictions = model.predict(validated)
                    probabilities = model.predict_proba(validated)[:, 1]

                    results_df = uploaded_df.copy()
                    results_df["Predicted_Class"] = predictions
                    results_df["Positive_Class_Probability"] = probabilities

                    total_records = int(len(results_df))
                    positive_count = int((predictions == 1).sum())
                    negative_count = int((predictions == 0).sum())
                    batch_rate = float(probabilities.mean())

                    st.session_state[f"dianova_batch_results_{CURRENT_USER['id']}"] = results_df
                    st.session_state[f"dianova_batch_source_{CURRENT_USER['id']}"] = uploaded_file.name

                    save_prediction(
                        user_id=CURRENT_USER["id"],
                        prediction=int(positive_count > negative_count),
                        probability=batch_rate,
                        source="Dataset Atelier",
                        file_name=uploaded_file.name,
                        record_count=total_records,
                        positive_count=positive_count,
                        negative_count=negative_count
                    )

                    st.success("Your batch summary has been saved to My Insights.")

                if (st.session_state.get(f"dianova_batch_source_{CURRENT_USER['id']}") == uploaded_file.name
                        and f"dianova_batch_results_{CURRENT_USER['id']}" in st.session_state):
                    results_df = st.session_state[f"dianova_batch_results_{CURRENT_USER['id']}"]
                    st.success("Batch analysis completed.")
                    st.markdown("### Your results")
                    st.dataframe(results_df.head(30), use_container_width=True)
                    positive_count = int((results_df["Predicted_Class"] == 1).sum())
                    negative_count = len(results_df) - positive_count
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Total classified", f"{len(results_df):,}")
                    r2.metric("Positive category", f"{positive_count:,}")
                    r3.metric("Negative category", f"{negative_count:,}")
                    chart_df = pd.DataFrame({"Records":[negative_count, positive_count]}, index=["Negative category", "Positive category"])
                    st.bar_chart(chart_df)
                    st.download_button(
                        "Download predictions as CSV  ↓",
                        results_df.to_csv(index=False).encode("utf-8"),
                        file_name="dianova_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption("These are model classifications, not confirmed clinical diagnoses.")
        except Exception as error:
            st.error(f"Unable to read or analyze this file: {error}")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Required input columns**")
        st.code(", ".join(FEATURES), language="text")
        st.caption("For uploaded files, categorical fields must use the dataset's original numeric codes: Age 1–13, GenHlth 1–5, Sex 0–1, Education 1–6 and Income 1–8. The individual check-in page displays readable labels instead.")



# =========================================================
# 4. MY INSIGHTS — only records belonging to the logged-in user
# =========================================================
elif page == "My Insights":
    page_header(
        "YOUR PERSONAL STORY",
        "My insights.",
        "A private view of prediction activity saved to your account."
    )

    history = get_user_history(CURRENT_USER["id"])

    if history.empty:
        st.markdown(
            '<div class="pin-card pin-sage"><h3>Your story starts here 🌱</h3>'
            '<p>No saved predictions yet. Complete a Health Check-in or analyse a dataset '
            'to see your personal insights here.</p></div>',
            unsafe_allow_html=True
        )
    else:
        batch_history = history[history["source"] == "Dataset Atelier"].copy()
        individual_history = history[history["source"] != "Dataset Atelier"].copy()

        total_activities = int(len(history))
        total_records = int(history["record_count"].fillna(1).sum())
        individual_checkins = int(len(individual_history))

        # Only batch records have meaningful positive/negative totals.
        # Older batch entries are NULL because the previous database did not save them.
        known_batch_counts = batch_history.dropna(
            subset=["positive_count", "negative_count"]
        )
        positive_total = int(known_batch_counts["positive_count"].sum())
        negative_total = int(known_batch_counts["negative_count"].sum())
        batch_records_with_counts = int(known_batch_counts["record_count"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Saved activities", f"{total_activities:,}")
        c2.metric("Records analysed", f"{total_records:,}")
        c3.metric("Individual check-ins", f"{individual_checkins:,}")

        st.markdown("### Batch prediction summary")
        if batch_history.empty:
            st.info("You have not analysed any datasets yet.")
        elif known_batch_counts.empty:
            st.info(
                "Your previous batch analyses are available below, but their positive and "
                "negative totals were not stored by the older version. New batch analyses "
                "will save both counts."
            )
        else:
            b1, b2, b3 = st.columns(3)
            b1.metric("Positive records", f"{positive_total:,}")
            b2.metric("Negative records", f"{negative_total:,}")
            b3.metric("Batch records with saved counts", f"{batch_records_with_counts:,}")

        st.markdown("### Dataset analysis history")
        if not batch_history.empty:
            batch_display = batch_history.rename(columns={
                "file_name": "Uploaded file",
                "record_count": "Total records",
                "positive_count": "Positive records",
                "negative_count": "Negative records",
                "probability": "Average positive-class probability",
                "created_at": "Date and time"
            })[[
                "Uploaded file", "Total records", "Positive records",
                "Negative records", "Average positive-class probability", "Date and time"
            ]].copy()

            batch_display["Positive records"] = batch_display["Positive records"].fillna("Not available (older entry)")
            batch_display["Negative records"] = batch_display["Negative records"].fillna("Not available (older entry)")
            batch_display["Average positive-class probability"] = (
                pd.to_numeric(batch_display["Average positive-class probability"], errors="coerce")
                .mul(100).round(1).astype(str) + "%"
            )
            st.dataframe(batch_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No dataset analysis history yet.")

        st.markdown("### Individual check-in history")
        if not individual_history.empty:
            individual_display = individual_history.rename(columns={
                "prediction": "Model classification",
                "probability": "Positive-class score",
                "created_at": "Date and time"
            })[["Model classification", "Positive-class score", "Date and time"]].copy()
            individual_display["Model classification"] = individual_display["Model classification"].map(
                {0: "Negative category", 1: "Positive category"}
            )
            individual_display["Positive-class score"] = (
                individual_display["Positive-class score"].astype(float).mul(100).round(1).astype(str) + "%"
            )
            st.dataframe(individual_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No individual check-ins saved yet.")

        st.download_button(
            "Download my complete history as CSV ↓",
            history.to_csv(index=False).encode("utf-8"),
            file_name="my_dianova_history.csv",
            mime="text/csv"
        )

        if len(batch_history) > len(known_batch_counts):
            st.caption(
                "Note: Positive/negative counts are unavailable for older batch entries "
                "created before this update. They will be saved for all new batch analyses."
            )

        st.caption("Only records linked to your signed-in account are shown.")

    st.caption("These records are machine-learning outputs, not medical diagnoses.")

# =========================================================
# 5. DIA NOVA COMPANION — Gemini-powered educational assistant
# =========================================================
elif page == "DiaNova Companion":
    page_header(
        "A FRIENDLY PLACE TO ASK",
        "Meet DiaNova Companion.",
        "Ask general questions about diabetes, wellness indicators and healthy habits."
    )

    st.markdown(
        '<div class="note-card"><b>A gentle reminder</b><br>'
        'DiaNova Companion provides general educational information only. '
        'It cannot diagnose, assess your personal medical condition, recommend medication '
        'or replace a qualified healthcare professional. Avoid sharing identifying or '
        'sensitive medical information.</div>',
        unsafe_allow_html=True
    )

    # Read the API key from Streamlit secrets first, then the environment.
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        gemini_api_key = ""

    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    if not gemini_api_key:
        st.warning("DiaNova Companion needs a Gemini API key before it can answer questions.")
        st.markdown(
            "**Setup required:** Add `GEMINI_API_KEY` to your local `.env`/environment "
            "or your deployment platform's Streamlit secrets. Never paste the key into "
            "`app.py` or share it in chat."
        )
        st.code('GEMINI_API_KEY = "your_api_key_here"', language="toml")
    else:
        try:
            from google import genai

            gemini_client = genai.Client(api_key=gemini_api_key)
        except Exception as error:
            gemini_client = None
            st.error(f"Could not initialise Gemini. Check that the google-genai package is installed. Details: {error}")

        if gemini_client is not None:
            chat_history = get_chat_history(CURRENT_USER["id"])

            if st.button("Clear companion chat", key="clear_companion_chat"):
                with get_connection() as connection:
                    connection.execute(
                        "DELETE FROM assistant_messages WHERE user_id = ?",
                        (CURRENT_USER["id"],)
                    )
                st.rerun()

            # Render the signed-in user's saved conversation.
            for chat_item in chat_history:
                if chat_item["role"] in ("user", "assistant"):
                    with st.chat_message(chat_item["role"]):
                        st.markdown(chat_item["message"])

            user_prompt = st.chat_input("Ask DiaNova Companion a question...")

            if user_prompt:
                save_chat_message(CURRENT_USER["id"], "user", user_prompt)

                with st.chat_message("user"):
                    st.markdown(user_prompt)

                # Include a limited recent conversation for context.
                recent_history = get_chat_history(CURRENT_USER["id"])[-12:]
                conversation_text = "\n".join(
                    f"{'User' if item['role'] == 'user' else 'DiaNova Companion'}: {item['message']}"
                    for item in recent_history
                )

                system_guidance = (
                    "You are DiaNova Companion, an educational health-information assistant. "
                    "Give clear, friendly, easy-to-understand general information. "
                    "You may explain diabetes, common health indicators, nutrition, physical "
                    "activity and general wellness. Do not diagnose, calculate an individual's "
                    "medical risk, prescribe or change medication, interpret emergencies as safe, "
                    "or claim that the project model is clinically validated. Explain uncertainty "
                    "and encourage users to consult a qualified healthcare professional for "
                    "personal medical decisions. If the user describes severe or urgent symptoms, "
                    "advise them to seek urgent local medical care. Do not ask for identifying "
                    "information. Keep answers concise and supportive."
                )

                full_prompt = (
                    f"{system_guidance}\n\n"
                    f"Conversation so far:\n{conversation_text}\n\n"
                    "DiaNova Companion:"
                )

                try:
                    with st.chat_message("assistant"):
                        with st.spinner("DiaNova Companion is thinking..."):
                            response = None
                            last_error = None

                            # Retry only temporary service/rate-limit errors.
                            # Invalid API keys and unsupported model errors are not retried.
                            for attempt in range(3):
                                try:
                                    response = gemini_client.models.generate_content(
                                        model="gemini-3.1-flash-lite",
                                        contents=full_prompt
                                    )
                                    break
                                except Exception as error:
                                    last_error = error
                                    error_text = str(error).lower()
                                    status_code = getattr(error, "code", None)
                                    status_code = status_code or getattr(error, "status_code", None)

                                    temporary_error = (
                                        status_code in (429, 500, 502, 503, 504)
                                        or any(
                                            marker in error_text
                                            for marker in (
                                                "429", "500", "502", "503", "504",
                                                "resource_exhausted", "unavailable",
                                                "internal server error", "deadline exceeded"
                                            )
                                        )
                                    )

                                    if not temporary_error or attempt == 2:
                                        raise

                                    # Short exponential backoff: 1 second, then 2 seconds.
                                    time.sleep(2 ** attempt)

                            if response is None:
                                raise last_error or RuntimeError(
                                    "Gemini did not return a response."
                                )

                            answer = (response.text or "").strip()

                            if not answer:
                                answer = "I couldn't generate a response just now. Please try asking in a different way."

                        st.markdown(answer)

                    save_chat_message(CURRENT_USER["id"], "assistant", answer)

                except Exception as error:
                    error_text = str(error).lower()

                    if any(code in error_text for code in ("503", "429", "500", "502", "504", "unavailable", "resource_exhausted")):
                        st.warning(
                            "Gemini is temporarily busy or rate-limited. I tried again automatically, "
                            "but it is still unavailable. Please wait a little and send your question again."
                        )
                    else:
                        st.error(
                            "I couldn't get a response from Gemini. Please check your API key, "
                            "model access and internet connection."
                        )

                    st.caption(f"Technical detail: {error}")

# =========================================================
# 6. PROFILE
# =========================================================
elif page == "Profile":
    page_header(
        "YOUR ACCOUNT",
        "A space that's yours.",
        "Manage your personal details, wellness preferences and account activity."
    )

    # Load only the signed-in user's saved profile information.
    with get_connection() as connection:
        profile = connection.execute(
            """
            SELECT full_name, email, age, dietary_preference,
                   activity_level, wellness_goal
            FROM users WHERE id = ?
            """,
            (CURRENT_USER["id"],)
        ).fetchone()

    history = get_user_history(CURRENT_USER["id"])
    total_predictions = int(len(history))
    total_checkins = int((history["source"] == "Health Check-in").sum()) if not history.empty else 0
    total_dataset_runs = int((history["source"] == "Dataset Atelier").sum()) if not history.empty else 0

    st.markdown(
        f'<div class="pin-card pin-sage"><div class="section-kicker">SIGNED-IN ACCOUNT</div>'
        f'<h3>👤 {profile["full_name"]}</h3><p>{profile["email"]}</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📊 My Activity")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Total predictions", f"{total_predictions:,}")
    stat2.metric("Health check-ins", f"{total_checkins:,}")
    stat3.metric("Dataset analyses", f"{total_dataset_runs:,}")

    st.markdown("---")
    st.markdown("### 👤 Personal Information")
    with st.form("profile_personal_form"):
        edited_name = st.text_input("Full name", value=profile["full_name"] or "")
        st.text_input("Email address", value=profile["email"] or "", disabled=True)
        edited_age = st.number_input(
            "Age (optional)", min_value=0, max_value=120,
            value=int(profile["age"] or 0), help="Enter 0 if you prefer not to provide your age."
        )
        personal_saved = st.form_submit_button("Save personal information")
    if personal_saved:
        if not edited_name.strip():
            st.error("Please enter your name.")
        else:
            with get_connection() as connection:
                connection.execute(
                    "UPDATE users SET full_name = ?, age = ? WHERE id = ?",
                    (edited_name.strip(), edited_age or None, CURRENT_USER["id"])
                )
            st.session_state["dianova_user"]["name"] = edited_name.strip()
            st.success("Your personal information has been saved.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 🥗 Health Preferences")
    dietary_options = ["Not set", "Vegetarian", "Non-vegetarian", "Vegan", "Eggetarian", "Other"]
    activity_options = ["Not set", "Low", "Moderate", "High"]
    goal_options = [
        "Not set", "Improve eating habits", "Increase physical activity",
        "Reduce added sugar", "Maintain a healthy routine"
    ]

    def profile_choice(options, current_value):
        return options.index(current_value) if current_value in options else 0

    with st.form("profile_preferences_form"):
        dietary = st.selectbox(
            "Dietary preference", dietary_options,
            index=profile_choice(dietary_options, profile["dietary_preference"])
        )
        activity = st.selectbox(
            "Activity level", activity_options,
            index=profile_choice(activity_options, profile["activity_level"])
        )
        goal = st.selectbox(
            "Wellness goal", goal_options,
            index=profile_choice(goal_options, profile["wellness_goal"])
        )
        preferences_saved = st.form_submit_button("Save health preferences")
    if preferences_saved:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET dietary_preference = ?, activity_level = ?, wellness_goal = ?
                WHERE id = ?
                """,
                (
                    None if dietary == "Not set" else dietary,
                    None if activity == "Not set" else activity,
                    None if goal == "Not set" else goal,
                    CURRENT_USER["id"]
                )
            )
        st.success("Your health preferences have been saved.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕘 Recent Prediction Activity")
    if history.empty:
        st.info("No prediction activity yet. Complete a Health Check-in or analyse a dataset to see activity here.")
    else:
        recent = history.head(5).copy()
        recent["source"] = recent["source"].replace({
            "Health Check-in": "Health check-in",
            "Dataset Atelier": "Dataset analysis"
        })
        recent = recent.rename(columns={
            "source": "Activity",
            "created_at": "Date and time",
            "record_count": "Records",
            "prediction": "Model category"
        })
        recent["Model category"] = recent["Model category"].map({
            0: "Negative category", 1: "Positive category"
        })
        st.dataframe(
            recent[["Activity", "Date and time", "Records", "Model category"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### 🔐 Change Password")
    with st.form("profile_password_form"):
        current_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_new_password = st.text_input("Confirm new password", type="password")
        password_saved = st.form_submit_button("Update password")
    if password_saved:
        with get_connection() as connection:
            account = connection.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (CURRENT_USER["id"],)
            ).fetchone()
            if not account or not verify_password(current_password, account["password_hash"]):
                st.error("Your current password is incorrect.")
            elif len(new_password) < 8:
                st.error("Your new password must contain at least 8 characters.")
            elif new_password != confirm_new_password:
                st.error("The new passwords do not match.")
            else:
                connection.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (password_digest(new_password), CURRENT_USER["id"])
                )
                st.success("Your password has been updated.")

    st.caption(
        "Health preferences are optional and are used only as profile information. "
        "DiaNova AI is an educational tool, not a medical service. "
        "Your account was created for this DiaNova AI installation."
    )


st.markdown(
    '<div class="dianova-footer">🌿 DiaNova AI · Smarter Insights. Healthier Futures.<br>Made with care for curious minds.</div>',
    unsafe_allow_html=True
)
