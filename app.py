from pathlib import Path
import math
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bean_classifier_model.pkl"
SCALER_PATH = BASE_DIR / "bean_scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
VIEWPORT_WIDTH = 940
VIEWPORT_HEIGHT = 520


CLASS_PROFILES = {
    "BARBUNYA": {"color": "#ff7a4d", "code": "BRB", "risk": "ELEVATED"},
    "BOMBAY": {"color": "#f6d365", "code": "BMB", "risk": "LOW"},
    "CALI": {"color": "#6ee7ff", "code": "CAL", "risk": "MEDIUM"},
    "DERMASON": {"color": "#9dffb3", "code": "DRM", "risk": "LOW"},
    "HOROZ": {"color": "#b69cff", "code": "HRZ", "risk": "HIGH"},
    "SEKER": {"color": "#ff9fcb", "code": "SKR", "risk": "MEDIUM"},
    "SIRA": {"color": "#73fbd3", "code": "SIR", "risk": "LOW"},
}


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');

        :root {
            --cyan: #5ee7ff;
            --cyan-soft: rgba(94, 231, 255, 0.18);
            --panel: rgba(8, 20, 34, 0.82);
            --panel-border: rgba(129, 223, 255, 0.22);
            --muted: #8ea8ba;
            --text: #eaf8ff;
            --lime: #94ffb8;
            --danger: #ff6d6d;
        }

        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 14% 12%, rgba(76, 201, 255, 0.18), transparent 28%),
                radial-gradient(circle at 82% 2%, rgba(133, 255, 190, 0.12), transparent 24%),
                linear-gradient(135deg, #03070d 0%, #071421 48%, #02050a 100%);
        }

        .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(94, 231, 255, 0.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(94, 231, 255, 0.045) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.9), rgba(0,0,0,0.08));
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        header, footer, #MainMenu {
            visibility: hidden;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 10px 14px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: linear-gradient(90deg, rgba(7, 17, 30, 0.9), rgba(10, 36, 52, 0.72));
            box-shadow: 0 0 32px rgba(94, 231, 255, 0.08);
            margin-bottom: 14px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .brand-mark {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 18px var(--cyan), 0 0 42px rgba(94, 231, 255, 0.7);
        }

        .ticker {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .ticker span {
            color: var(--lime);
            font-weight: 800;
        }

        .panel {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 16px;
            background:
                linear-gradient(180deg, rgba(11, 31, 48, 0.86), rgba(3, 11, 20, 0.9)),
                repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 5px);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 18px 44px rgba(0,0,0,0.28);
            min-height: 100%;
        }

        .panel-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--cyan);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 12px;
        }

        .thin-rule {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(94,231,255,0.62), transparent);
            margin: 10px 0 14px;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0;
        }

        .mini-card {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(129, 223, 255, 0.15);
            background: rgba(3, 13, 24, 0.72);
        }

        .mini-label {
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
        }

        .mini-value {
            color: var(--text);
            font-size: 20px;
            font-weight: 800;
            margin-top: 3px;
        }

        .target-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border-radius: 999px;
            color: var(--cyan);
            border: 1px solid rgba(94, 231, 255, 0.28);
            background: rgba(94, 231, 255, 0.08);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .bottom-nav {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }

        .nav-cell {
            color: #7fb9cc;
            border: 1px solid rgba(129, 223, 255, 0.18);
            background: rgba(5, 18, 30, 0.8);
            border-radius: 6px;
            padding: 9px 8px;
            font-size: 10px;
            text-align: center;
            text-transform: uppercase;
        }

        .result-card {
            border: 1px solid rgba(94, 231, 255, 0.3);
            border-radius: 8px;
            padding: 18px;
            background: radial-gradient(circle at 20% 0%, rgba(94,231,255,0.18), rgba(6,16,28,0.9) 54%);
        }

        .result-title {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
        }

        .result-class {
            margin-top: 4px;
            font-size: 38px;
            line-height: 1;
            font-weight: 800;
        }

        .stNumberInput label, .stSlider label, .stSelectbox label {
            color: #b9d7e4 !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
        }

        div[data-testid="stMetric"] {
            background: rgba(3, 13, 24, 0.72);
            border: 1px solid rgba(129, 223, 255, 0.15);
            border-radius: 8px;
            padding: 12px;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
        }

        .stButton > button, .stFormSubmitButton > button {
            width: 100%;
            border-radius: 6px;
            border: 1px solid rgba(94, 231, 255, 0.45);
            color: #03101b;
            background: linear-gradient(90deg, #66e7ff, #8dffbf);
            font-weight: 800;
            text-transform: uppercase;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid rgba(129, 223, 255, 0.18);
            border-radius: 6px;
            background: rgba(5, 18, 30, 0.8);
            color: #9fc6d8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """Load the trained model and preprocessing objects once per session."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
        with SCALER_PATH.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with ENCODER_PATH.open("rb") as encoder_file:
            label_encoder = pickle.load(encoder_file)

    return model, scaler, label_encoder


@st.cache_data
def make_orbital_view(target_x=None, target_y=None):
    width, height = VIEWPORT_WIDTH, VIEWPORT_HEIGHT
    rng = random.Random(1610)
    image = Image.new("RGB", (width, height), "#020814")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(180):
        x = rng.randrange(width)
        y = rng.randrange(height)
        alpha = rng.randrange(60, 190)
        color = (180, 235, 255, alpha)
        draw.point((x, y), fill=color)

    sun_center = (42, 52)
    for radius, alpha in [(96, 18), (68, 30), (38, 72), (18, 220)]:
        draw.ellipse(
            [
                sun_center[0] - radius,
                sun_center[1] - radius,
                sun_center[0] + radius,
                sun_center[1] + radius,
            ],
            fill=(105, 221, 255, alpha),
        )

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    earth = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    edraw = ImageDraw.Draw(earth)
    earth_box = (-330, -130, 530, 730)
    for i in range(74):
        inset = i * 2
        tone = max(0, 120 - i)
        edraw.ellipse(
            [
                earth_box[0] + inset,
                earth_box[1] + inset,
                earth_box[2] - inset,
                earth_box[3] - inset,
            ],
            outline=(38, 170, 255, max(0, 88 - i)),
            width=2,
        )
        if i < 38:
            edraw.ellipse(
                [
                    earth_box[0] + i * 5,
                    earth_box[1] + i * 5,
                    earth_box[2] - i * 5,
                    earth_box[3] - i * 5,
                ],
                outline=(20, 70 + tone, 145 + tone, 18),
                width=4,
            )

    land_shapes = [
        [(160, 112), (228, 92), (292, 142), (270, 208), (205, 232), (146, 180)],
        [(252, 250), (332, 260), (374, 318), (320, 390), (238, 354)],
        [(84, 242), (132, 206), (174, 252), (150, 322), (74, 302)],
    ]
    for shape in land_shapes:
        edraw.polygon(shape, fill=(28, 84, 74, 84))

    for _ in range(80):
        x = rng.randrange(100, 370)
        y = rng.randrange(85, 392)
        if ((x - 100) ** 2 + (y - 300) ** 2) < 250000:
            edraw.ellipse((x, y, x + 2, y + 2), fill=(255, 210, 126, rng.randrange(85, 210)))

    earth = earth.filter(ImageFilter.GaussianBlur(0.35))
    image = Image.alpha_composite(image.convert("RGBA"), earth)
    draw = ImageDraw.Draw(image)

    for x in range(0, width, 78):
        draw.line((x, 0, x, height), fill=(65, 170, 220, 18), width=1)
    for y in range(0, height, 58):
        draw.line((0, y, width, y), fill=(65, 170, 220, 18), width=1)

    draw.line((468, 34, 468, height - 44), fill=(168, 232, 255, 150), width=2)
    draw.line((470, 34, 470, height - 44), fill=(48, 105, 130, 70), width=1)

    for i, (x, y) in enumerate([(610, 126), (682, 114), (742, 148), (812, 174), (706, 220), (782, 260)]):
        draw.ellipse((x - 2, y - 2, x + 3, y + 3), fill=(255, 211, 119, 220))
        if i > 0:
            px, py = [(610, 126), (682, 114), (742, 148), (812, 174), (706, 220), (782, 260)][i - 1]
            draw.line((px, py, x, y), fill=(255, 211, 119, 56), width=1)

    for y in range(34, height - 34, 16):
        draw.line((18, y, 32, y), fill=(94, 231, 255, 64), width=1)
        draw.line((width - 32, y, width - 18, y), fill=(94, 231, 255, 64), width=1)

    if target_x is not None and target_y is not None:
        x = int(target_x)
        y = int(target_y)
        for radius, alpha in [(58, 34), (38, 58), (20, 112)]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(94, 231, 255, alpha), width=2)
        draw.line((x - 46, y, x - 12, y), fill=(148, 255, 202, 220), width=2)
        draw.line((x + 12, y, x + 46, y), fill=(148, 255, 202, 220), width=2)
        draw.line((x, y - 46, x, y - 12), fill=(148, 255, 202, 220), width=2)
        draw.line((x, y + 12, x, y + 46), fill=(148, 255, 202, 220), width=2)
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline=(255, 255, 255, 230), width=1)

    draw.rectangle((0, 0, width - 1, height - 1), outline=(94, 231, 255, 88), width=1)
    return image.convert("RGB")


def initialize_state():
    if "target" not in st.session_state:
        st.session_state.target = {"x": 468, "y": 260}
    if "target_history" not in st.session_state:
        st.session_state.target_history = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None


def target_stats(target):
    x = target["x"]
    y = target["y"]
    nx = x / VIEWPORT_WIDTH
    ny = y / VIEWPORT_HEIGHT
    lon = round((nx - 0.5) * 360, 3)
    lat = round((0.5 - ny) * 180, 3)
    distance = math.sqrt((x - VIEWPORT_WIDTH / 2) ** 2 + (y - VIEWPORT_HEIGHT / 2) ** 2)
    lock = max(22, min(99, int(100 - distance / 6.8)))
    sector = f"{chr(65 + min(7, int(nx * 8)))}-{1 + min(5, int(ny * 6))}"
    return {
        "lat": lat,
        "lon": lon,
        "lock": lock,
        "sector": sector,
        "nx": nx,
        "ny": ny,
    }


def render_topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand"><span class="brand-mark"></span> ORBITAL DRY BEAN INTELLIGENCE</div>
            <div class="ticker">
                OPTIC LATEST NEWS&nbsp;&nbsp; NTDA PERFORMANCE <span>+4.35%</span>&nbsp;&nbsp; SIGNAL UPLINK: STABLE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_target_panel(target, stats):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title"><span>Target Acquisition</span><span class="target-chip">Live cursor lock</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Click anywhere inside the orbital viewport. The interface stores the clicked point,
        converts it into normalized targeting coordinates, and keeps the latest acquisitions.
        """
    )
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    st.metric("Cursor X", f"{target['x']:03d}px")
    st.metric("Cursor Y", f"{target['y']:03d}px")
    st.metric("Estimated latitude", f"{stats['lat']:+.3f}")
    st.metric("Estimated longitude", f"{stats['lon']:+.3f}")

    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">Sector</div>
            <div class="mini-value">{stats["sector"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(stats["lock"] / 100, text=f"Target lock confidence: {stats['lock']}%")

    if st.button("Clear target history"):
        st.session_state.target_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_viewport(target):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title"><span>Geospatial Classifier Viewport</span><span>OPTIC FEED 03</span></div>',
        unsafe_allow_html=True,
    )
    image = make_orbital_view(target["x"], target["y"])
    coords = streamlit_image_coordinates(image, key="orbital_target")

    if coords is not None and {"x", "y"}.issubset(coords):
        new_target = {"x": int(coords["x"]), "y": int(coords["y"])}
        if new_target != st.session_state.target:
            st.session_state.target = new_target
            stats = target_stats(new_target)
            st.session_state.target_history.insert(
                0,
                {
                    "x": new_target["x"],
                    "y": new_target["y"],
                    "sector": stats["sector"],
                    "lat": stats["lat"],
                    "lon": stats["lon"],
                    "lock": stats["lock"],
                },
            )
            st.session_state.target_history = st.session_state.target_history[:8]
            st.rerun()

    st.markdown(
        """
        <div class="bottom-nav">
            <div class="nav-cell">Scan</div>
            <div class="nav-cell">Target</div>
            <div class="nav-cell">Telemetry</div>
            <div class="nav-cell">Classifier</div>
            <div class="nav-cell">Archive</div>
            <div class="nav-cell">Signal</div>
            <div class="nav-cell">Launch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def build_input_form(feature_names, default_values, scale_values):
    tabs = st.tabs(["Core Geometry", "Shape Ratios", "Signal Calibration"])
    groups = [
        feature_names[:6],
        feature_names[6:12],
        feature_names[12:],
    ]
    values = {}

    for tab, group in zip(tabs, groups):
        with tab:
            columns = st.columns(2)
            for index, feature in enumerate(group):
                feature_index = feature_names.index(feature)
                default = float(default_values[feature_index])
                scale = float(scale_values[feature_index])
                step = max(scale / 30, 0.0001)
                with columns[index % 2]:
                    values[feature] = st.number_input(
                        feature,
                        value=default,
                        format="%.6f",
                        step=step,
                    )

    return pd.DataFrame([values], columns=feature_names)


def render_prediction_card(bean_class, decision_scores, label_encoder):
    profile = CLASS_PROFILES.get(bean_class, {"color": "#73fbd3", "code": "UNK", "risk": "UNKNOWN"})
    score_array = np.asarray(decision_scores).reshape(1, -1)
    score_df = pd.DataFrame(score_array, columns=label_encoder.classes_)
    top_score = float(score_df.iloc[0].max())

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Classifier Output</div>
            <div class="result-class" style="color: {profile["color"]};">{bean_class}</div>
            <div class="thin-rule"></div>
            <div class="metric-row">
                <div class="mini-card"><div class="mini-label">Class code</div><div class="mini-value">{profile["code"]}</div></div>
                <div class="mini-card"><div class="mini-label">Risk band</div><div class="mini-value">{profile["risk"]}</div></div>
                <div class="mini-card"><div class="mini-label">Model</div><div class="mini-value">SVC</div></div>
                <div class="mini-card"><div class="mini-label">Top score</div><div class="mini-value">{top_score:.2f}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(score_df, use_container_width=True, hide_index=True)


def render_history():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><span>Acquisition Log</span><span>Latest 8</span></div>', unsafe_allow_html=True)
    if st.session_state.target_history:
        st.dataframe(pd.DataFrame(st.session_state.target_history), use_container_width=True, hide_index=True)
    else:
        st.info("No target points captured yet. Click the viewport to begin acquisition.")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Orbital Bean Classifier", layout="wide")
    apply_theme()
    initialize_state()
    render_topbar()

    try:
        model, scaler, label_encoder = load_artifacts()
    except FileNotFoundError as error:
        st.error(f"Missing model artifact: {error.filename}")
        st.stop()

    target = st.session_state.target
    stats = target_stats(target)
    left_col, center_col, right_col = st.columns([0.95, 2.3, 1.25], gap="large")

    with left_col:
        render_target_panel(target, stats)

    with center_col:
        render_viewport(target)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span>Classifier Controls</span><span>SVC C=100</span></div>', unsafe_allow_html=True)
        feature_names = list(scaler.feature_names_in_)
        default_values = getattr(scaler, "mean_", [0.0] * len(feature_names))
        scale_values = getattr(scaler, "scale_", [1.0] * len(feature_names))

        with st.form("bean_features"):
            input_df = build_input_form(feature_names, default_values, scale_values)
            submitted = st.form_submit_button("Run classification sequence")

        if submitted:
            scaled_input = scaler.transform(input_df)
            scaled_df = pd.DataFrame(scaled_input, columns=feature_names)
            encoded_prediction = model.predict(scaled_df)
            bean_class = label_encoder.inverse_transform(encoded_prediction)[0]
            decision_scores = model.decision_function(scaled_df)
            st.session_state.last_prediction = {
                "bean_class": bean_class,
                "decision_scores": decision_scores,
            }

        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([1.15, 1], gap="large")

    with lower_left:
        if st.session_state.last_prediction is None:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title"><span>Classifier Output</span><span>Awaiting run</span></div>', unsafe_allow_html=True)
            st.info("Run the classification sequence to display the predicted dry bean class and decision scores.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            prediction = st.session_state.last_prediction
            render_prediction_card(
                prediction["bean_class"],
                prediction["decision_scores"],
                label_encoder,
            )

    with lower_right:
        render_history()

    with st.expander("System manifest"):
        manifest_df = pd.DataFrame(
            {
                "Artifact": ["Classifier", "Scaler", "Label encoder"],
                "File": [MODEL_PATH.name, SCALER_PATH.name, ENCODER_PATH.name],
                "Status": ["Loaded", "Loaded", "Loaded"],
            }
        )
        st.dataframe(manifest_df, use_container_width=True, hide_index=True)
        st.code("\n".join(feature_names), language="text")


if __name__ == "__main__":
    main()
