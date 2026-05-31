from pathlib import Path
import pickle
import warnings

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bean_classifier_model.pkl"
SCALER_PATH = BASE_DIR / "bean_scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


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


def build_input_form(feature_names, default_values):
    values = {}
    columns = st.columns(2)

    for index, feature in enumerate(feature_names):
        default = float(default_values[index])
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                value=default,
                format="%.6f",
                step=0.01,
            )

    return pd.DataFrame([values], columns=feature_names)


def main():
    st.set_page_config(page_title="Dry Bean Classifier")

    st.title("Dry Bean Classifier")
    st.write(
        "Enter the 16 bean measurements in the same feature scale used during "
        "training, then predict the bean variety."
    )

    try:
        model, scaler, label_encoder = load_artifacts()
    except FileNotFoundError as error:
        st.error(f"Missing model artifact: {error.filename}")
        st.stop()

    feature_names = list(scaler.feature_names_in_)
    default_values = getattr(scaler, "mean_", [0.0] * len(feature_names))

    with st.form("bean_features"):
        input_df = build_input_form(feature_names, default_values)
        submitted = st.form_submit_button("Predict bean class")

    if submitted:
        scaled_input = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_input, columns=feature_names)
        encoded_prediction = model.predict(scaled_df)
        bean_class = label_encoder.inverse_transform(encoded_prediction)[0]

        st.success(f"Predicted bean class: {bean_class}")

        if hasattr(model, "decision_function"):
            decision_scores = model.decision_function(scaled_df)
            score_df = pd.DataFrame(decision_scores, columns=label_encoder.classes_)
            st.caption("Decision scores from the SVC model")
            st.dataframe(score_df, use_container_width=True, hide_index=True)

    with st.expander("Model details"):
        st.write(f"Model: `{type(model).__name__}`")
        st.write("Classes:")
        st.write(", ".join(label_encoder.classes_))
        st.write("Expected feature order:")
        st.code("\n".join(feature_names))


if __name__ == "__main__":
    main()
