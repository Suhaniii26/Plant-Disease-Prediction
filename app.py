import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ---------- PAGE SETTINGS ----------

st.set_page_config(
    page_title="LeafLens | Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ---------- STYLING ----------

st.markdown("""
<style>
    .stApp {
        background: #0d1713;
        color: #e7eee9;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #e7f5eb;
    }

    [data-testid="stSidebar"] {
        background: #122019;
    }

    [data-testid="stMetric"] {
        background: #16271e;
        border: 1px solid #2b4635;
        border-radius: 16px;
        padding: 20px;
    }

    [data-testid="stMetricValue"] {
        color: #9cdeac;
    }

    [data-testid="stFileUploader"] {
        background: #16271e;
        border: 1px solid #2b4635;
        border-radius: 16px;
        padding: 16px;
    }

    div.stButton > button[kind="primary"] {
        background: #72bd88;
        color: #0d1713;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #95d6a7;
        color: #0d1713;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }
    /* Larger tab labels and spacing */
.stTabs [data-baseweb="tab"] {
    padding: 16px 24px;
    min-height: 60px;
}

.stTabs [data-baseweb="tab"] p {
    font-size: 20px;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL AND CLASS LABELS ----------

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets():
    model_path = BASE_DIR / "plant_disease_model.keras"
    labels_path = BASE_DIR / "class_names.json"

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError(
            "class_names.json must contain an ordered list of class names."
        )

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(
            "Expected a model with input shape (224, 224, 3)."
        )

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            "The model output count does not match class_names.json."
        )

    return model, labels


try:
    with st.spinner("Loading the trained model..."):
        model, class_names = load_assets()

except Exception as error:
    st.error("Unable to load the model or class labels.")

    st.info(
        "Keep plant_disease_model.keras and class_names.json "
        "in the same folder as app.py."
    )

    with st.expander("Error details"):
        st.code(str(error))

    st.stop()


def readable_label(name):
    return name.replace("___", " · ").replace("_", " ")


# ---------- SIDEBAR ----------

with st.sidebar:
    st.title("🌿 LeafLens")
    st.caption("Plant Disease Intelligence")

    st.divider()

    st.markdown("### Model overview")
    st.write("**Classifier:** Convolutional Neural Network")
    st.write(f"**Supported classes:** {len(class_names)}")
    st.write("**Image input:** 224 × 224 · RGB")

    with st.expander("Explore supported classes"):
        for name in class_names:
            st.write("• " + readable_label(name))

    st.divider()

    st.markdown("### Image guidelines")
    st.markdown(
        "- Photograph a single leaf.\n"
        "- Use good lighting.\n"
        "- Keep the image in focus.\n"
        "- Choose a supported plant class."
    )

    st.divider()

    st.caption(
        "B.Tech · Artificial Intelligence "
        "& Machine Learning"
    )


# ---------- MAIN HEADER ----------

st.title("🌿 Plant Disease Analysis")
st.caption("CNN-based leaf classification • AIML Project")


st.divider()


# ---------- MAIN TABS ----------

analysis_tab, model_tab = st.tabs([
    "🔍 Leaf Analysis",
    "🧠 Model Details"
])


# ---------- LEAF ANALYSIS TAB ----------

with analysis_tab:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Upload & preview")

        uploaded_file = st.file_uploader(
            "Choose a clear leaf image",
            type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file is not None:
            try:
                image = ImageOps.exif_transpose(
                    Image.open(uploaded_file)
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded leaf",
                    width=350
                )

                st.caption(
                    f"Original size: "
                    f"{image.width} × {image.height} pixels"
                )

            except Exception:
                st.error(
                    "This image could not be opened. "
                    "Please upload a valid JPG or PNG."
                )

        else:
            st.info("Upload a leaf photograph to begin.")

        predict_clicked = st.button(
            "🔍 Analyze leaf",
            type="primary",
            disabled=image is None,
            use_container_width=True
        )

    with right:
        st.subheader("Analysis results")

        if predict_clicked and image is not None:
            try:
                with st.spinner("Analyzing the leaf..."):
                    # Match the training input:
                    # RGB, 224 × 224, pixel values from 0 to 255.
                    # The CNN already contains a rescaling layer.
                    image_array = np.array(
                        image,
                        dtype=np.float32
                    )

                    resized_image = tf.image.resize(
                        image_array,
                        (224, 224),
                        method="bilinear"
                    )

                    # Add the batch dimension:
                    # (224, 224, 3) -> (1, 224, 224, 3)
                    input_image = tf.expand_dims(
                        resized_image,
                        axis=0
                    )

                    predictions = np.asarray(
                        model.predict(input_image, verbose=0)
                    )[0]

                    if (
                        predictions.shape != (len(class_names),)
                        or not np.isfinite(predictions).all()
                    ):
                        raise ValueError(
                            "The model returned invalid prediction scores."
                        )

                    predicted_index = int(
                        np.argmax(predictions)
                    )

                    disease = class_names[predicted_index]

                    confidence = float(
                        predictions[predicted_index]
                    )

                st.markdown("#### Most likely class")
                st.info(readable_label(disease))

                st.metric(
                    "Model confidence",
                    f"{confidence:.2%}"
                )

                st.progress(
                    float(np.clip(confidence, 0, 1))
                )

                st.caption(
                    "This score reflects the model's output, "
                    "not a verified diagnosis."
                )

                st.markdown("#### Top 5 predictions")

                top_indices = np.argsort(
                    predictions
                )[::-1][:5]

                for index in top_indices:
                    label = readable_label(
                        class_names[index]
                    )

                    score = float(predictions[index])

                    st.write(
                        f"**{label}** — {score:.1%}"
                    )

                    st.progress(
                        float(np.clip(score, 0, 1))
                    )

            except Exception as error:
                st.error(
                    "The prediction could not be completed."
                )

                with st.expander("Error details"):
                    st.code(str(error))

        else:
            st.info(
                "Your prediction and class scores "
                "will appear here."
            )

            st.markdown("#### For a useful analysis")

            st.markdown(
                "- Use a clear photograph of a single leaf.\n"
                "- Keep the leaf well-lit and in focus.\n"
                "- Use a plant represented in the training classes."
            )

    st.divider()

    st.caption(
        "The model may assign a known class to unrelated images. "
        "Confirm findings before making treatment decisions."
    )

    st.divider()

    st.markdown("### Model performance overview")

    card1, card2, card3 = st.columns(3)

    card1.metric("Test accuracy", "96.03%")
    card2.metric("Weighted F1-score", "0.96")
    card3.metric("Test images evaluated", "5,369")

    st.caption(
        "Reported evaluation results • "
        "Full breakdown in Model Details"
    )


# ---------- MODEL DETAILS TAB ----------

with model_tab:
    st.subheader("Model performance")

    st.caption(
        "Reported test results from the evaluation "
        "screenshot supplied by the project team."
    )

    accuracy_col, loss_col, samples_col = st.columns(3)

    accuracy_col.metric("Test accuracy", "96.03%")
    loss_col.metric("Test loss", "0.1225")
    samples_col.metric("Test images", "5,369")

    precision_col, recall_col, f1_col = st.columns(3)

    precision_col.metric("Weighted precision", "0.96")
    recall_col.metric("Weighted recall", "0.96")
    f1_col.metric("Weighted F1-score", "0.96")

    st.caption(
        "These are reported evaluation values, not metrics "
        "recalculated by this dashboard. Precision, recall "
        "and F1-score are rounded to two decimal places."
    )

    with st.expander("Understanding the evaluation metrics"):
        st.markdown(
            "- **Test accuracy:** Fraction of test images "
            "classified correctly.\n"
            "- **Precision:** How often predictions for "
            "a class are correct.\n"
            "- **Recall:** How many actual examples of "
            "a class are identified.\n"
            "- **F1-score:** Balance between precision "
            "and recall.\n"
            "- **Weighted average:** Class scores averaged "
            "according to the number of test images per class.\n"
            "- **Test loss:** Error measured using the "
            "model's loss function."
        )

        st.info(
            "Test accuracy measures performance across a labeled "
            "dataset. Prediction confidence is a score for "
            "one uploaded image."
        )

    st.divider()

    st.subheader("Inside the CNN")

    st.write(
        "The architecture below is read directly "
        "from the loaded model."
    )

    param_col, layers_col, output_col = st.columns(3)

    param_col.metric(
        "Total parameters",
        f"{model.count_params():,}"
    )

    layers_col.metric(
        "Model layers",
        len(model.layers)
    )

    output_col.metric(
        "Output classes",
        model.output_shape[-1]
    )

    st.markdown("### Layer architecture")

    layer_details = []

    for layer in model.layers:
        layer_details.append({
            "Layer": layer.name,
            "Type": layer.__class__.__name__,
            "Parameters": layer.count_params()
        })

    st.dataframe(
        layer_details,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View full model summary"):
        summary_lines = []

        model.summary(
            print_fn=lambda line, **kwargs: summary_lines.append(line)
        )

        st.code(
            "\n".join(summary_lines),
            language="text"
        )

    st.markdown("### How an image becomes a prediction")

    st.markdown(
        "1. **Prepare:** Convert the image to RGB and "
        "resize it to 224 × 224 pixels.\n"
        "2. **Normalize:** The model's rescaling layer "
        "scales pixel values.\n"
        "3. **Extract features:** Convolution and pooling "
        "layers process visual patterns in the leaf.\n"
        "4. **Classify:** The output layer assigns a score "
        "to each supported class."
    )


