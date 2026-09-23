from hashlib import sha256

from PIL import Image
import streamlit as st

from ai.inference import (
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    extract_detections,
    load_model,
    run_inference,
)


ORIGINAL_IMAGE_WIDTH = 320
MODEL_IMAGE_WIDTH = 500


def prepare_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def analyze_image(
    image,
    model,
    confidence_threshold=0.25,
):
    results = run_inference(
        model=model,
        image=image,
        confidence_threshold=confidence_threshold,
    )

    result = results[0]

    annotated_image = result.plot()[..., ::-1].copy()

    detections = extract_detections(result)

    return annotated_image, detections


def get_image_id(uploaded_file):
    return sha256(
        uploaded_file.getvalue()
    ).hexdigest()


@st.cache_resource
def get_v1_model():
    return load_model(V1_MODEL_PATH)


@st.cache_resource
def get_v2_model():
    return load_model(V2_MODEL_PATH)


def make_detection_rows(detections):
    rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        rows.append(
            {
                "#": index,
                "Class": detection["class"],
                "Confidence": round(
                    detection["confidence"],
                    3,
                ),
            }
        )

    return rows


def get_best_detection(detections):
    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: detection["confidence"],
    )


def show_model_panel(
    title,
    subtitle,
    annotated_image,
    detections,
):
    st.markdown(
        f"### {title}"
    )

    st.caption(
        subtitle
    )

    best_detection = get_best_detection(
        detections
    )

    if best_detection:
        metric_column_1, metric_column_2 = st.columns(2)

        with metric_column_1:
            st.metric(
                "Top class",
                best_detection["class"],
            )

        with metric_column_2:
            st.metric(
                "Top confidence",
                f"{best_detection['confidence']:.2f}",
            )
    else:
        metric_column_1, metric_column_2 = st.columns(2)

        with metric_column_1:
            st.metric(
                "Top class",
                "None",
            )

        with metric_column_2:
            st.metric(
                "Top confidence",
                "0.00",
            )

    st.image(
        annotated_image,
        caption=f"{title} detection",
        width=MODEL_IMAGE_WIDTH,
    )

    st.metric(
        "Total detections",
        len(detections),
    )

    rows = make_detection_rows(
        detections
    )

    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            height=220,
            width="stretch",
        )
    else:
        st.warning(
            "No supported object detected "
            "above the selected threshold."
        )


def main():
    st.set_page_config(
        page_title="Smart Recycle AI-Hub - V1 vs V2",
        page_icon="♻️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }

        h1 {
            margin-bottom: 0.2rem;
        }

        h2, h3 {
            margin-top: 0.4rem;
            margin-bottom: 0.4rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(120, 120, 120, 0.06);
            border-radius: 10px;
            padding: 8px 12px;
        }

        [data-testid="stImage"] {
            text-align: center;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_column, status_column = st.columns(
        [3, 1]
    )

    with title_column:
        st.title(
            "♻️ Smart Recycle AI-Hub"
        )

        st.caption(
            "YOLO26n V1 vs V2 — Same image, same threshold"
        )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    if "tested_image_ids" not in st.session_state:
        st.session_state.tested_image_ids = set()

    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0

    control_column_1, control_column_2 = st.columns(
        [5, 1]
    )

    with control_column_1:
        uploaded_files = st.file_uploader(
            "Upload waste images",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            accept_multiple_files=True,
            key=(
                "comparison_images_"
                f"{st.session_state.uploader_version}"
            ),
        )

    with control_column_2:
        st.write("")

        if st.button(
            "Reset session",
            use_container_width=True,
        ):
            st.session_state.tested_image_ids = set()
            st.session_state.uploader_version += 1
            st.rerun()

    if not uploaded_files:
        st.info(
            "Upload an image to compare V1 and V2."
        )
        return

    file_names = [
        uploaded_file.name
        for uploaded_file in uploaded_files
    ]

    selected_name = st.selectbox(
        "Select image",
        file_names,
    )

    selected_file = next(
        uploaded_file
        for uploaded_file in uploaded_files
        if uploaded_file.name == selected_name
    )

    image = prepare_image(
        selected_file
    )

    v1_model = get_v1_model()
    v2_model = get_v2_model()

    v1_annotated_image, v1_detections = analyze_image(
        image=image,
        model=v1_model,
        confidence_threshold=confidence_threshold,
    )

    v2_annotated_image, v2_detections = analyze_image(
        image=image,
        model=v2_model,
        confidence_threshold=confidence_threshold,
    )

    image_id = get_image_id(
        selected_file
    )

    st.session_state.tested_image_ids.add(
        image_id
    )

    (
        image_count_column,
        tested_column,
        v1_count_column,
        v2_count_column,
    ) = st.columns(4)

    with image_count_column:
        st.metric(
            "Uploaded",
            len(uploaded_files),
        )

    with tested_column:
        st.metric(
            "Tested",
            len(
                st.session_state.tested_image_ids
            ),
        )

    with v1_count_column:
        st.metric(
            "V1 detections",
            len(v1_detections),
        )

    with v2_count_column:
        st.metric(
            "V2 detections",
            len(v2_detections),
        )

    st.divider()

    st.markdown(
        "### Original Image"
    )

    left_space, image_column, right_space = st.columns(
        [1, 1, 1]
    )

    with image_column:
        st.image(
            image,
            caption=selected_name,
            width=ORIGINAL_IMAGE_WIDTH,
        )

    st.divider()

    v1_column, v2_column = st.columns(
        2
    )

    with v1_column:
        show_model_panel(
            title="V1",
            subtitle="YOLO26n V1 baseline",
            annotated_image=v1_annotated_image,
            detections=v1_detections,
        )

    with v2_column:
        show_model_panel(
            title="V2",
            subtitle="YOLO26n V2 MuSGD best.pt",
            annotated_image=v2_annotated_image,
            detections=v2_detections,
        )

    st.divider()

    st.caption(
        "Important: More detections do not automatically mean "
        "better performance. Check correct classes, misses, "
        "false positives, and confidence."
    )


if __name__ == "__main__":
    main()