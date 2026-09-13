from hashlib import sha256

from PIL import Image
import streamlit as st

from ai.inference import (
    extract_detections,
    load_model,
    run_inference,
)


def prepare_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def analyze_image(image, model):
    results = run_inference(
        model=model,
        image=image,
    )

    result = results[0]

    annotated_image = result.plot()[..., ::-1].copy()

    detections = extract_detections(result)

    return annotated_image, detections


def get_image_id(uploaded_file):
    return sha256(
        uploaded_file.getvalue()
    ).hexdigest()


def main():
    st.set_page_config(
        page_title="Smart Recycle AI-Hub",
        page_icon="♻️",
        layout="wide",
    )

    st.title("♻️ Smart Recycle AI-Hub")
    st.write("V1 Image Inference Tester")

    if "tested_image_ids" not in st.session_state:
        st.session_state.tested_image_ids = set()

    if "detection_counts" not in st.session_state:
        st.session_state.detection_counts = {}

    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0

    if st.button(
        "Reset session",
    ):
        st.session_state.tested_image_ids = set()
        st.session_state.detection_counts = {}
        st.session_state.uploader_version += 1

    uploaded_files = st.file_uploader(
        "Upload waste images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=(
            "waste_images_"
            f"{st.session_state.uploader_version}"
        ),
    )

    selected_name = None
    image = None
    annotated_image = None
    detections = []

    if uploaded_files:
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

        image = prepare_image(selected_file)

        model = load_model()

        annotated_image, detections = analyze_image(
            image=image,
            model=model,
        )

        image_id = get_image_id(selected_file)

        st.session_state.tested_image_ids.add(
            image_id
        )

        if image_id not in st.session_state.detection_counts:
            st.session_state.detection_counts[
                image_id
            ] = len(detections)

    current_images_column, tested_column, detections_column = st.columns(3)

    with current_images_column:
        st.metric(
            "Current images",
            len(uploaded_files),
        )

    with tested_column:
        st.metric(
            "Tested this session",
            len(st.session_state.tested_image_ids),
        )

    with detections_column:
        st.metric(
            "Total detections",
            sum(
                st.session_state.detection_counts.values()
            ),
        )

    if uploaded_files:
        original_column, detection_column = st.columns(2)

        with original_column:
            st.subheader("Original image")

            st.image(
                image,
                caption=selected_name,
                width="stretch",
            )

        with detection_column:
            st.subheader("YOLO detection result")

            st.image(
                annotated_image,
                caption="Detected objects",
                width="stretch",
            )

        detection_rows = []

        for detection in detections:
            detection_rows.append(
                {
                    "Class": detection["class"],
                    "Confidence": round(
                        detection["confidence"],
                        3,
                    ),
                }
            )

        st.subheader("Detection results")

        st.dataframe(
            detection_rows,
            hide_index=True,
        )


if __name__ == "__main__":
    main()