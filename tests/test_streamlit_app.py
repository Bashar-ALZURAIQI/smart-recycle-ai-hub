from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from ai.inference import (
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    load_model,
)
from ai.streamlit_app import (
    analyze_image,
    get_best_detection,
    main,
    make_detection_rows,
    prepare_image,
)


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "ai"
    / "streamlit_app.py"
)


def make_test_image_bytes(color):
    image_buffer = BytesIO()

    Image.new(
        "RGB",
        (120, 90),
        color,
    ).save(
        image_buffer,
        format="PNG",
    )

    return image_buffer.getvalue()


def metric_values(app):
    return {
        metric.label: metric.value
        for metric in app.metric
    }


def test_streamlit_app_has_main_function():
    assert callable(main)


def test_prepare_image_returns_rgb_image():
    image_buffer = BytesIO()

    Image.new(
        "RGBA",
        (100, 80),
        (255, 0, 0, 128),
    ).save(
        image_buffer,
        format="PNG",
    )

    image_buffer.seek(0)

    image = prepare_image(
        image_buffer
    )

    assert image.mode == "RGB"
    assert image.size == (100, 80)


def test_app_has_confidence_slider():
    app = AppTest.from_file(
        APP_PATH
    ).run()

    assert len(
        app.slider
    ) == 1

    assert (
        app.slider[0].label
        == "Confidence threshold"
    )

    assert (
        app.slider[0].value
        == 0.25
    )


def test_app_has_multiple_image_uploader():
    app = AppTest.from_file(
        APP_PATH
    ).run()

    assert len(
        app.file_uploader
    ) == 1

    uploader = app.file_uploader[0]

    assert (
        uploader.label
        == "Upload waste images"
    )

    assert (
        uploader.accept_multiple_files
        is True
    )


def test_reset_session_button_exists():
    app = AppTest.from_file(
        APP_PATH
    ).run()

    assert len(
        app.button
    ) == 1

    assert (
        app.button[0].label
        == "Reset session"
    )


def test_uploaded_images_have_selector():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    assert len(
        app.selectbox
    ) == 1

    assert (
        app.selectbox[0].label
        == "Select image"
    )


def test_multiple_uploaded_images_are_available_in_selector():
    red_image = make_test_image_bytes(
        (255, 0, 0)
    )

    blue_image = make_test_image_bytes(
        (0, 0, 255)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].set_value(
        [
            (
                "red.png",
                red_image,
                "image/png",
            ),
            (
                "blue.png",
                blue_image,
                "image/png",
            ),
        ]
    )

    app.run(
        timeout=60
    )

    assert (
        app.selectbox[0].options
        == [
            "red.png",
            "blue.png",
        ]
    )

    app.selectbox[0].set_value(
        "blue.png"
    ).run(
        timeout=60
    )

    assert (
        app.selectbox[0].value
        == "blue.png"
    )


def test_uploaded_count_metric():
    red_image = make_test_image_bytes(
        (255, 0, 0)
    )

    blue_image = make_test_image_bytes(
        (0, 0, 255)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].set_value(
        [
            (
                "red.png",
                red_image,
                "image/png",
            ),
            (
                "blue.png",
                blue_image,
                "image/png",
            ),
        ]
    )

    app.run(
        timeout=60
    )

    metrics = metric_values(
        app
    )

    assert (
        metrics["Uploaded"]
        == "2"
    )


def test_tested_metric_counts_selected_image():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    metrics = metric_values(
        app
    )

    assert (
        metrics["Tested"]
        == "1"
    )


def test_tested_metric_does_not_duplicate_same_image():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    first_metrics = metric_values(
        app
    )

    app.run(
        timeout=60
    )

    second_metrics = metric_values(
        app
    )

    assert (
        first_metrics["Tested"]
        == "1"
    )

    assert (
        second_metrics["Tested"]
        == "1"
    )


def test_v1_and_v2_detection_metrics_exist():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    metrics = metric_values(
        app
    )

    assert (
        "V1 detections"
        in metrics
    )

    assert (
        "V2 detections"
        in metrics
    )

    assert int(
        metrics["V1 detections"]
    ) >= 0

    assert int(
        metrics["V2 detections"]
    ) >= 0


def test_reset_session_clears_uploaded_and_tested_counts():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    metrics = metric_values(
        app
    )

    assert (
        metrics["Uploaded"]
        == "1"
    )

    assert (
        metrics["Tested"]
        == "1"
    )

    app.button[0].click().run(
        timeout=60
    )

    assert len(
        app.file_uploader[0].value
    ) == 0


def test_uploaded_image_shows_original_v1_and_v2_images():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    assert len(
        app.image
    ) == 3


def test_comparison_page_contains_v1_and_v2_titles():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    markdown_texts = [
        markdown.value
        for markdown in app.markdown
    ]

    assert any(
        "V1" in text
        for text in markdown_texts
    )

    assert any(
        "V2" in text
        for text in markdown_texts
    )


def test_uploaded_image_shows_two_detection_tables():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(
        APP_PATH
    ).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(
        timeout=60
    )

    assert len(
        app.dataframe
    ) <= 2


def test_make_detection_rows_returns_expected_fields():
    detections = [
        {
            "class": "plastic",
            "confidence": 0.87654,
            "box": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    ]

    rows = make_detection_rows(
        detections
    )

    assert rows == [
        {
            "#": 1,
            "Class": "plastic",
            "Confidence": 0.877,
        }
    ]


def test_get_best_detection_returns_highest_confidence():
    detections = [
        {
            "class": "glass",
            "confidence": 0.42,
            "box": [
                1,
                2,
                3,
                4,
            ],
        },
        {
            "class": "metal",
            "confidence": 0.81,
            "box": [
                5,
                6,
                7,
                8,
            ],
        },
    ]

    best = get_best_detection(
        detections
    )

    assert (
        best["class"]
        == "metal"
    )

    assert (
        best["confidence"]
        == 0.81
    )


def test_get_best_detection_returns_none_for_empty_list():
    assert (
        get_best_detection([])
        is None
    )


def test_analyze_image_returns_v1_result():
    model = load_model(
        V1_MODEL_PATH
    )

    image = Image.new(
        "RGB",
        (640, 640),
        (0, 0, 0),
    )

    annotated_image, detections = analyze_image(
        image=image,
        model=model,
        confidence_threshold=0.25,
    )

    assert (
        annotated_image.shape[:2]
        == (640, 640)
    )

    assert (
        annotated_image.shape[2]
        == 3
    )

    assert isinstance(
        detections,
        list,
    )


def test_analyze_image_returns_v2_result():
    model = load_model(
        V2_MODEL_PATH
    )

    image = Image.new(
        "RGB",
        (640, 640),
        (0, 0, 0),
    )

    annotated_image, detections = analyze_image(
        image=image,
        model=model,
        confidence_threshold=0.25,
    )

    assert (
        annotated_image.shape[:2]
        == (640, 640)
    )

    assert (
        annotated_image.shape[2]
        == 3
    )

    assert isinstance(
        detections,
        list,
    )