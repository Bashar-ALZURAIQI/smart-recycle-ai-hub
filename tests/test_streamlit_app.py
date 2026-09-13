from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from ai.inference import load_model
from ai.streamlit_app import analyze_image, main, prepare_image


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

    image = prepare_image(image_buffer)

    assert image.mode == "RGB"
    assert image.size == (100, 80)


def test_app_has_multiple_image_uploader():
    app = AppTest.from_file(APP_PATH).run()

    assert len(app.file_uploader) == 1
    assert app.file_uploader[0].label == "Upload waste images"
    assert app.file_uploader[0].accept_multiple_files is True


def test_reset_session_button_exists():
    app = AppTest.from_file(APP_PATH).run()

    assert len(app.button) == 1
    assert app.button[0].label == "Reset session"


def test_uploaded_images_have_selector():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    assert len(app.selectbox) == 1
    assert app.selectbox[0].label == "Select image"


def test_multiple_uploaded_images_are_available_in_selector():
    red_image = make_test_image_bytes(
        (255, 0, 0)
    )

    blue_image = make_test_image_bytes(
        (0, 0, 255)
    )

    app = AppTest.from_file(APP_PATH).run()

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

    app.run(timeout=30)

    assert app.selectbox[0].options == [
        "red.png",
        "blue.png",
    ]

    app.selectbox[0].set_value(
        "blue.png"
    ).run(timeout=30)

    assert app.selectbox[0].value == "blue.png"


def test_current_images_metric_counts_uploaded_files():
    red_image = make_test_image_bytes(
        (255, 0, 0)
    )

    blue_image = make_test_image_bytes(
        (0, 0, 255)
    )

    app = AppTest.from_file(APP_PATH).run()

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

    app.run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Current images"] == "2"


def test_tested_session_counts_same_image_only_once():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Tested this session"] == "1"

    app.run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Tested this session"] == "1"


def test_tested_session_counts_second_selected_image():
    red_image = make_test_image_bytes(
        (255, 0, 0)
    )

    blue_image = make_test_image_bytes(
        (0, 0, 255)
    )

    app = AppTest.from_file(APP_PATH).run()

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

    app.run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Tested this session"] == "1"

    app.selectbox[0].set_value(
        "blue.png"
    ).run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Tested this session"] == "2"


def test_total_detections_is_not_counted_twice_on_rerun():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    metrics = metric_values(app)

    assert "Total detections" in metrics

    first_total = metrics["Total detections"]

    assert int(first_total) >= 0

    app.run(timeout=30)

    metrics = metric_values(app)

    second_total = metrics["Total detections"]

    assert second_total == first_total


def test_reset_session_clears_uploads_and_statistics():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Current images"] == "1"
    assert metrics["Tested this session"] == "1"

    app.button[0].click().run(timeout=30)

    metrics = metric_values(app)

    assert metrics["Current images"] == "0"
    assert metrics["Tested this session"] == "0"
    assert metrics["Total detections"] == "0"


def test_metrics_are_side_by_side():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    assert len(app.columns) == 5

    assert len(app.columns[0].metric) == 1
    assert len(app.columns[1].metric) == 1
    assert len(app.columns[2].metric) == 1

    assert app.columns[0].metric[0].label == "Current images"
    assert app.columns[1].metric[0].label == "Tested this session"
    assert app.columns[2].metric[0].label == "Total detections"


def test_uploaded_image_shows_original_and_annotated_images():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    assert len(app.image) == 2


def test_original_and_annotated_images_are_side_by_side():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    assert len(app.columns) >= 2

    image_columns = app.columns[-2:]

    assert len(image_columns[0].image) == 1
    assert len(image_columns[1].image) == 1


def test_uploaded_image_shows_detections_table():
    image_bytes = make_test_image_bytes(
        (255, 0, 0)
    )

    app = AppTest.from_file(APP_PATH).run()

    app.file_uploader[0].upload(
        "test.png",
        image_bytes,
        "image/png",
    ).run(timeout=30)

    assert len(app.dataframe) == 1


def test_analyze_image_returns_annotated_image_and_detections():
    model = load_model()

    image = Image.new(
        "RGB",
        (640, 640),
        (0, 0, 0),
    )

    annotated_image, detections = analyze_image(
        image=image,
        model=model,
    )

    assert annotated_image.shape[:2] == (640, 640)
    assert annotated_image.shape[2] == 3
    assert isinstance(detections, list)