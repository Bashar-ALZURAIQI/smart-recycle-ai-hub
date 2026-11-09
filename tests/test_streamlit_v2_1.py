from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from ai.inference import (
    SUPPORTED_CLASSES,
    V2_1_MODEL_PATH,
)
from ai.streamlit_app import (
    get_v2_1_model,
)


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "ai"
    / "streamlit_app.py"
)


def make_test_image_bytes():
    image_buffer = BytesIO()

    Image.new(
        "RGB",
        (120, 90),
        (255, 0, 0),
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


def test_streamlit_default_threshold_is_v2_1_operating_threshold():
    app = AppTest.from_file(
        APP_PATH
    ).run()

    assert len(app.slider) == 1

    assert (
        app.slider[0].label
        == "Confidence threshold"
    )

    assert (
        app.slider[0].value
        == 0.50
    )


def test_streamlit_can_load_v2_1_model():
    model = get_v2_1_model()

    assert (
        Path(model.ckpt_path).resolve()
        == V2_1_MODEL_PATH.resolve()
    )

    assert (
        list(model.names.values())
        == SUPPORTED_CLASSES
    )


def test_streamlit_compares_v2_with_v2_1():
    image_bytes = make_test_image_bytes()

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
        "V2 detections"
        in metrics
    )

    assert (
        "V2.1 detections"
        in metrics
    )

    assert (
        "V1 detections"
        not in metrics
    )