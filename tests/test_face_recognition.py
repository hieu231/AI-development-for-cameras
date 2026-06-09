import base64
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from src.core.model_factory import ModelFactory
from src.face_recognition import face_engine as face_engine_module
from src.face_recognition import validator as validator_module
from src.face_recognition.face_engine import FaceRecognitionEngine
from src.face_recognition.schemas import FaceProfileCreate
from src.face_recognition.validator import FacePhotoValidator


def _make_base64_image() -> str:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success is True
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class DummyFaceEngine:
    def __init__(self, process_result=None, same_person_result=(True, 0.99)):
        self.process_result = process_result or {
            "success": True,
            "num_faces": 1,
            "faces": [{"embedding": [0.1, 0.2, 0.3], "confidence": 0.98}],
            "error": None,
        }
        self.same_person_result = same_person_result

    def process_image(self, image, min_resolution=0):
        return self.process_result

    def validate_same_person(self, embeddings, threshold=0.45):
        return self.same_person_result


class TestFaceEngineSingleton:
    def test_get_face_engine_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_RECOGNITION", "false")
        monkeypatch.setattr(face_engine_module, "_engine", None)

        assert face_engine_module.get_face_engine(force_reload=True) is None

    def test_get_face_engine_caches_initialized_instance(self, monkeypatch):
        created = []

        class StubEngine:
            def __init__(self):
                created.append(self)

        monkeypatch.setenv("ENABLE_FACE_RECOGNITION", "true")
        monkeypatch.setattr(face_engine_module, "_engine", None)
        monkeypatch.setattr(face_engine_module, "FaceRecognitionEngine", StubEngine)

        first = face_engine_module.get_face_engine(force_reload=True)
        second = face_engine_module.get_face_engine()

        assert first is second
        assert len(created) == 1


class TestFaceRecognitionEngineHelpers:
    def test_validate_same_person_returns_true_for_similar_embeddings(self):
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.8, 0.6], dtype=np.float32),
        ]

        is_same, min_similarity = engine.validate_same_person(embeddings, threshold=0.7)

        assert bool(is_same) is True
        assert min_similarity == pytest.approx(0.8, abs=1e-6)

    def test_validate_same_person_returns_false_for_dissimilar_embeddings(self):
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        is_same, min_similarity = engine.validate_same_person(embeddings, threshold=0.1)

        assert bool(is_same) is False
        assert min_similarity == pytest.approx(0.0, abs=1e-6)

    def test_compute_average_embedding_normalizes_output(self):
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]

        avg_embedding = engine.compute_average_embedding(embeddings)

        assert avg_embedding.tolist() == pytest.approx([1.0, 0.0])
        assert np.linalg.norm(avg_embedding) == pytest.approx(1.0)

    def test_compare_embeddings_returns_cosine_similarity(self):
        engine = FaceRecognitionEngine.__new__(FaceRecognitionEngine)

        similarity = engine.compare_embeddings([1.0, 0.0], [0.5, 0.5])

        assert similarity == pytest.approx(0.5)


class TestModelFactoryFaceRegistration:
    def test_get_available_types_includes_face_recognition_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_RECOGNITION", "true")

        available_types = ModelFactory.get_available_types()

        assert "face_recognition" in available_types

    def test_get_available_types_excludes_face_recognition_when_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FACE_RECOGNITION", "false")

        available_types = ModelFactory.get_available_types()

        assert "face_recognition" not in available_types

    def test_create_model_ignores_reserved_constructor_keys(self, monkeypatch):
        created = {}

        class DummyModel:
            def __init__(self, model_path=None, **kwargs):
                created["model_path"] = model_path
                created["kwargs"] = kwargs

        monkeypatch.setattr(
            ModelFactory,
            "_get_model_type_map",
            classmethod(lambda cls: {"dummy_model": "tests.dummy.DummyModel"}),
        )
        monkeypatch.setattr(ModelFactory, "_import_class", staticmethod(lambda _: DummyModel))

        instance = ModelFactory.create_model(
            {
                "name": "Dummy",
                "model_type": "dummy_model",
                "model_path": "weights/dummy.pt",
                "parameters": {
                    "default_alert_level": "HIGH",
                    "model_name": "InjectedName",
                    "confidence_threshold": 0.42,
                },
                "additional_params": {
                    "model_path": "ignored.pt",
                    "extra_flag": True,
                },
            }
        )

        assert isinstance(instance, DummyModel)
        assert created["model_path"] == "weights/dummy.pt"
        assert created["kwargs"] == {
            "confidence_threshold": 0.42,
            "extra_flag": True,
        }


class TestFacePhotoValidator:
    def test_decode_base64_image_rejects_invalid_payload(self):
        validator = FacePhotoValidator.__new__(FacePhotoValidator)

        success, image, error = validator.decode_base64_image("not-base64")

        assert success is False
        assert image is None
        assert "Invalid image format" in error

    def test_validate_single_photo_returns_no_face_error(self):
        validator = FacePhotoValidator.__new__(FacePhotoValidator)
        validator.min_resolution = 0
        validator.engine = DummyFaceEngine(
            process_result={
                "success": True,
                "num_faces": 0,
                "faces": [],
                "error": None,
            }
        )

        result = validator.validate_single_photo(_make_base64_image())

        assert result["valid"] is False
        assert result["error"] == "No face detected in image"

    def test_validate_single_photo_returns_embedding_for_single_face(self):
        monkeypatch_engine = DummyFaceEngine()
        validator = FacePhotoValidator.__new__(FacePhotoValidator)
        validator.min_resolution = 0
        validator.engine = monkeypatch_engine

        result = validator.validate_single_photo(_make_base64_image())

        assert result["valid"] is True
        assert result["num_faces"] == 1
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["confidence"] == pytest.approx(0.98)

    def test_validate_multiple_photos_rejects_different_people(self):
        validator = FacePhotoValidator.__new__(FacePhotoValidator)
        validator.min_resolution = 0
        validator.same_person_threshold = 0.45
        validator.engine = DummyFaceEngine(same_person_result=(False, 0.21))

        result = validator.validate_multiple_photos([
            _make_base64_image(),
            _make_base64_image(),
            _make_base64_image(),
        ])

        assert result["valid"] is False
        assert "different people" in result["error"]
        assert "0.21" in result["error"]

    def test_save_image_writes_png_file(self, tmp_path):
        validator = FacePhotoValidator.__new__(FacePhotoValidator)

        success, file_path, error = validator.save_image(
            _make_base64_image(),
            str(tmp_path),
            "employee_001",
        )

        assert success is True
        assert error == ""
        assert Path(file_path).exists()
        assert Path(file_path).suffix == ".png"

    def test_get_validator_caches_instance(self, monkeypatch):
        created = []

        class StubValidator:
            def __init__(self):
                created.append(self)

        monkeypatch.setattr(validator_module, "_validator", None)
        monkeypatch.setattr(validator_module, "FacePhotoValidator", StubValidator)

        first = validator_module.get_validator(force_reload=True)
        second = validator_module.get_validator()

        assert first is second
        assert len(created) == 1


class TestFaceRecognitionSchemas:
    def test_face_profile_create_accepts_exactly_three_images(self):
        payload = {
            "employee_id": "EMP001",
            "employee_name": "Nguyen Van A",
            "images_base64": [
                _make_base64_image(),
                _make_base64_image(),
                _make_base64_image(),
            ],
        }

        schema = FaceProfileCreate(**payload)

        assert schema.employee_id == "EMP001"
        assert len(schema.images_base64) == 3

    def test_face_profile_create_rejects_wrong_image_count(self):
        with pytest.raises(ValidationError):
            FaceProfileCreate(
                employee_id="EMP001",
                employee_name="Nguyen Van A",
                images_base64=[_make_base64_image(), _make_base64_image()],
            )

    def test_face_profile_create_rejects_invalid_base64(self):
        with pytest.raises(ValidationError):
            FaceProfileCreate(
                employee_id="EMP001",
                employee_name="Nguyen Van A",
                images_base64=["bad", "bad", "bad"],
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
