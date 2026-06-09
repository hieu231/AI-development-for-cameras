from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from src.utils.parameter_defaults import get_default_additional_parameters


def test_apply_mapping_request_creates_and_disables_assignments(monkeypatch):
    from src.models.ai_model import AiModel
    from src.models.camera_model import CameraModel

    import src.core.mqtt_bridge as mqtt_bridge

    camera_id = uuid4()
    keep_model_id = uuid4()
    remove_model_id = uuid4()
    new_model_id = uuid4()

    keep_model = AiModel(
        id=keep_model_id,
        name="Helmet",
        description="helmet",
        version="1.0.0",
        model_path="weights.pt",
        parameters={"model_type": "helmet_detection"},
        is_active=True,
        is_latest_used=True,
        changelog=None,
    )
    new_model = AiModel(
        id=new_model_id,
        name="Smoke",
        description="smoke",
        version="1.0.0",
        model_path="weights.pt",
        parameters={"model_type": "smoke_fire"},
        is_active=True,
        is_latest_used=True,
        changelog=None,
    )

    existing_enabled = CameraModel(
        camera_id=camera_id,
        model_id=keep_model_id,
        is_enabled=True,
        additional_parameters={"roi": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    )
    existing_enabled.ai_model = keep_model

    existing_to_disable = CameraModel(
        camera_id=camera_id,
        model_id=remove_model_id,
        is_enabled=True,
        additional_parameters={"roi": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    )

    assignments_by_model = {
        keep_model_id: existing_enabled,
        remove_model_id: existing_to_disable,
    }
    active_models = [existing_enabled, existing_to_disable]
    models_by_id = {
        keep_model_id: keep_model,
        new_model_id: new_model,
    }
    added: list[CameraModel] = []
    commits: list[str] = []
    reloaded: list[object] = []

    class FakeQuery:
        def __init__(self, model):
            self.model = model
            self._filters = []

        def filter(self, *conditions):
            self._filters.extend(conditions)
            return self

        def first(self):
            if self.model is CameraModel:
                return assignments_by_model.get(self._filters[1].right.value)
            if self.model is AiModel:
                return models_by_id.get(self._filters[0].right.value)
            return None

        def all(self):
            if self.model is CameraModel:
                return list(active_models)
            return []

    class FakeSession:
        def query(self, model):
            return FakeQuery(model)

        def add(self, value):
            added.append(value)
            assignments_by_model[value.model_id] = value

        def commit(self):
            commits.append("commit")

        def rollback(self):
            raise AssertionError("rollback should not be called")

        def close(self):
            return None

    monkeypatch.setattr(mqtt_bridge, "flag_modified", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mqtt_bridge.thread_manager,
        "reload_models",
        lambda value: reloaded.append(value) or True,
    )

    service = mqtt_bridge.CameraMappingBridgeService()
    response = service.apply_mapping_request(
        db=FakeSession(),
        payload={
            "correlationId": "corr-1",
            "cameraId": str(camera_id),
            "baseModelConfigVersion": 7,
            "aiModelIds": [str(keep_model_id), str(new_model_id)],
        },
    )

    assert response["correlationId"] == "corr-1"
    assert response["accepted"] is True
    assert response["reason"] is None
    assert response["aiModelIds"] == [str(keep_model_id), str(new_model_id)]
    assert commits == ["commit"]
    assert reloaded == [camera_id]
    assert existing_enabled.is_enabled is True
    assert existing_to_disable.is_enabled is False
    assert len(added) == 1
    assert added[0].camera_id == camera_id
    assert added[0].model_id == new_model_id
    assert added[0].is_enabled is True
    assert added[0].additional_parameters == get_default_additional_parameters(
        "smoke_fire"
    )


def test_apply_mapping_request_rejects_unknown_model_ids(monkeypatch):
    import src.core.mqtt_bridge as mqtt_bridge

    camera_id = uuid4()
    requested_model_id = uuid4()

    class FakeQuery:
        def __init__(self, model):
            self.model = model

        def filter(self, *_conditions):
            return self

        def first(self):
            return None

        def all(self):
            return []

    class FakeSession:
        def query(self, model):
            return FakeQuery(model)

        def add(self, _value):
            raise AssertionError("add should not be called")

        def commit(self):
            raise AssertionError("commit should not be called")

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        mqtt_bridge.thread_manager, "reload_models", lambda _value: True
    )

    service = mqtt_bridge.CameraMappingBridgeService()
    response = service.apply_mapping_request(
        db=FakeSession(),
        payload={
            "correlationId": "corr-2",
            "cameraId": str(camera_id),
            "baseModelConfigVersion": 1,
            "aiModelIds": [str(requested_model_id)],
        },
    )

    assert response == {
        "correlationId": "corr-2",
        "accepted": False,
        "reason": f"AI Model {requested_model_id} not found",
        "committedModelConfigVersion": 1,
        "committedChecksum": None,
        "aiModelIds": [],
    }


def test_build_mapping_topics_uses_configured_namespace():
    import src.core.mqtt_bridge as mqtt_bridge

    settings = mqtt_bridge.MqttBridgeSettings(
        enabled=True,
        host="127.0.0.1",
        port=1883,
        topic_namespace="aibox/org-1/loc-1/edge-001",
    )

    assert (
        settings.mapping_request_topic
        == "aibox/org-1/loc-1/edge-001/camera/+/mapping/request"
    )
    assert settings.mapping_response_topic(str(uuid4())).endswith("/mapping/response")


def test_handle_message_routes_response_to_camera_topic(monkeypatch):
    import src.core.mqtt_bridge as mqtt_bridge

    camera_id = str(uuid4())
    settings = mqtt_bridge.MqttBridgeSettings(
        enabled=True,
        host="127.0.0.1",
        port=1883,
        topic_namespace="aibox/org-1/loc-1/edge-001",
    )
    bridge = mqtt_bridge.MqttBridge(settings=settings)

    monkeypatch.setattr(
        mqtt_bridge,
        "SessionLocal",
        lambda: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        bridge.mapping_service,
        "apply_mapping_request",
        lambda _db, payload: {
            "correlationId": payload["correlationId"],
            "accepted": True,
            "reason": None,
            "committedModelConfigVersion": payload["baseModelConfigVersion"] + 1,
            "committedChecksum": "checksum-1",
            "aiModelIds": payload["aiModelIds"],
        },
    )

    response = bridge.handle_message(
        topic=f"aibox/org-1/loc-1/edge-001/camera/{camera_id}/mapping/request",
        payload_raw=(
            "{"
            '"correlationId":"corr-3",'
            f'"cameraId":"{camera_id}",'
            '"baseModelConfigVersion":2,'
            '"aiModelIds":[]'
            "}"
        ),
    )

    assert response == {
        "correlationId": "corr-3",
        "accepted": True,
        "reason": None,
        "committedModelConfigVersion": 3,
        "committedChecksum": "checksum-1",
        "aiModelIds": [],
        "_cameraId": camera_id,
    }
