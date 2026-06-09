from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from src.core.thread_manager import thread_manager
from src.database import SessionLocal
from src.models.ai_model import AiModel
from src.models.camera_model import CameraModel
from src.utils.parameter_defaults import get_default_additional_parameters

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - optional runtime dependency
    mqtt = None


@dataclass(slots=True)
class MqttBridgeSettings:
    enabled: bool
    host: str
    port: int
    topic_namespace: str
    client_id: str = "ai-be-pa-bridge"

    @property
    def mapping_request_topic(self) -> str:
        return f"{self.topic_namespace}/camera/+/mapping/request"

    def mapping_response_topic(self, camera_id: str) -> str:
        return f"{self.topic_namespace}/camera/{camera_id}/mapping/response"

    @classmethod
    def from_env(cls) -> "MqttBridgeSettings":
        enabled = os.getenv("MQTT_BRIDGE_ENABLED", "false").lower() == "true"
        host = os.getenv("MQTT_BRIDGE_HOST") or os.getenv("EMQX_HOST") or "localhost"
        port = int(os.getenv("MQTT_BRIDGE_PORT") or os.getenv("EMQX_PORT") or "1883")
        topic_namespace = (
            os.getenv("MQTT_BRIDGE_TOPIC_NAMESPACE", "").strip().strip("/")
        )
        client_id = os.getenv("MQTT_BRIDGE_CLIENT_ID", "ai-be-pa-bridge")
        return cls(
            enabled=enabled,
            host=host,
            port=port,
            topic_namespace=topic_namespace,
            client_id=client_id,
        )


class CameraMappingBridgeService:
    def apply_mapping_request(
        self, db: Session, payload: dict[str, Any]
    ) -> dict[str, Any]:
        correlation_id = str(payload.get("correlationId") or "")
        camera_id_raw = str(payload.get("cameraId") or "")
        requested_ids = self._normalize_model_ids(payload.get("aiModelIds"))
        base_version = int(payload.get("baseModelConfigVersion") or 0)

        if not correlation_id:
            return self._rejected_response(
                "", "Missing correlationId", [], base_version
            )
        if not camera_id_raw:
            return self._rejected_response(
                correlation_id, "Missing cameraId", [], base_version
            )

        try:
            camera_id = UUID(camera_id_raw)
        except ValueError:
            return self._rejected_response(
                correlation_id, f"Invalid cameraId {camera_id_raw}", [], base_version
            )

        desired_models: list[AiModel] = []
        for model_id in requested_ids:
            ai_model = db.query(AiModel).filter(AiModel.id == model_id).first()
            if ai_model is None:
                return self._rejected_response(
                    correlation_id,
                    f"AI Model {model_id} not found",
                    [],
                    base_version,
                )
            desired_models.append(ai_model)

        existing_assignments = (
            db.query(CameraModel).filter(CameraModel.camera_id == camera_id).all()
        )
        assignments_by_model_id = {
            assignment.model_id: assignment for assignment in existing_assignments
        }

        for assignment in existing_assignments:
            assignment.is_enabled = assignment.model_id in requested_ids

        for ai_model in desired_models:
            assignment = assignments_by_model_id.get(ai_model.id)
            if assignment is None:
                assignment = CameraModel(
                    camera_id=camera_id,
                    model_id=ai_model.id,
                    is_enabled=True,
                    additional_parameters=get_default_additional_parameters(
                        ai_model.model_type
                    ),
                )
                db.add(assignment)
                if assignment.additional_parameters:
                    flag_modified(assignment, "additional_parameters")
                assignments_by_model_id[ai_model.id] = assignment
            else:
                assignment.is_enabled = True
                if not assignment.additional_parameters:
                    assignment.additional_parameters = (
                        get_default_additional_parameters(ai_model.model_type)
                    )
                    if assignment.additional_parameters:
                        flag_modified(assignment, "additional_parameters")

        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error(
                "Failed to apply mapping request for camera %s: %s",
                camera_id,
                exc,
                exc_info=True,
            )
            return self._rejected_response(correlation_id, str(exc), [], base_version)

        thread_manager.reload_models(camera_id)

        committed_model_ids = [str(model_id) for model_id in requested_ids]
        return {
            "correlationId": correlation_id,
            "accepted": True,
            "reason": None,
            "committedModelConfigVersion": base_version + 1,
            "committedChecksum": self._checksum(committed_model_ids),
            "aiModelIds": committed_model_ids,
        }

    def _normalize_model_ids(self, values: Any) -> list[UUID]:
        if not isinstance(values, list):
            return []
        normalized: list[UUID] = []
        seen: set[UUID] = set()
        for value in values:
            try:
                model_id = UUID(str(value).strip())
            except ValueError:
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            normalized.append(model_id)
        return normalized

    def _checksum(self, model_ids: list[str]) -> str:
        digest = hashlib.sha256("|".join(model_ids).encode("utf-8")).hexdigest()
        return digest

    def _rejected_response(
        self,
        correlation_id: str,
        reason: str,
        ai_model_ids: list[str],
        base_version: int,
    ) -> dict[str, Any]:
        return {
            "correlationId": correlation_id,
            "accepted": False,
            "reason": reason,
            "committedModelConfigVersion": base_version,
            "committedChecksum": None,
            "aiModelIds": ai_model_ids,
        }


class MqttBridge:
    def __init__(self, settings: MqttBridgeSettings | None = None):
        self.settings = settings or MqttBridgeSettings.from_env()
        self.mapping_service = CameraMappingBridgeService()
        self.client: Any = None

    def start(self) -> None:
        if not self.settings.enabled:
            logger.info("MQTT bridge disabled")
            return
        if not self.settings.topic_namespace:
            logger.warning(
                "MQTT bridge enabled but MQTT_BRIDGE_TOPIC_NAMESPACE is empty"
            )
            return
        if mqtt is None:
            logger.warning("MQTT bridge enabled but paho-mqtt is not installed")
            return

        client = mqtt.Client(client_id=self.settings.client_id)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect(self.settings.host, self.settings.port)
        client.loop_start()
        self.client = client
        logger.info(
            "MQTT bridge connected to %s:%s and listening on %s",
            self.settings.host,
            self.settings.port,
            self.settings.mapping_request_topic,
        )

    def stop(self) -> None:
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.client = None

    def _on_connect(
        self,
        client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _properties: Any = None,
    ) -> None:
        if reason_code != 0:
            logger.error("MQTT bridge failed to connect: %s", reason_code)
            return
        client.subscribe(self.settings.mapping_request_topic)

    def _on_message(self, client: Any, _userdata: Any, message: Any) -> None:
        response_payload = self.handle_message(
            message.topic, message.payload.decode("utf-8")
        )
        if response_payload is None:
            return
        camera_id = response_payload.pop("_cameraId")
        client.publish(
            self.settings.mapping_response_topic(camera_id),
            json.dumps(response_payload),
        )

    def handle_message(self, topic: str, payload_raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid MQTT mapping payload on topic %s", topic)
            return None

        camera_id = payload.get("cameraId")
        if not isinstance(camera_id, str):
            logger.warning(
                "Ignoring mapping payload without cameraId on topic %s", topic
            )
            return None

        db = SessionLocal()
        try:
            response = self.mapping_service.apply_mapping_request(db, payload)
        finally:
            db.close()

        response["_cameraId"] = camera_id
        return response


mqtt_bridge = MqttBridge()
