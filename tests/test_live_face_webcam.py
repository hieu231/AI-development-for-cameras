import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "live_face_webcam.py"
SPEC = importlib.util.spec_from_file_location("live_face_webcam", SCRIPT_PATH)
live_face_webcam = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(live_face_webcam)


class DummyProfile:
    def __init__(self, profile_id, employee_id, employee_name):
        self.id = profile_id
        self.employee_id = employee_id
        self.employee_name = employee_name


class DummyRepo:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def find_best_match(self, embedding, threshold, active_only):
        self.calls.append(
            {
                "embedding": embedding,
                "threshold": threshold,
                "active_only": active_only,
            }
        )
        return self.results[len(self.calls) - 1]


def test_find_face_matches_returns_profile_payloads():
    repo = DummyRepo(
        [
            (DummyProfile("profile-1", "EMP001", "Nguyen Van A"), 0.91),
            None,
        ]
    )

    matches = live_face_webcam.find_face_matches(
        repo,
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
        threshold=0.45,
    )

    assert matches == [
        {
            "profile_id": "profile-1",
            "employee_id": "EMP001",
            "employee_name": "Nguyen Van A",
            "similarity": 0.91,
        },
        None,
    ]
    assert repo.calls[0]["threshold"] == 0.45
    assert repo.calls[0]["active_only"] is True


def test_find_face_matches_returns_unknowns_when_repo_missing():
    matches = live_face_webcam.find_face_matches(
        None,
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
        threshold=0.45,
    )

    assert matches == [None, None]