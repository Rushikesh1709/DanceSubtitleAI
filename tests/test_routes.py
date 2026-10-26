"""Flask route tests. The pose model is replaced by FakeBackend (see conftest.py)."""
import io
import json

import pytest

from helpers import make_video
from app import create_app
from app.models import ProcessingJob
from app.services import pipeline


def upload(client, path, name="dance.mp4", **extra):
    data = {"video": (open(path, "rb"), name), "title": "Test dance", "dance_form": "Bharatanatyam",
            "sampling_fps": "5", "language": "en", **extra}
    return client.post("/upload", data=data, content_type="multipart/form-data")


@pytest.fixture()
def finished_job(app, client, tmp_path):
    video = make_video(tmp_path / "v.mp4", n_frames=100, fps=10)       # 10 s
    resp = upload(client, video)
    assert resp.status_code == 302 and "/processing" in resp.headers["Location"]
    pid = resp.headers["Location"].split("/")[2]
    with app.extensions["db_factory"]() as s:
        job_id = s.query(ProcessingJob).filter_by(public_id=pid).one().id
    pipeline.run_job(app, job_id)                                       # run synchronously
    return pid


def test_pages_load(client):
    for url in ("/", "/upload", "/history", "/settings"):
        assert client.get(url).status_code == 200
    assert b"Upload a video" in client.get("/").data


def test_upload_rejects_wrong_type_and_missing_file(client):
    r = client.post("/upload", data={"video": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"Unsupported file type" in r.data
    r = client.post("/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"choose a video" in r.data


def test_upload_rejects_corrupted_video_and_cleans_up(app, client):
    r = client.post("/upload", data={"video": (io.BytesIO(b"junk" * 500), "bad.mp4")}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"could not be opened" in r.data or b"could not be" in r.data
    assert list(app.config["UPLOAD_DIR"].iterdir()) == []                # file removed again


def test_upload_rejects_bad_sampling_value(client, tmp_path):
    video = make_video(tmp_path / "v.mp4", n_frames=20)
    r = upload(client, video, sampling_fps="500")
    assert r.status_code == 400


def test_full_pipeline_with_mocked_model(app, client, finished_job):
    pid = finished_job
    status = client.get(f"/api/jobs/{pid}/status").get_json()
    assert status["status"] == "completed" and status["frames_done"] > 30
    page = client.get(f"/jobs/{pid}")
    assert page.status_code == 200 and b"Download SRT" in page.data and b"<video" in page.data
    segs = client.get(f"/api/jobs/{pid}/segments").get_json()["segments"]
    assert len(segs) >= 3
    assert any("raises both arms" in s["text"] for s in segs)
    for url in (f"/jobs/{pid}/edit", f"/jobs/{pid}/visualization", f"/jobs/{pid}/video",
                f"/api/jobs/{pid}/analysis", f"/jobs/{pid}/download/features"):
        assert client.get(url).status_code == 200, url
    analysis = client.get(f"/api/jobs/{pid}/analysis").get_json()
    assert analysis["samples"][0]["pose"][11][2] > 0.5 and analysis["meta"]["duration"] > 9


def test_downloads_are_valid_srt_and_vtt(client, finished_job):
    pid = finished_job
    srt = client.get(f"/jobs/{pid}/download/srt")
    assert srt.status_code == 200 and "attachment" in srt.headers["Content-Disposition"]
    text = srt.data.decode("utf-8")
    assert text.startswith("1\n00:00:") and " --> " in text
    vtt = client.get(f"/jobs/{pid}/download/vtt").data.decode("utf-8")
    assert vtt.startswith("WEBVTT")
    assert client.get(f"/jobs/{pid}/subtitles.vtt").mimetype == "text/vtt"


def test_editing_persists_and_is_exported(client, finished_job):
    pid = finished_job
    url = f"/api/jobs/{pid}/segments"
    segs = client.get(url).get_json()["segments"]
    segs[0]["text"] = "Edited by the user — नमस्ते"
    segs[0]["start"], segs[0]["end"] = 0.5, 2.0
    r = client.put(url, json={"segments": segs[:1] + [dict(s, start=s["start"], end=s["end"]) for s in segs[1:]]})
    assert r.status_code == 200 and r.get_json()["ok"]
    assert "Edited by the user" in client.get(f"/jobs/{pid}/download/srt").data.decode("utf-8")
    assert "Edited by the user" in client.get(f"/jobs/{pid}/subtitles.vtt").data.decode("utf-8")


def test_invalid_edits_are_rejected(client, finished_job):
    url = f"/api/jobs/{finished_job}/segments"
    bad = [{"start": 2, "end": 1, "text": "x"}]
    assert client.put(url, json={"segments": bad}).status_code == 400
    overlap = [{"start": 0, "end": 3, "text": "a"}, {"start": 2, "end": 4, "text": "b"}]
    r = client.put(url, json={"segments": overlap})
    assert r.status_code == 400 and "overlaps" in r.get_json()["errors"][0]
    assert client.put(url, json={"segments": [{"start": "zz", "end": 1, "text": "a"}]}).status_code == 400
    assert client.put(url, data="not json", content_type="text/plain").status_code == 400


def test_transform_merge_and_split(client, finished_job):
    url = f"/api/jobs/{finished_job}/segments/transform"
    segs = [{"start": 0, "end": 2, "text": "one two three four"}, {"start": 2.1, "end": 4, "text": "five"}]
    merged = client.post(url, json={"op": "merge", "index": 0, "segments": segs}).get_json()["segments"]
    assert len(merged) == 1
    split = client.post(url, json={"op": "split", "index": 0, "at": 1.0, "segments": segs}).get_json()["segments"]
    assert len(split) == 3
    assert client.post(url, json={"op": "split", "index": 0, "at": 0.1, "segments": segs}).status_code == 400
    assert client.post(url, json={"op": "zzz", "index": 0, "segments": segs}).status_code == 400


def test_unknown_job_and_path_traversal_are_404(client, finished_job):
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.get("/jobs/doesnotexist/video").status_code == 404
    assert client.get(f"/jobs/{finished_job}/download/../../config.py").status_code == 404
    assert client.get(f"/jobs/{finished_job}/download/passwd").status_code == 404
    assert client.get(f"/jobs/{finished_job}/download/video").status_code == 404   # nothing exported yet


def test_history_and_delete(app, client, finished_job):
    assert b"Test dance" in client.get("/history").data
    r = client.post(f"/jobs/{finished_job}/delete")
    assert r.status_code == 302
    assert client.get(f"/jobs/{finished_job}").status_code == 404
    assert list(app.config["UPLOAD_DIR"].iterdir()) == []


def test_settings_save_and_validate(client):
    r = client.post("/settings", data={"sampling_fps": "7", "sensitivity": "1.5", "output_format": "vtt"})
    assert r.status_code == 302
    assert b'value="7.0"' in client.get("/settings").data
    assert client.post("/settings", data={"sampling_fps": "999"}).status_code == 400


def test_pose_failure_is_reported_safely(tmp_path):
    class NoPerson:
        def __init__(self, o=None): pass
        def analyze(self, f): return {"pose": None, "hands": {}, "face": None}
        def close(self): pass
    app = create_app({"TESTING": True, "DATA_DIR": tmp_path / "d", "AUTO_START_JOBS": False,
                      "BACKEND_FACTORY": lambda o: NoPerson()})
    client = app.test_client()
    video = make_video(tmp_path / "v.mp4", n_frames=30)
    resp = upload(client, video)
    pid = resp.headers["Location"].split("/")[2]
    with app.extensions["db_factory"]() as s:
        job_id = s.query(ProcessingJob).filter_by(public_id=pid).one().id
    pipeline.run_job(app, job_id)
    status = client.get(f"/api/jobs/{pid}/status").get_json()
    assert status["status"] == "failed" and "No person was detected" in status["error"]
    assert "Traceback" not in status["error"]
    assert b"Traceback" not in client.get(f"/jobs/{pid}/processing").data
