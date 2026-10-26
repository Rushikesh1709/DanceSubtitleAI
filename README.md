# DanceSubtitleAI — AI-Based Automatic Subtitle Generation for Indian Classical Dance Videos

## 1. Project overview
A Flask web application that analyses a dance video (focus: **Bharatanatyam**) with computer vision and writes **descriptive,
time-synchronised subtitles** of the visible movement — e.g. *"The dancer raises both arms while standing upright."* You can
preview them next to the video, edit text and timing, and download **SRT / WebVTT** files.

It describes **observations only**. It does *not* decide the meaning of a mudra, a story, or the dancer's emotion (rasa). See `docs/limitations.md`.

## 2. Features
* Upload MP4 / MOV / AVI / MKV / WebM, with validation and corrupted-file handling
* Pose landmarks (MediaPipe Pose), skeleton overlay, movement-intensity chart
* Arm / leg / turning / sideways / held / fast / slow / repeated-pattern detection
* Optional hand **shape** analysis and optional facial-expression **estimates**
* Natural-language subtitles, uncertainty markers, low-reliability handling (neutral sentence / hide / show)
* Video player with synchronised subtitles, timeline, subtitle on/off
* Subtitle editor: edit text/times, add, delete, merge, split, validate, save
* SRT and VTT download, optional embedded-subtitle video (FFmpeg), movement CSV
* History (reopen / delete jobs), settings stored in the database

## 3. Technology stack (and why)
| Technology | Why |
|---|---|
| Python 3.11 | MediaPipe 0.10.14 supports it; simple for students |
| Flask | small, easy-to-read web framework |
| Bootstrap 5 + JavaScript | clean responsive UI without a build step (loaded from a CDN) |
| OpenCV | reads videos and frames; handles timestamps |
| MediaPipe Pose / Hands / Face Mesh | pretrained, runs on CPU, models bundled in the package |
| NumPy, Pandas | vector maths, smoothing, CSV export |
| SQLite + SQLAlchemy | zero-setup database for history, subtitles, settings |
| FFmpeg (optional) | browser-friendly preview copy, burned-in subtitles |
| Pytest | automated tests |

Not used on purpose: PyTorch and scikit-learn (no custom model is trained and no labelled dataset exists, so classification is
rule-based and transparent), React/Node, Docker, paid APIs.

## 4. Project structure
```
DanceSubtitleAI/
├── app.py  config.py  init_database.py  run_project.bat  requirements.txt  .env.example  pytest.ini
├── app/
│   ├── __init__.py  db.py  models.py  routes.py
│   ├── data/gesture_vocabulary.json
│   ├── services/ video_service.py  pose_service.py  backend.py  hand_gesture_service.py
│   │             expression_service.py  movement_service.py  subtitle_service.py
│   │             export_service.py  pipeline.py  settings_service.py  utils.py
│   ├── templates/ base, index, upload, processing, results, editor, visualization, history, settings, error
│   └── static/ css/style.css  js/ main.js video_player.js subtitle_editor.js visualization.js
├── data/ uploads/ processed/ outputs/        (created automatically; videos stay on disk, not in SQLite)
├── models/README.md   tests/   docs/ project_workflow.md  limitations.md
```

## 5–8. Installation and start (Windows, VS Code, PowerShell)
**Easiest:** double-click `run_project.bat` (it checks Python 3.11, creates `.venv` if missing, installs packages, checks
OpenCV/MediaPipe, initialises the database, checks FFmpeg and starts the app).

**Manual commands** (VS Code → Terminal → PowerShell):
```powershell
cd path\to\DanceSubtitleAI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python init_database.py
python app.py
```
Open **http://127.0.0.1:5000** (change `PORT` in `.env` if needed). Stop with `CTRL+C`.

Tested version set (`requirements.txt`): Flask 3.0.3, SQLAlchemy 2.0.35, numpy 1.26.4, pandas 2.2.2,
opencv-contrib-python 4.10.0.84, mediapipe 0.10.14, pytest 8.3.3. Internet is needed once to install packages and
while using the app (Bootstrap is loaded from a CDN).

## 9. Uploading a Bharatanatyam video
1. **Upload video** → choose the file; enter a title; choose *Bharatanatyam* (only changes some wording).
2. Keep sampling at **5** frames/second (try 8–10 for fast footwork; slower to process).
3. Keep *Analyse hand shapes* on; *facial expressions* is optional and slower.
4. **Start processing** — the processing page shows the real stage and sampled-frame count. You are redirected to the results.

Tips for best results: one dancer, full body visible, steady camera, good light, dancer taller than ~1/3 of the frame, clip of 20 s–3 min first.

## 10. Editing subtitles
Results → **Edit subtitles**. Change text or times (`HH:MM:SS.mmm`, `MM:SS.mmm` or seconds). **Split** cuts at the current video time,
**Merge with next** joins neighbours, **Add subtitle at current time**, **Delete**. The preview under the video uses your edits
immediately; press **Save changes** to store them. Overlaps and invalid times are rejected with a message.

## 11. Downloads
Results or editor → **Download SRT / VTT**. Downloads always contain the saved (edited) version. Line length/lines come from Settings.

## 12. Embedded-subtitle video (optional)
Install FFmpeg (a *full* build with libass, e.g. gyan.dev "full"), add its `bin` folder to PATH (or set `FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe` in `.env`),
reopen the terminal and check `ffmpeg -version`. On the results page click **Create video with embedded subtitles**. A new file
is written; your original upload is never modified.

## 13. Model details
Only pretrained MediaPipe models (bundled with `mediapipe==0.10.14`): BlazePose (33 landmarks), Hands (21), Face Mesh (468).
**No custom model was trained; no accuracy figures are claimed.** See `models/README.md`.

## 14. Algorithms
Landmark cleaning and smoothing, torso-normalised joint angles, threshold rules for arms/legs, yaw from shoulder width,
speed in torso lengths/second, majority-vote temporal smoothing, run-length interval grouping, template-based wording and
timing rules. Details with formulas: `docs/project_workflow.md`.

## 15. Testing
```powershell
python -m pytest -v                      # all tests (real-video test is skipped unless configured)
python -m pytest -m "not real_video" -v
$env:DANCE_TEST_VIDEO="C:\path\to\short_dance_clip.mp4"; python -m pytest tests/test_real_video.py -v -s
```
Test kinds: `unit` (pure logic, real OpenCV on a generated video), `mocked` (synthetic landmarks instead of a model; Flask routes use
a fake pose backend), `real_video` (real MediaPipe inference on your own clip).

## 16. Troubleshooting
| Problem | Fix |
|---|---|
| `py -3.11` not found | Install Python 3.11 (python.org), tick *py launcher*; reopen terminal |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd` and `.venv\Scripts\activate.bat` |
| MediaPipe install fails / wrong Python | MediaPipe 0.10.14 needs Python 3.8–3.11 64-bit; recreate `.venv` with 3.11 |
| `module 'mediapipe' has no attribute 'solutions'` | `python -m pip install --force-reinstall mediapipe==0.10.14 numpy==1.26.4` |
| NumPy 2 error | `python -m pip install "numpy==1.26.4"` |
| `ModuleNotFoundError` | The venv is not active: run `.\.venv\Scripts\Activate.ps1` |
| Video does not play in browser | Install FFmpeg; a browser-friendly preview is then created automatically |
| "No person was detected" | Use a clip with a larger, well-lit dancer; raise sampling; try model quality *Accurate* in Settings |
| Port 5000 busy | Set `PORT=5001` in `.env` |
| Page unstyled | Bootstrap CDN needs internet |
| FFmpeg export fails | Use a full FFmpeg build with the `subtitles` filter; check `ffmpeg -filters` |

## 17. Limitations
Summarised: 2-D ambiguity, no mudra meaning or rasa/emotion claims, one tracked dancer, occlusion/fast motion/distance problems,
English only, local single-user use. Full text: `docs/limitations.md`.

## 18. Future enhancements
Evaluated mudra classifier on a licensed dataset, multi-dancer tracking, adavu recognition, music/tala alignment, more languages, annotated video export.

## 19. References
* MediaPipe Pose / Hands / Face Mesh documentation (Google): https://developers.google.com/mediapipe
* Bazarevsky et al., *BlazePose: On-device Real-time Body Pose tracking*, 2020.
* OpenCV documentation: https://docs.opencv.org · Flask: https://flask.palletsprojects.com · SQLAlchemy: https://docs.sqlalchemy.org
* SubRip (SRT) and WebVTT (W3C) subtitle format specifications · FFmpeg `subtitles` filter documentation.

## 20. Demonstration steps for faculty
1. Show the home page and explain "observation, not interpretation". 2. Upload a 30–60 s Bharatanatyam clip.
3. Show the real processing stages. 4. Play the video with subtitles; click the timeline. 5. Open *Visualisation*: skeleton, intensity chart.
6. Edit one subtitle, split another, save, download SRT/VTT and open in VLC. 7. Show History and Settings (threshold changes).
8. Run `python -m pytest -v`. 9. Present `docs/limitations.md` honestly.
