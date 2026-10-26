# AI-Based Automatic Subtitle Generation for Indian Classical Dance Videos Using Computer Vision

## 1. Abstract
Indian classical dance conveys information through body posture, footwork, hand gestures and facial expression. People who
cannot see the performance well, new learners, and archivists benefit from a written description of what happens on stage.
This project builds a web application that takes a dance video, detects the dancer's body (and optionally hands and face) with
pretrained MediaPipe models, measures joint angles and movement over time, groups similar frames into time intervals and writes
**descriptive, time-synchronised subtitles** such as "The dancer raises both arms while standing upright." The subtitles can be
previewed, edited and exported as SRT/WebVTT. The system describes only observable facts; it does not claim to decode mudra
meanings, stories or emotions (rasa).

## 2. Introduction
Subtitles are normally speech transcripts. A dance has no speech, so text must come from **visual analysis**. Computer vision can
find body joints in each frame ("pose estimation"). By following those joints through time we can say whether an arm is raised,
whether knees are bent, whether the dancer turns or holds a posture, and then turn these facts into sentences.

## 3. Problem statement
Given a video of an Indian classical dance performance, automatically produce readable, correctly timed subtitles that describe
the visible movements, while clearly separating *observation* from *interpretation*, and let the user correct and download them.

## 4. Motivation
* Accessibility for blind/low-vision audiences (audio-description style text).
* Learning aid: students can read what a movement consists of.
* Cataloguing and search of dance archives.
* A realistic, honest example of applying pretrained vision models to cultural data.

## 5. Objectives
1. Accept common video formats and validate them. 2. Detect pose (and optionally hands/face) with pretrained models.
3. Compute movement features and stable movement labels. 4. Generate natural-language, timestamped subtitles.
5. Provide a video player with synchronised subtitles, an editor, SRT/VTT export and optional burned-in export.
6. Store history in a database. 7. Document limitations honestly and test the software.

## 6. Existing system
* Speech-to-text subtitle tools (useless for silent dance). * Manual audio description written by experts (slow, costly).
* Research systems for action recognition (e.g. trained on sports or daily activities) that output generic labels like "dancing".

## 7. Limitations of existing methods
Generic labels do not describe *which* limb moves; speech tools produce nothing; manual work does not scale; trained dance-specific
models need large labelled datasets that are not freely available for the eight classical forms.

## 8. Proposed system
A rule-based, transparent pipeline on top of pretrained landmark detectors:
pose estimation -> landmark cleaning/smoothing -> joint-angle features -> per-frame state rules -> temporal smoothing ->
interval segmentation -> template-based sentence generation -> subtitle timing rules -> editing/export.
No custom neural network is trained, so no training accuracy is claimed.

## 9. System architecture
```
 Browser (Bootstrap + JS)  <---->  Flask routes  <---->  SQLite (SQLAlchemy)
                                      |
                               Background worker thread
                                      |
   video_service -> backend (pose / hands / face) -> movement_service -> subtitle_service -> export_service
```
* `app/routes.py` - pages and JSON API. * `app/services/pipeline.py` - orchestrates a job and records real status.
* `app/models.py` - tables `processing_jobs`, `subtitle_segments`, `analysis_events`, `application_settings`.

## 10. Data flow diagram
```
Upload video --> validate (OpenCV) --> job saved (status: queued)
   --> sample frames (keeping original timestamps)
   --> Pose landmarks (33 points) [+ hand landmarks] [+ face metrics]
   --> clean & smooth landmarks --> joint angles / elevations / speed
   --> per-frame states (arm, leg) --> majority-vote smoothing
   --> group into intervals --> describe each interval in English
   --> merge/split/time subtitles --> save segments + analysis.json
   --> results page (player, timeline, charts) --> edit --> SRT / VTT / embedded video
```

## 11. Algorithms used
* **Pose estimation (MediaPipe Pose / BlazePose):** a pretrained CNN that returns 33 body landmarks with a visibility score.
* **Landmark cleaning:** landmarks below visibility 0.5 are dropped; gaps of up to 2 samples are interpolated; a 3-sample
  median filter reduces jitter; longer gaps stay empty.
* **Normalisation:** x is multiplied by width/height so angles are not distorted; distances are divided by the torso length so
  results do not depend on resolution or zoom.
* **Joint angle:** angle at the elbow/knee from three points using the dot product, `cos(theta) = (u.v)/(|u||v|)`.
* **State rules:** e.g. wrist more than 0.2 torso lengths above the shoulder = "raised"; straight elbow + wrist far to the side
  = "out"; knee angle < 150 degrees = bent; ankle separation > 0.9 torso lengths = wide stance.
* **Body orientation (yaw):** shoulder width relative to its video-wide 90th percentile, `yaw = arccos(width / reference)`.
  A change of more than 30 degrees within about 1.2 s = turning.
* **Movement speed:** mean speed of wrists, ankles and hip centre in torso lengths per second; classified as still / slow /
  moderate / fast using thresholds divided by the sensitivity setting.
* **Temporal smoothing:** sliding-window majority vote (about 1 s) on labels; very short intervals are absorbed by neighbours.
* **Repetition:** alternating A-B-A-B-A posture intervals are flagged as a repeated pattern.
* **Hand shape:** a finger is "extended" when its tip is clearly farther from the wrist than its middle joint; the pattern of
  extended fingers gives a shape name. **Expression:** mouth/eyebrow geometry compared with the dancer's own median.
* **Subtitle timing:** merge equal texts, enforce minimum/maximum duration, enforce minimum gaps, wrap lines (Unicode safe).

## 12. Modules
| Module | Purpose |
|---|---|
| `video_service.py` | validation, metadata, timestamp-preserving frame sampling |
| `pose_service.py` | MediaPipe Pose wrapper, landmark indices |
| `hand_gesture_service.py` | hand-shape rules, gesture vocabulary, hand events |
| `expression_service.py` | optional facial-expression estimates |
| `movement_service.py` | features, states, smoothing, intervals, events |
| `subtitle_service.py` | wording, timing rules, validation, merge/split, SRT/VTT |
| `export_service.py` | FFmpeg preview copy and burned-in subtitles |
| `pipeline.py` | background job, real status updates, persistence |
| `settings_service.py` | validation and storage of settings |

## 13. Hardware requirements
Windows 10/11 64-bit PC, 8 GB RAM (16 GB recommended), any modern CPU (no GPU needed), 2 GB free disk space plus space for videos.
Analysis speed on a typical laptop CPU is roughly real time or slower at 5 samples/s, depending on resolution and options.

## 14. Software requirements
Python 3.11, Flask 3.0.3, SQLAlchemy 2.0.35, OpenCV (contrib) 4.10, MediaPipe 0.10.14, NumPy 1.26.4, Pandas 2.2.2, Pytest 8.3.3,
FFmpeg (optional), a modern browser (Chrome/Edge/Firefox), VS Code + PowerShell.

## 15. Implementation methodology
Iterative: video I/O first, then pose analysis with mocked landmarks (so logic is testable without a model), then wording and
timing rules, then web interface, then tests and documentation. Each rule is a small, documented function with unit tests.

## 16. Expected output
For each video: a list of subtitle segments with start/end times, text, category and a reliability score; SRT and VTT files; an
intensity chart; skeleton overlay; a CSV of per-frame movement features; optionally a video with burned-in subtitles.

## 17. Advantages
Transparent rules (every sentence can be traced to measurements); no training data or GPU needed; runs offline on a PC;
honest about uncertainty; editable output; modular for new dance forms.

## 18. Limitations
See `limitations.md`: ambiguity of 2-D views, no mudra meaning, no emotion/rasa claim, occlusion, single-dancer tracking,
no automatic dance-form recognition, English only.

## 19. Applications
Accessibility descriptions, teaching material, archive indexing, research on movement statistics, preparing captions for review by experts.

## 20. Future scope
Train and evaluate a mudra classifier on a licensed labelled dataset; multi-person tracking; adavu/footwork recognition using
sequence models; audio (music/tala) alignment; more languages; expert-in-the-loop vocabulary editing; annotated video export.

## 21. Conclusion
The project shows that useful, time-aligned, human-readable descriptions of dance movement can be produced from pretrained pose
models and transparent rules, while keeping a clear boundary between observation and cultural interpretation.
