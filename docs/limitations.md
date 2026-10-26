# Limitations

This project **describes what can be seen** (arm positions, knee bend, turning, held postures, simple hand shapes).
It deliberately does **not** claim to understand the art form. The reasons are listed here.

## 1. Visual movements are ambiguous
* A single 2-D camera view hides depth. An arm pointing at the camera looks short; two different poses can look alike.
* "Left" and "right" in the subtitles mean the **dancer's own** left/right as reported by the pose model. If the video is
  mirrored, or the dancer faces away from the camera, labels can be swapped.
* Thresholds (e.g. "wrist clearly above shoulder") are fixed rules. Unusual body proportions, costumes or camera angles can
  push a pose over a threshold and change the wording.
* Speed is measured relative to the dancer's torso length. Camera panning/zooming looks like dancer movement.

## 2. Mudra meanings need cultural context
* The hand module only measures **which fingers are extended** and reports a *hand shape* ("a flat hand with the thumb tucked in").
* A mudra is a symbolic gesture whose meaning depends on the tradition, the sequence, the lyrics/music and the dancer's intent.
  The same hand shape can belong to different mudras, and a mudra's meaning changes with context.
* `app/data/gesture_vocabulary.json` lists a few *possible* resemblances. They are **off by default**, are shown as
  "possible resemblance ... not verified", and are not evaluated against any labelled dataset. No dataset of mudras was
  used, so no accuracy is reported.
* Hands are small in most dance videos; MediaPipe Hands often fails on distant, blurred or self-occluded hands.

## 3. Facial expression does not prove emotion
* The expression module measures geometry (mouth corners, mouth width, eyebrow height) **relative to the same dancer's median
  face in the same video**. "neutral" therefore means "typical for this video".
* Stylised *abhinaya* expressions are deliberate, exaggerated performances - they are not spontaneous emotions. A smile
  shape can be part of a *rasa* or just part of a pose.
* **Rasa is not detected.** Deciding a rasa needs context (gesture sequence, music, narrative, tradition) that this system
  does not have, and no evaluated model exists in this project. Only visible expression estimates are given, labelled as estimates.
* Faces smaller than about 80 px, profile views and heavy make-up/ornaments can make the estimates unreliable; such frames are skipped.

## 4. Pose estimation can fail
* **Occlusion** (skirt/costume folds, arms crossing the body, other dancers) causes wrong or missing landmarks.
* **Fast movement** blurs the frame; at low sampling rates (e.g. 5 samples/s) quick moves are under-sampled.
* **Distant dancers** produce small, noisy skeletons. Frames where the torso is too small are ignored.
* **Multiple performers:** MediaPipe Pose tracks one person. In group dances it may jump between dancers; the subtitles then
  describe whichever person was detected. Multi-dancer support is not implemented.
* **Poor lighting, backlight and busy backgrounds** reduce detection. Low-visibility landmarks (< 0.5) are discarded and
  long detection gaps are reported as "not clearly visible" instead of being guessed.
* Legs hidden by the frame edge: the torso length is estimated from shoulder width, which is less accurate; reliability drops.

## 5. Recognising each classical dance form needs data
* The dance form is **chosen by the user**; it is never recognised automatically and only affects a few words.
* Reliable recognition of Bharatanatyam, Kathak, Kuchipudi, Odissi, Kathakali, Manipuri, Mohiniyattam and Sattriya would need a
  labelled, licensed video dataset, a trained model, and evaluation on held-out performers. None of this is included.
  Any accuracy figure without such evaluation would be invented, so none is given.
* Movements specific to a style (footwork patterns / *adavus*, spins, eye movements, neck movements) are **not** recognised.

## 6. Other technical limits
* "Reliability" is a consistency/visibility score (0-1). It is **not** a probability that a subtitle is correct.
* English subtitles only. Subtitle text is template based, so wording is repetitive for long, uniform sequences.
* Browser playback depends on the codec; MOV/AVI/MKV/HEVC files are converted to an H.264 preview if FFmpeg is installed.
* A background job is lost if the application is closed during processing (it is then marked as failed).
* The app is intended for local single-user use (no login). Do not expose it to the internet as it is.
