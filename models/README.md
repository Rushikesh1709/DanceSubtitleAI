# models/

No model files are stored here and none need to be downloaded.

DanceSubtitleAI uses **pretrained MediaPipe models that are bundled inside the `mediapipe==0.10.14` Python package**:

| Task | MediaPipe component | Used for |
|------|--------------------|----------|
| Body pose (33 landmarks) | `mp.solutions.pose` (BlazePose, lite/full/heavy) | skeleton, joint angles, movement |
| Hand landmarks (21 points) | `mp.solutions.hands` | which fingers are extended |
| Face landmarks (468 points) | `mp.solutions.face_mesh` | smile / eyebrow / mouth geometry |

No custom model was trained by this project, so there are no weights, training scripts or accuracy figures.
The folder exists so that you can later add your own trained classifier (e.g. a mudra classifier) together with its
training script and evaluation report.
