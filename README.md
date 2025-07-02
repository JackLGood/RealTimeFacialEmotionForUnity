# Unity Real-Time Emotion Detection

A Unity based ML model to Webcam pipeline showing real-time facial emotion recognition using the [FER+ DCNN](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx) and Unity’s Barracuda inference engine ([https://github.com/Unity-Technologies/barracuda-release?tab=readme-ov-file#](https://github.com/Unity-Technologies/barracuda-release?tab=readme-ov-file#)). Webcam frames are face-aligned via OpenCV landmarks and fed into the network to detect one of eight emotions.

---

## Features

- **Real-time** webcam emotion inference (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt).  
- **Face alignment** with 68-point landmarks (eyes, nose, mouth, chin) for robust cropping.  
- **Raw 0–255 pixel input** matching the FER+ preprocessing recipe.  
- **Inspector controls** for zoom/padding, smoothing window, and preprocessing mode.  
- **Debug preview** of the exact 64×64 grayscale crop sent to the model.

---

## How It Works

1. **Snapping a Selfie in Code**  
   Every frame we grab the latest webcam image into an OpenCV `Mat`— its basically a quick digital snapshot of whatever your camera sees.
2. **Finding Your Face**  
   A Haar cascade scans the image for face rectangles. If it spots a face, then it knows roughly where you are; if not, it'll wait for the next frame.
3. **Zeroing In with Landmarks**  
   The code passes that face region to the LBF facemark detector, which pinpoints 68 key landmarks (eyebrows, eyes, nose, mouth corners, chin).
4. **Aligning & Squaring Up**  
   Faces tilt and turn, so it'll rotate and scale the grayscale image so your eyes lie perfectly horizontal at a fixed distance. Then it computes the tightest square that contains all landmarks (with a bit of padding).
5. **Shrinking to 64×64**  
   That aligned square is resized to exactly 64×64 pixels—matching the input size the FER+ ONNX model was trained on.
6. **Raw Pixels → Tensor**  
   My code then reads each of those 64×64 gray values directly (0…255), pack them into a float array, and wrap it in a Barracuda `Tensor` shaped `(1×64×64×1)`.
7. **Firing Up the FER+ Network**  
   The tensor is fed into the pre-trained FER+ ONNX model via Unity Barracuda, producing 8 raw scores (logits), one for each emotion class.
8. **Softmax & Smooth**  
   Next it applys softmax to convert logits into percentages, then optionally average over a few frames to avoid jitters when you blink or twitch.
9. **Showtime**  
   The top emotion (and its confidence) is overlaid on the UI—and a second RawImage shows you the exact 64×64 crop the network “sees,” so you can verify it always includes your full face.

---

## Dependencies

- **Unity 2020.3 LTS** or later  
- **Unity Barracuda** package  
- **ONNX Model Zoo** FER+ model
- **OpenCV for Unity** Asset Store plugin (not in this repo)  

---

## Installation
1. **Clone this repo**
2. **Open in Unity**
- Launch Unity Hub and add the project folder.
- Open the EmotionDemo scene.
3. **Import packages**
- In Window → Package Manager, verify Barracuda is installed. You'll have to manually insert the package via the github link.
- Purchase & import “OpenCV for Unity” from the Asset Store; you’ll need it for face‐landmark detection.
4. **Verify StreamingAssets**
- Make sure Assets/StreamingAssets/haarcascade_frontalface_default.xml and lbfmodel.yaml link are present.
5. **Assign ONNX model**
- In the EmotionController in the scene, drag the emotion-ferplus-8.onnx asset into the Model Asset slot.

---

## Acknowledgments
- ONNX Model Zoo for the FER+ model
- CNTK FER+ codebase for training references
- OpenCV for Unity for landmark detection
- Unity’s Barracuda for on-device inference
