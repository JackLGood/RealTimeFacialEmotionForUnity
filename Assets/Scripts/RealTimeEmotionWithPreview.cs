using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;      // ← for Utils
using OpenCVForUnity.FaceModule;

public class RealTimeEmotionWithPreview : MonoBehaviour {
    public NNModel modelAsset;
    public int inputWidth = 64, inputHeight = 64;
    public RawImage previewImage;
    [Header("Debug Preview")]
    public RawImage processedImage;     // drag your new RawImage here

    Texture2D processedTexture;         // runtime buffer
    public Text emotionText;

    WebCamTexture webcam;
    CascadeClassifier faceCascade;
    IWorker worker;
    Queue<float[]> smoothQueue = new Queue<float[]>();
    private int smoothWindow = 1;
    private readonly string[] labels = {
    "Neutral",
    "Happy",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt"
    };
    private Facemark facemark;

    public enum PreprocMode { RawZeroOne, GlobalContrast, GCNplusLCN }

    [Header("Preprocessing Mode")]
    public PreprocMode preproc = PreprocMode.GlobalContrast;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);

        // Log each input’s name and shape
        foreach (var inp in ModelLoader.Load(modelAsset).inputs)
            Debug.Log($"[MODEL] Input expects [B,H,W,C] = [{inp.shape[0]}, {inp.shape[1]}, {inp.shape[2]}, {inp.shape[3]}]");

        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);

        // 1) Create & start the camera
        webcam = new WebCamTexture();
        webcam.Play();

        // 2) Hook it up to your UI preview (if you have one)
        if (previewImage != null)
        {
            previewImage.texture = webcam;
        }
        else
        {
            Debug.LogError("PreviewImage RawImage reference is missing in the Inspector!");
        }

        // 3) Load your face cascade
        string cascadePath = Path.Combine(Application.streamingAssetsPath,
                                        "haarcascade_frontalface_default.xml");
        faceCascade = new CascadeClassifier(cascadePath);
        if (faceCascade.empty())
            Debug.LogError($"Failed to load cascade at {cascadePath}");

        // Create a Texture2D to hold the preprocessed face (gray single-channel)
        processedTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        // Assign it to your RawImage so it renders every frame
        if (processedImage != null)
            processedImage.texture = processedTexture;
        else
            Debug.LogWarning("ProcessedImage RawImage not assigned!");

        // Load the LBF landmarks model
        string lbfPath = Path.Combine(Application.streamingAssetsPath, "lbfmodel.yaml");
        var fm = Face.createFacemarkLBF();
        fm.loadModel(lbfPath);

        // Store it to a field so Update() can use it
        this.facemark = fm;  
    }

    void Update()
    {
        if (!webcam.didUpdateThisFrame) return;

        // 1. Grab frame into an RGBA Mat
        Mat rgba = new Mat(webcam.height, webcam.width, CvType.CV_8UC4);
        Utils.webCamTextureToMat(webcam, rgba);

        // 2. Convert to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);

        // 3. Face detection
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(
            gray, faces,
            1.1, 2, 0,
            new Size(30, 30),   // minFace size
            new Size()          // maxFace size (no limit)
        );
        OpenCVForUnity.CoreModule.Rect[] faceArray = faces.toArray();
        if (faceArray.Length == 0)
        {
            emotionText.text = "No face detected";
            return;
        }
        var faceRect = faceArray.OrderBy(r => r.width * r.height).Last();

        // 4. Align face using landmarks (FER+ style)
        // 4.1 Fit LBF landmarks to the whole gray frame
        List<MatOfPoint2f> allLandmarks = new List<MatOfPoint2f>();
        facemark.fit(gray, faces, allLandmarks);
        if (allLandmarks.Count == 0)
        {
            emotionText.text = "No landmarks";
            return;
        }
        Point[] pts = allLandmarks[0].toArray();

        // 4.2 Compute eye centers (points 36–41 & 42–47)
        Point leftEye = new Point(), rightEye = new Point();
        for (int i = 36; i <= 41; i++)
        {
            leftEye.x += pts[i].x;
            leftEye.y += pts[i].y;
        }
        for (int i = 42; i <= 47; i++)
        {
            rightEye.x += pts[i].x;
            rightEye.y += pts[i].y;
        }
        leftEye.x /= 6; leftEye.y /= 6;
        rightEye.x /= 6; rightEye.y /= 6;

        // 4.3 Compute rotation angle & scale factor
        double dy = rightEye.y - leftEye.y;
        double dx = rightEye.x - leftEye.x;
        double angleDeg = System.Math.Atan2(dy, dx) * 180.0 / System.Math.PI;
        double eyeDist = System.Math.Sqrt(dx * dx + dy * dy);
        double desired = inputWidth * 0.5;      // you want eyes half‐width apart
        double scale = desired / eyeDist;

        // 4.4 Build and apply the rotation+scale transform
        Point eyesCenter = new Point((leftEye.x + rightEye.x) / 2,
                                    (leftEye.y + rightEye.y) / 2);
        Mat rotMat = Imgproc.getRotationMatrix2D(eyesCenter, angleDeg, scale);
        Mat aligned = new Mat();
        Imgproc.warpAffine(
            gray, aligned,
            rotMat,
            new Size(gray.cols(), gray.rows()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT,
            new Scalar(0)
        );

        Point mouthCenter = new Point(0, 0);
        for (int i = 48; i <= 59; i++)
        {
            mouthCenter.x += pts[i].x;
            mouthCenter.y += pts[i].y;
        }
        mouthCenter.x /= 12;
        mouthCenter.y /= 12;

        // 2) Combined center (eyes + mouth)
        double centerX = (eyesCenter.x + mouthCenter.x) * 0.5;
        double centerY = (eyesCenter.y + mouthCenter.y) * 0.5;

        // 3) Square crop of size inputWidth around (centerX, centerY)
        int side = (int)(inputWidth * 1.2f);
        int x0_s = Mathf.Clamp((int)(centerX - side / 2), 0, aligned.cols() - side);
        int y0_s = Mathf.Clamp((int)(centerY - side / 2), 0, aligned.rows() - side);
        var square = new OpenCVForUnity.CoreModule.Rect(x0_s, y0_s, side, side);

        // 4.6 Extract, equalize, and resize
        Mat faceMat = new Mat(aligned, square);
        // Imgproc.equalizeHist(faceMat, faceMat);
        Imgproc.resize(
            faceMat, faceMat,
            new Size(inputWidth, inputHeight),
            0, 0,
            Imgproc.INTER_LINEAR
        );

        int N = inputWidth * inputHeight;
        float[] data = new float[N];
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                // NO /255f: feed raw 0–255 values
                data[y * inputWidth + x] = (float)faceMat.get(y, x)[0];
            }
        }

        // // Copy the OpenCV Mat (64×64 gray) into our Texture2D
        Utils.matToTexture2D(faceMat, processedTexture);


        // // --- Flexible Preprocessing ---
        // int N = inputWidth * inputHeight;
        // float[] data = new float[N];

        // // 0) Always start with a 0→1 float array
        // for (int y = 0; y < inputHeight; y++)
        //     for (int x = 0; x < inputWidth; x++)
        //         data[y * inputWidth + x] = (float)faceMat.get(y, x)[0] / 255f;

        // switch (preproc)
        // {
        //     case PreprocMode.RawZeroOne:
        //         // nothing more to do
        //         break;

        //     case PreprocMode.GlobalContrast:
        //     {
        //         // compute mean & L2 norm on data[]
        //         double sum = 0, sumSq = 0;
        //         for (int i = 0; i < N; i++)
        //         {
        //             sum   += data[i];
        //             sumSq += data[i] * data[i];
        //         }
        //         double mean  = sum / N;
        //         double denom = System.Math.Sqrt(sumSq - N * mean * mean) + 1e-6;
        //         for (int i = 0; i < N; i++)
        //             data[i] = (float)((data[i] - mean) / denom);
        //         break;
        //     }

        //     case PreprocMode.GCNplusLCN:
        //     {
        //         // first do GCN
        //         double sum = 0, sumSq = 0;
        //         for (int i = 0; i < N; i++)
        //         {
        //             sum   += data[i];
        //             sumSq += data[i] * data[i];
        //         }
        //         double mean  = sum / N;
        //         double denom = System.Math.Sqrt(sumSq - N * mean * mean) + 1e-6;
        //         for (int i = 0; i < N; i++)
        //             data[i] = (float)((data[i] - mean) / denom);

        //         // then approximate LCN via a Laplacian+normalize
        //         // rebuild faceMat from data[]
        //         Mat procMat = new Mat(inputHeight, inputWidth, CvType.CV_32F);
        //         procMat.put(0, 0, data);
        //         Mat lcn = new Mat();
        //         Imgproc.Laplacian(procMat, lcn, CvType.CV_32F);
        //         Core.normalize(lcn, procMat, 0, 1, Core.NORM_MINMAX, CvType.CV_32F);
        //         procMat.get(0, 0, data);  // write back into data[]

        //         break;
        //     }
        // }

        // -- Inference
        using var input = new Tensor(1, inputHeight, inputWidth, 1, data);

        // Debug for model
        Debug.Log($"[TENSOR] shape = {input.shape}");
        // log first few values
        string sample = "";
        for (int i = 0; i < Mathf.Min(10, data.Length); i++)
            sample += data[i].ToString("F3") + (i < 9 ? "," : "");
        Debug.Log($"[TENSOR] data[0..9] = {sample}");
        // log min/max
        float min = data.Min(), max = data.Max();
        Debug.Log($"[TENSOR] data range = {min:F3} … {max:F3}");

        worker.Execute(input);
        float[] raw = worker.PeekOutput().AsFloats();
        input.Dispose();

        // -- Softmax
        float maxLogit = raw.Max();
        float sumExp = raw.Select(r => Mathf.Exp(r - maxLogit)).Sum();
        float[] probs = raw.Select(r => Mathf.Exp(r - maxLogit) / sumExp).ToArray();

        for (int i = 0; i < probs.Length; i++)
            Debug.Log($"[RAW PROB] class {i} = {probs[i] * 100f:F1}%");

        // -- Smoothing
        smoothQueue.Enqueue(probs);
        if (smoothQueue.Count > smoothWindow)
            smoothQueue.Dequeue();
        float[] avg = new float[probs.Length];
        foreach (var p in smoothQueue)
            for (int i = 0; i < p.Length; i++)
                avg[i] += p[i];
        for (int i = 0; i < avg.Length; i++)
            avg[i] /= smoothQueue.Count;

        // -- new UI update: show all emotion percentages
        var sb = new System.Text.StringBuilder();

        // highlight the top emotion
        int best = avg
            .Select((v, i) => (v, i))
            .OrderByDescending(t => t.v)
            .First().i;
        sb.AppendLine($"► {labels[best]}: {avg[best] * 100f:F1}% ◄\n");

        for (int i = 0; i < labels.Length; i++)
        {
            sb.AppendLine($"{labels[i]}: {avg[i] * 100f:F1}%");
        }

        emotionText.text = sb.ToString();

    }

    void OnDestroy() {
        webcam?.Stop();
        worker?.Dispose();
    }
}