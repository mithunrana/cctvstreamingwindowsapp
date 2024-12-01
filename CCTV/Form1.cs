using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using AForge.Video;
using AForge.Video.DirectShow;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
namespace CCTV
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private InferenceSession session;

        public Form1()
        {
            string modelPath = @"C:\Users\SMCIT\Desktop\ML LEARN\best.onnx";
            session = new InferenceSession(modelPath);
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void startButton_Click(object sender, EventArgs e)
        {
            string rtspUrl = "rtsp://admin:admin@192.168.1.15:554/stream0"; // Update with your RTSP URL

            capture = new VideoCapture(rtspUrl);
            if (!capture.IsOpened())
            {
                MessageBox.Show("Failed to open RTSP stream");
                return;
            }

            // Read frames in a loop
            Timer timer = new Timer();
            timer.Interval = 1000 / 30; // 30 fps
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private Bitmap ResizeBitmap(Bitmap bitmap, int targetWidth, int targetHeight)
        {
            // Resize Bitmap using System.Drawing.Size
            var resizedBitmap = new Bitmap(bitmap, new System.Drawing.Size(targetWidth, targetHeight));  // Fully qualified Size
            return resizedBitmap;
        }

        private Mat BitmapToMat(Bitmap bitmap)
        {
            // Convert Bitmap to Mat (OpenCvSharp)
            using (var ms = new System.IO.MemoryStream())
            {
                bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                ms.Seek(0, System.IO.SeekOrigin.Begin);
                return Cv2.ImDecode(ms.ToArray(), ImreadModes.Color); // Decodes image as Mat (in color)
            }
        }


        private Bitmap MatToBitmap(Mat mat)
        {
            if (mat.Empty()) return null;

            // Convert Mat to Bitmap
            Bitmap bitmap;
            using (var ms = new System.IO.MemoryStream(mat.ToBytes()))
            {
                bitmap = new Bitmap(ms);
            }
            return bitmap;
        }



        private async void Timer_Tick(object sender, EventArgs e)
        {
            if (capture.IsOpened())
            {
                // Read a frame
                Mat frame = new Mat();
                capture.Read(frame);

                if (!frame.Empty())
                {
                    // Run inference in a background task
                    var result = await Task.Run(() =>
                    {
                        // Convert Mat to Bitmap
                        Bitmap bitmap = MatToBitmap(frame);

                        // Resize the Bitmap (assuming targetWidth and targetHeight are 640)
                        var resizedBitmapResult = ResizeBitmap(bitmap, 640, 640);

                        // Run inference on the resized frame
                        var detectionsResult = DetectObjects(resizedBitmapResult);

                        return new { ResizedBitmap = resizedBitmapResult, Detections = detectionsResult };
                    });

                    // Get the result from background task
                    var detections = result.Detections;
                    var resizedBitmap = result.ResizedBitmap;

                    // Draw detections (bounding boxes and labels)
                    DrawDetections(resizedBitmap, detections);

                    // Update the PictureBox with the processed frame
                    pictureBox1.Image = resizedBitmap;
                }
            }
        }




        private float[][] DetectObjects(Bitmap frame)
        {
            // Convert Bitmap to Mat
            Mat matFrame = BitmapToMat(frame);

            // Preprocess frame to fit model input requirements
            var inputTensor = PreprocessImage(matFrame);

            // Run inference
            var inputName = session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor(inputName, inputTensor)};

            using (var results = session.Run(inputs))
            {
                // Get model outputs (adjust based on your ONNX model's output)
                var outputName = session.OutputMetadata.Keys.First();
                var output = results.FirstOrDefault(x => x.Name == outputName);
                var detections = output.AsEnumerable<float>().ToArray();
                return ParseDetections(detections);
            }
        }



        private Tensor<float> PreprocessImage(Mat frame)
        {
            const int targetWidth = 640;
            const int targetHeight = 640;

            // Resize the frame to the target size using OpenCV
            Cv2.Resize(frame, frame, new OpenCvSharp.Size(targetWidth, targetHeight));

            // Convert the Mat to a tensor
            var inputData = new float[1 * 3 * targetWidth * targetHeight];
            int idx = 0;

            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    var pixel = frame.Get<Vec3b>(y, x); // Get pixel at (x, y)
                    inputData[idx++] = pixel.Item2 / 255.0f; // Normalize Green
                    inputData[idx++] = pixel.Item1 / 255.0f; // Normalize Red
                    inputData[idx++] = pixel.Item0 / 255.0f; // Normalize Blue
                }
            }

            // Return the processed tensor
            return new DenseTensor<float>(inputData, new[] { 1, 3, targetHeight, targetWidth });
        }




        private float[][] ParseDetections(float[] detections)
        {
            var parsedDetections = new List<float[]>();

            for (int i = 0; i < detections.Length; i += 6)
            {
                var score = detections[i + 4]; // Confidence score
                if (score > 0.5) // Confidence threshold
                {
                    var bbox = new float[]
                    {
                        detections[i], detections[i + 1], detections[i + 2], detections[i + 3] // x1, y1, x2, y2
                    };
                    parsedDetections.Add(bbox);
                }
            }

            return parsedDetections.ToArray();
        }


        private void DrawDetections(Bitmap frame, float[][] detections)
        {
            using (Graphics g = Graphics.FromImage(frame))
            {
                foreach (var detection in detections)
                {
                    var x1 = detection[0] * frame.Width; // Scale by image width
                    var y1 = detection[1] * frame.Height; // Scale by image height
                    var x2 = detection[2] * frame.Width;
                    var y2 = detection[3] * frame.Height;

                    // Draw bounding box
                    g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);

                    // Draw label
                    g.DrawString("License Plate", new Font("Arial", 12), Brushes.Yellow, new PointF(x1, y1 - 20));
                }
            }

            // Update the PictureBox with annotated frame
            pictureBox1.Image = frame;
        }


        private void StopButton_Click(object sender, EventArgs e)
        {
            if (capture != null && capture.IsOpened())
            {
                capture.Release();
            }
        }

    }
}
