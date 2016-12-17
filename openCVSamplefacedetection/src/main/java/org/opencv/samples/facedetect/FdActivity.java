package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;


    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        loadYukari();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    private long startTime = System.currentTimeMillis();
    enum State { ORIGNAL, BLUR, FILTER, CANNY, RECT, EYE, EYES }
    private State state = State.ORIGNAL;
    private static final int step = 3; // sec

    private void updateState() {
        long currentTime = System.currentTimeMillis();
        double duration = (currentTime - startTime) / 1000.0; // sec
        if (duration < step * 0.5) {
            this.state = State.ORIGNAL;
        } else if (duration < step * 1) {
            this.state = State.BLUR;
        } else if (duration < step * 2) {
            this.state = State.FILTER;
        } else if (duration < step * 3) {
            this.state = State.CANNY;
        } else if (duration < step * 4) {
            this.state = State.RECT;
        } else if (duration < step * 5) {
            this.state = State.EYE;
        } else if (duration < step * 80) {
            this.state = State.EYES;
        } else {
            startTime = currentTime;
        }
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        updateState();
        if (debug && this.state == State.ORIGNAL) {
            Imgproc.putText(mRgba, this.state.toString(), new Point(140, 140), Core.FONT_HERSHEY_SIMPLEX, 5f, eyeColor);
            return mRgba;
        }

        Mat blurImage = new Mat();
        Imgproc.GaussianBlur(mGray, blurImage, new Size(7, 7), 0, 0);
        if (debug && this.state == State.BLUR) {
            Imgproc.putText(blurImage, this.state.toString(), new Point(140, 140), Core.FONT_HERSHEY_SIMPLEX, 5f, eyeColor);
            return blurImage;
        }

        Mat thresholdImage = new Mat();
        double highThreshold = Imgproc.threshold(blurImage, thresholdImage, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        double lowThreshold = 0.5 * highThreshold;
        if (debug && this.state == State.FILTER) {
            Imgproc.putText(thresholdImage, this.state.toString(), new Point(140, 140), Core.FONT_HERSHEY_SIMPLEX, 5f, eyeColor);
            return thresholdImage;
        }

        Mat cannyImage = new Mat();
        Imgproc.Canny(thresholdImage, cannyImage, lowThreshold, highThreshold);
        Log.d("Canny", "threshold low, high = " + lowThreshold + ", " + highThreshold);
        if (debug && this.state == State.CANNY) {
            Imgproc.putText(cannyImage, this.state.toString(), new Point(140, 140), Core.FONT_HERSHEY_SIMPLEX, 5f, eyeColor);
            return cannyImage;
        }

        this.findContours(cannyImage);

        mRgba = this.drawEyes(mGray, mRgba);
        Imgproc.putText(mRgba, this.state.toString(), new Point(140, 140), Core.FONT_HERSHEY_SIMPLEX, 5f, eyeColor);
        return this.drawRect(mRgba); // debug
    }

    private static final boolean debug = true;
    private static final double longAspect = 23/6;
    private static final double shortAspect = 17/6;
    private static final double minWidth = 10;
    private static final double minHeight = shortAspect * minWidth;
    private static final double maxWidth = 40;
    private static final double maxHeight = longAspect * maxWidth;
    private static final Scalar eyeColor = new Scalar(255, 0, 0);

    private Mat yukari;

    private void loadYukari() {
//        try {
            // yukari = Utils.loadResource(this, R.drawable.yukari, CvType.CV_8UC4);
            Bitmap bmp = BitmapFactory.decodeResource(getResources(), R.drawable.yukari);
            yukari = new Mat();
            Utils.bitmapToMat(bmp, yukari);
            Log.d("imread", "yukari.size = " + yukari.size() + " channels = " + yukari.channels());
//        } catch (IOException e) {
//            e.printStackTrace();
//            Log.d("imread", "error");
//        }
    }


    /**
     *  RGBA画像からAlphaを削除してRGB画像にする
     * @param img_rgba
     * @return
     */
    private Mat removeAlpha(Mat img_rgba) {
        List<Mat> planes_rgba = new ArrayList<>();
        for (int i = 0; i< 4; i++){
            planes_rgba.add(new Mat());
        }
        Core.split(img_rgba, planes_rgba);

        List<Mat> planes_rgb = new ArrayList<>();
        planes_rgb.add(planes_rgba.get(0));
        planes_rgb.add(planes_rgba.get(1));
        planes_rgb.add(planes_rgba.get(2));
        Mat img_rgb = new Mat();
        Core.merge(planes_rgb, img_rgb);
        return img_rgb;
    }

    /***
     *
     * @param backgroundImage
     * @param foregroundImage
     * @return
     */
    private Mat overlayImage(Mat backgroundImage, Mat foregroundImage) {
        assert(backgroundImage.channels() == foregroundImage.channels());
        assert(backgroundImage.size() == foregroundImage.size());

        int maxVal = 255; //(int)(Math.pow(2, 8 * backgroundImage.elemSize1()) - 1);

        // チャンネルに分解
        List<Mat> planes_rgba = new ArrayList<>();
        Core.split(foregroundImage, planes_rgba);

        //RGBA画像をRGBに変換
        Mat img_rgb = new Mat();
        Core.merge(Arrays.asList(planes_rgba.get(0), planes_rgba.get(1), planes_rgba.get(2)), img_rgb);

        //RGBA画像からアルファチャンネル抽出
        Mat img_aaa = new Mat();
        Core.merge(Arrays.asList(planes_rgba.get(3), planes_rgba.get(3), planes_rgba.get(3)), img_aaa);

        //背景用アルファチャンネル
        Mat negpos = new Mat();
        Core.bitwise_not(planes_rgba.get(3), negpos);
        Mat img_1ma = new Mat();
        Core.merge(Arrays.asList(negpos, negpos, negpos), img_1ma);


        Mat img_dst1 = new Mat();
        Mat img_dst2 = new Mat();
        Mat img_dst3 = new Mat();
        Core.multiply(img_rgb, img_aaa, img_dst1, 1.0 / maxVal);
        Core.multiply(backgroundImage, img_1ma, img_dst2, 1.0 / maxVal);
        Core.add(img_dst1, img_dst2, img_dst3);
        return img_dst3;
    }

    /**
     *  縁ちゃん画像の目の位置を合わせるためのアフィン変換行列を計算
     * @param foregroundImage
     * @param rr1
     * @param rr2
     * @return
     */
    private Mat computeAffineTransform(Mat foregroundImage, RotatedRect rr1, RotatedRect rr2) {
        Point [] srcTri = new Point[]{
                new Point(0.0f, 0.0f),
                new Point(foregroundImage.width(), 0.0f),
                new Point(foregroundImage.width(), foregroundImage.height()),
        };

        Point points1[] = new Point[4];
        rr1.points(points1);

        Point points2[] = new Point[4];
        rr2.points(points2);

        Point tl = points1[0];
        Point br = points1[0];
        double tld = Math.sqrt(Math.pow(tl.x, 2) + Math.pow(tl.y, 2));
        double brd = Math.sqrt(Math.pow(br.x, 2) + Math.pow(br.y, 2));
        for (Point point : points1) {
            double d = Math.sqrt(Math.pow(point.x, 2) + Math.pow(point.y, 2));
            if (d <= tld){
                tl = point;
                tld = d;
            }
            if (d >= brd) {
                br = point;
                brd = d;
            }
        }
        for (Point point : points2) {
            double d = Math.sqrt(Math.pow(point.x, 2) + Math.pow(point.y, 2));
            if (d <= tld){
                tl = point;
                tld = d;
            }
            if (d >= brd) {
                br = point;
                brd = d;
            }
        }
        double width = br.x - tl.x;
        double height = br.y - tl.y;

        Point [] dstTri = new Point[] {
                new Point(tl.x - 1.5 * width, tl.y - 2.3 * height),
                new Point(br.x + 1.8 * width, tl.y - 2.3 * height),
                new Point(br.x + 1.8 * width, br.y + 3.4 * height),
        };

        // 変形行列を作成
        Mat affineTrans = Imgproc.getAffineTransform(new MatOfPoint2f(srcTri), new MatOfPoint2f(dstTri));

        return affineTrans;
    }

    /**
     * 縁ちゃんを描く
     * @param frame
     * @return
     */
    private Mat drawYukari(Mat frame) {
        Mat backgroundImage = removeAlpha(frame);

        for (List<RotatedRect> eyePair : eyes) {
            RotatedRect rr1 = eyePair.get(0);
            RotatedRect rr2 = eyePair.get(1);

            Mat affineTrans = computeAffineTransform(yukari, rr1, rr2);

            // 出力画像と同じ幅・高さのアルファ付き画像を作成
            Mat alpha0 = new Mat(frame.rows(), frame.cols(), yukari.type());
            Imgproc.warpAffine(yukari, alpha0, affineTrans, alpha0.size(), Imgproc.INTER_LINEAR, Core.BORDER_TRANSPARENT, new Scalar(0, 0, 0));

            backgroundImage = overlayImage(backgroundImage, alpha0);
        }
        return backgroundImage;
    }

    /**
     *  目の基準となる四角を描く
     * @param frame
     * @return
     */
    private Mat drawRect(Mat frame) {
        if (!debug) {
            return frame;
        }
        Scalar color = new Scalar(0, 255, 0);
        double x = 500;
        double y = 500;
        Imgproc.rectangle(frame, new Point(y + 0, x + 0), new Point(y + minHeight, x + minWidth), color, 2);
        Imgproc.rectangle(frame, new Point(y + 0, x + 0), new Point(y + maxHeight, x + maxWidth), color, 2);
        return frame;
    }

    private List<MatOfPoint> contours;
    private Mat hierarchy;

    private void findContours(Mat maskedImage) {
        // init
        contours = new ArrayList<>();
        hierarchy = new Mat();

        // find contours
        // TODO 階層関係使ってないのでRETR_LISTで十分
        Imgproc.findContours(maskedImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
    }

    /**
     * 目の条件を満たしているか調べる。１つの目だけの条件
     * @param width
     * @param height
     * @return
     */
    private boolean isEye(double width, double height) {
        double areaMinThreshold = minWidth * minHeight;
        double areaMaxThreshold = maxWidth * maxHeight;
        double aspectMinThreshold = longAspect * 0.7;
        double aspectMaxThreshold = longAspect / 0.7;

        if (debug && this.state == State.RECT) {
            areaMinThreshold *= 0.8;
            areaMaxThreshold /= 0.8;
            aspectMinThreshold *= 0.8;
            aspectMaxThreshold /= 0.8;
        }

        double area = width * height;
        if (area < areaMinThreshold || area > areaMaxThreshold) {
            return false;
        }
        double aspect = Math.max(width, height) / Math.min(width, height);
        if (aspect < aspectMinThreshold || aspect > aspectMaxThreshold) {
            return false;
        }
        return true;
    }

    /**
     * 目の条件を満たしているか調べる。２つの目での条件
     * @param rr1
     * @param rr2
     * @return
     */
    private boolean isEyes(RotatedRect rr1, RotatedRect rr2) {
        if (debug && this.state == State.EYE || debug && this.state == State.RECT) {
            return true;
        }
        // 大きさがだいたい同じ
        double w1 = rr1.size.width;
        double h1 = rr1.size.height;
        double w2 = rr2.size.width;
        double h2 = rr2.size.height;
        if (Math.max(w1, w2) / Math.min(w1, w2) > 1.4 || Math.max(h1, h2) / Math.min(h1, h2) > 1.4) {
            return false;
        }

        // 角度がだいたい同じ
        double a1 = rr1.angle;
        double a2 = rr2.angle;
        if (Math.abs(a1 - a2) > 3) {
            return false;
        }

        // 距離がいい感じ
        double cx1;
        double cy1;
        {
            Point points[] = new Point[4];
            rr1.points(points);
            double sumX = 0.0;
            double sumY = 0.0;
            for (Point p : points) {
                sumX += p.x;
                sumY += p.y;
            }
            cx1 = sumX / 4.0;
            cy1 = sumY / 4.0;
        }
        double cx2;
        double cy2;
        double nearDistance = 0.0;
        double farDistance = 0.0;
        {
            Point points[] = new Point[4];
            rr2.points(points);
            double sumX = 0.0;
            double sumY = 0.0;
            int pi = 0;
            for (Point p : points) {
                sumX += p.x;
                sumY += p.y;

                double dist = Math.sqrt(Math.pow(cx1 - p.x, 2) + Math.pow(cy1 - p.y, 2));
                if (pi == 0) {
                    nearDistance = dist;
                    farDistance = dist;
                } else {
                    nearDistance = Math.min(nearDistance, dist);
                    farDistance = Math.max(farDistance, dist);
                }
                pi++;
            }
            cx2 = sumX / 4.0;
            cy2 = sumY / 4.0;
        }

        // 左右の目の重心は、目の横幅の5倍くらい離れているのが理想
        double distance = Math.sqrt(Math.pow(cx1 - cx2, 2) + Math.pow(cy1 - cy2, 2));
        double expected = Math.min(w1, h1) * 5;
        if (distance < expected * 0.7 || distance > expected / 0.7) {
            return false;
        }


        Log.d("isEyes", "near, far = " + nearDistance + ", " + farDistance);
        if (farDistance / nearDistance > 1.3) {
            return false;
        }

        return true;
    }

    private Mat drawEye(RotatedRect bbox, Mat frame) {
        Point points[] = new Point[4];
        bbox.points(points);
        MatOfPoint mop = new MatOfPoint(points);
        List<MatOfPoint> contours = new ArrayList<>();
        contours.add(mop);
        Imgproc.drawContours(frame, contours, 0, eyeColor, 2);
        return frame;
    }

    private List<List<RotatedRect>> eyes = new ArrayList<>();

    private Mat drawEyes(Mat maskedImage, Mat frame) {
        List<RotatedRect> rrList = new ArrayList<RotatedRect>();

        // if any contour exist...
        if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
            // for each contour, display it in blue
            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
                MatOfPoint pmat = contours.get(idx);
                MatOfPoint2f ptmat2 = new MatOfPoint2f(pmat.toArray()); // API的にFloat型に変換する
                RotatedRect bbox = Imgproc.minAreaRect(ptmat2); // 回転を考慮した外接矩形

                if (isEye(bbox.size.width, bbox.size.height)) {
                    rrList.add(bbox);
                }
            }
        }

        eyes.clear();
        int len = rrList.size();
        for (int i = 0; i< len; i ++) {
            RotatedRect rr1 = rrList.get(i);
            for (int j =0; j<len; j++){
                if (j == i){
                    continue;
                }
                RotatedRect rr2 = rrList.get(j);
                if (isEyes(rr1, rr2)) {
                    drawEye(rr1, frame);
                    drawEye(rr2, frame);
                    List<RotatedRect> eyePair = new ArrayList<>();
                    eyePair.add(rr1);
                    eyePair.add(rr2);
                    eyes.add(eyePair);
                }
            }
        }
        if(this.debug && this.state == State.EYES) {
            frame = drawYukari(frame);
        }
        return frame;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
