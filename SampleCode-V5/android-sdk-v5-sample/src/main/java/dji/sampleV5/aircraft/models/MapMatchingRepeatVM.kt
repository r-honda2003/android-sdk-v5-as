package dji.sampleV5.aircraft.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.util.Log
import androidx.lifecycle.MutableLiveData
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStreamReader
import java.util.LinkedList
import kotlin.math.hypot

class MapMatchingRepeatVM : DJIViewModel() {

    companion object {
        private const val TAG = "MapMatchingRepeatVM"
        private const val INTERVAL_MS = 200L
        private const val HISTORY_LEN = 5
    }

    data class PlanEntry(
        val time: Long,
        val imagePath: String,
        val cumDistanceM: Double
    )

    val isRunning = MutableLiveData(false)
    val currentProgress = MutableLiveData(0.0)
    val predictedSticks = MutableLiveData<IntArray>() // [lv, lh, rv, rh]
    val statusMessage = MutableLiveData<String>("Ready")

    private var plan = listOf<PlanEntry>()
    private var model: Module? = null
    private var handler: Handler? = null
    private var imageDir: File? = null
    private var currentFrameIndex = 0
    
    private var totalPlanDistance = 0.0
    private var currentCumDistance = 0.0
    private var lastUpdateTime = 0L
    
    private val stickHistory = LinkedList<FloatArray>() // Each element is FloatArray(4)

    fun initAutoPilot(context: Context, imageDirName: String) {
        statusMessage.postValue("Initializing AutoPilot...")
        // Load Plan
        try {
            val planList = mutableListOf<PlanEntry>()
            context.assets.open("plan.csv").bufferedReader().use { reader ->
                reader.readLine() // Skip header
                reader.forEachLine { line ->
                    val cols = line.split(",")
                    if (cols.size >= 6) {
                        planList.add(PlanEntry(
                            time = cols[1].toLong(),
                            imagePath = cols[2],
                            cumDistanceM = cols[5].toDouble()
                        ))
                    }
                }
            }
            plan = planList.sortedBy { it.cumDistanceM }
            totalPlanDistance = plan.lastOrNull()?.cumDistanceM ?: 0.0
            Log.d(TAG, "Plan loaded: ${plan.size} entries, total distance: $totalPlanDistance")
            statusMessage.postValue("Plan OK: ${plan.size} frames (${String.format("%.1f", totalPlanDistance)}m)")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to load plan.csv: $e", e)
            statusMessage.postValue("ERROR loading plan.csv: ${e.message}")
            return
        }

        // Load Model
        // 注: 現在アセット内にあるモデルファイル (.ptl / .pt) は PyTorch Mobile 1.13.1 の C++ Native Loader と
        // バージョン非互換があり、Module.load を実行すると Native SIGABRT でアプリが落ちるためロードを安全に停止します。
        statusMessage.postValue("Plan OK. Model skipped (Native crash prevention)")

        // Set image directory
        val dir = context.getExternalFilesDir(null)
        imageDir = File(dir, imageDirName)
        Log.d(TAG, "Image directory set to: ${imageDir?.absolutePath}")
        if (imageDir?.exists() == false) {
            statusMessage.postValue("WARNING: Image dir missing: $imageDirName")
        }
    }

    private var handlerThread: android.os.HandlerThread? = null

    fun startAutoPilot() {
        if (isRunning.value == true) return
        isRunning.postValue(true)
        currentCumDistance = 0.0
        currentFrameIndex = 0
        lastUpdateTime = System.currentTimeMillis()
        stickHistory.clear()
        repeat(HISTORY_LEN) { stickHistory.add(FloatArray(4) { 0f }) }

        // バーチャラスティック制御権の獲得
        try {
            VirtualStickManager.getInstance().speedLevel = 1.0
            VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(true)

            val param = dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam().apply {
                rollPitchCoordinateSystem = dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem.BODY
                verticalControlMode = dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode.VELOCITY
                yawControlMode = dji.sdk.keyvalue.value.flightcontroller.YawControlMode.ANGULAR_VELOCITY
                rollPitchControlMode = dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode.VELOCITY // ★ 速度制御モード (m/s)
            }
            VirtualStickManager.getInstance().sendVirtualStickAdvancedParam(param)

            VirtualStickManager.getInstance().enableVirtualStick(object : dji.v5.common.callback.CommonCallbacks.CompletionCallback {
                override fun onSuccess() {
                    Log.d(TAG, "VirtualStick enable SUCCESS")
                }
                override fun onFailure(error: dji.v5.common.error.IDJIError) {
                    Log.e(TAG, "VS_ERROR [Code:${error.errorCode()}]: ${error.description()}")
                }
            })
        } catch (e: Throwable) {
            Log.e(TAG, "VS_EX: ${e.message}")
        }

        handlerThread = android.os.HandlerThread("AutoPilotThread").apply { start() }
        handler = Handler(handlerThread!!.looper)
        handler?.post(autoPilotTask)
        Log.d(TAG, "AutoPilot started")
    }

    fun stopAutoPilot() {
        isRunning.postValue(false)
        handler?.removeCallbacksAndMessages(null)
        handlerThread?.quitSafely()
        handlerThread = null
        VirtualStickManager.getInstance().leftStick.horizontalPosition = 0
        VirtualStickManager.getInstance().leftStick.verticalPosition = 0
        VirtualStickManager.getInstance().rightStick.horizontalPosition = 0
        VirtualStickManager.getInstance().rightStick.verticalPosition = 0
        Log.d(TAG, "AutoPilot stopped")
    }

    private val autoPilotTask = object : Runnable {
        override fun run() {
            if (isRunning.value != true) return

            try {
                val now = System.currentTimeMillis()
                val dt = (now - lastUpdateTime) / 1000.0
                lastUpdateTime = now

                // 1. Update progress using velocity
                val velocity = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAircraftVelocity))
                if (velocity != null) {
                    val speed = hypot(velocity.x, velocity.y)
                    currentCumDistance += speed * dt
                }
                
                val progressRate = if (totalPlanDistance > 0) currentCumDistance / totalPlanDistance else 0.0
                currentProgress.postValue(progressRate)

                // 2. Find reference image
                val refEntry = findRefEntry(currentCumDistance)
                val refBitmap = loadRefBitmap(refEntry?.imagePath)
                if (refBitmap == null && refEntry?.imagePath != null) {
                    statusMessage.postValue("Ref img missing: ${refEntry.imagePath}")
                }

                // 3. Get current image
                val currBitmap = getCurrentImageAsBitmap()
                if (currBitmap == null) {
                    statusMessage.postValue("Waiting for camera frame...")
                }

                // 4. AI推論 ＆ デフォルト確実前進保証
                val rawSticks = if (currBitmap != null && refBitmap != null && model != null) {
                    runInference(currBitmap, refBitmap)
                } else {
                    intArrayOf(0, 0, 350, 0)
                }

                // 5. 前進速度変換 (VELOCITYモード: 秒速 0.3m/s ＝ 30)
                val pitchInput = if (rawSticks[2] > 50) (rawSticks[2] / 660.0) * 0.4 else 0.3
                val finalRV = (pitchInput * 100).toInt().coerceIn(20, 40) // 毎秒 0.2m〜0.4m (約0.3m/sノロノロ安全走行)

                // 6. 旋回Yaw角度フィードバック制御 (スタート直後3秒間はまっすぐ前進優先)
                val attitude = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude))
                val currentYaw = attitude?.yaw ?: 0.0
                val targetYaw = currentYaw
                var yawErr = targetYaw - currentYaw
                while (yawErr > 180.0) yawErr -= 360.0
                while (yawErr < -180.0) yawErr += 360.0

                val absYawErr = Math.abs(yawErr)
                val calculatedLH = when {
                    absYawErr < 3.0 -> 0
                    absYawErr < 15.0 -> (if (yawErr > 0) 1 else -1) * 15
                    else -> (if (yawErr > 0) 1 else -1) * 35
                }

                // スタート初期 (Frame 15コマ / 約3秒) は回転せずにまっすぐ前進を最優先！
                val finalLH = if (currentFrameIndex < 15) 0 else calculatedLH

                predictedSticks.postValue(rawSticks)
                statusMessage.postValue(
                    "AutoPilot Active | Dist:${String.format(java.util.Locale.US, "%.1f", currentCumDistance)}m\n" +
                    "【送信命令】Speed:${String.format(java.util.Locale.US, "%.2f", finalRV / 100.0)}m/s | Yaw:${finalLH}deg/s\n" +
                    "【実機レスポンス】Pitch:${String.format(java.util.Locale.US, "%.1f", attitude?.pitch ?: 0.0)}°"
                )

                // 7. ドローンへ確実に有効コマンド送信
                VirtualStickManager.getInstance().leftStick.verticalPosition = 0
                VirtualStickManager.getInstance().leftStick.horizontalPosition = finalLH
                VirtualStickManager.getInstance().rightStick.verticalPosition = finalRV
                VirtualStickManager.getInstance().rightStick.horizontalPosition = 0

                updateHistory(rawSticks)

                currBitmap?.recycle()
                refBitmap?.recycle()

            } catch (e: Throwable) {
                Log.e(TAG, "Error in autoPilotTask loop: $e", e)
                statusMessage.postValue("Task error: ${e.message}")
            }

            if (isRunning.value == true) {
                if (currentCumDistance < totalPlanDistance + 1.0) { // Keep running slightly past the end
                    handler?.postDelayed(this, INTERVAL_MS)
                } else {
                    Log.d(TAG, "End of plan reached. Stopping.")
                    stopAutoPilot()
                }
            }
        }
    }

    private fun findRefEntry(distance: Double): PlanEntry? {
        if (plan.isEmpty()) return null
        return plan.minByOrNull { Math.abs(it.cumDistanceM - distance) }
    }

    private fun loadRefBitmap(imagePath: String?): Bitmap? {
        if (imagePath == null || imageDir == null) return null
        val file = File(imageDir, imagePath)
        if (!file.exists()) {
            Log.e(TAG, "Reference image not found: ${file.absolutePath}")
            return null
        }
        return BitmapFactory.decodeFile(file.absolutePath)
    }

    private fun getCurrentImageAsBitmap(): Bitmap? {
        val data = ImageLogger.getLatestFrameData() ?: return null
        val width = ImageLogger.getLatestFrameWidth()
        val height = ImageLogger.getLatestFrameHeight()
        if (width <= 0 || height <= 0) return null

        return try {
            val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            val ok = yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
            if (!ok) return null
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to get current image as bitmap: $e")
            null
        }
    }

    private fun runInference(curr: Bitmap, ref: Bitmap): IntArray {
        return try {
            // Preprocess
            val currTensor = bitmapToTensor(curr)
            val refTensor = bitmapToTensor(ref)
            
            // History tensor [1, HISTORY_LEN * 4]
            val histData = FloatArray(HISTORY_LEN * 4)
            for (i in 0 until HISTORY_LEN) {
                val s = stickHistory[i]
                histData[i * 4 + 0] = s[0] / 660f
                histData[i * 4 + 1] = s[1] / 660f
                histData[i * 4 + 2] = s[2] / 660f
                histData[i * 4 + 3] = s[3] / 660f
            }
            val histTensor = Tensor.fromBlob(histData, longArrayOf(1, (HISTORY_LEN * 4).toLong()))

            // Forward
            val output = model?.forward(
                IValue.from(currTensor),
                IValue.from(refTensor),
                IValue.from(histTensor)
            )?.toTensor()

            val outData = output?.dataAsFloatArray ?: floatArrayOf(0f, 0f, 0f, 0f)
            
            // Convert to Int sticks (-660 to 660)
            IntArray(4) { i ->
                (outData[i] * 660).toInt().coerceIn(-660, 660)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Inference error: $e", e)
            IntArray(4) { 0 }
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        if (resized != bitmap) {
            resized.recycle()
        }
        return tensor
    }

    private fun updateHistory(sticks: IntArray) {
        stickHistory.removeFirst()
        stickHistory.addLast(FloatArray(4) { i -> sticks[i].toFloat() })
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        try {
            val assetFd = context.assets.openFd(assetName)
            val assetLength = assetFd.length
            assetFd.close()
            if (file.exists() && file.length() == assetLength) {
                return file.absolutePath
            }
        } catch (e: Throwable) {
            // Fallback if openFd is compressed
        }

        if (file.exists()) {
            file.delete()
        }
        context.assets.open(assetName).use { `is` ->
            file.outputStream().use { os ->
                val buffer = ByteArray(64 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
        }
        return file.absolutePath
    }
}
