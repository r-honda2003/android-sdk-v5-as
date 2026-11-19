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

    private var logFileDir: File? = null
    private var logFileWriter: java.io.FileWriter? = null
    private var lastInferenceTimeMs: Long = 0L

    fun initAutoPilot(context: Context, imageDirName: String) {
        statusMessage.postValue("Initializing AutoPilot...")
        logFileDir = context.getExternalFilesDir(null)
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
        try {
            val modelPath = assetFilePath(context, "map_matching_model_android.pt")
            model = Module.load(modelPath)
            Log.d(TAG, "PyTorch Mobile Model successfully loaded: $modelPath")
            statusMessage.postValue("Plan & AI Model OK (${plan.size} frames)")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to load model: $e", e)
            statusMessage.postValue("Plan OK. Model skipped: ${e.message}")
        }

        // Set image directory
        val dir = context.getExternalFilesDir(null)
        var targetDir = File(dir, imageDirName)
        if (!targetDir.exists()) {
            // 指定のディレクトリが存在しない場合、最新の images_* ディレクトリを自動選択
            val latestImagesDir = dir?.listFiles { file -> file.isDirectory && file.name.startsWith("images_") }
                ?.maxByOrNull { it.name }
            if (latestImagesDir != null) {
                targetDir = latestImagesDir
                Log.d(TAG, "Fallback to latest images directory: ${targetDir.name}")
            }
        }
        imageDir = targetDir
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

        // CSVログファイルの初期化
        try {
            val sessionId = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())
            val logFile = File(logFileDir, "autopilot_exec_$sessionId.csv")
            logFileWriter = java.io.FileWriter(logFile, true).apply {
                appendLine("timestamp,cum_distance_m,pred_lv,pred_lh,pred_rv,pred_rh,send_lv,send_lh,send_rv,send_rh,inference_time_ms,total_loop_time_ms,ref_image_path,pitch,roll,yaw")
                flush()
            }
            Log.d(TAG, "AutoPilot CSV logger started: ${logFile.absolutePath}")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to start CSV logger: $e", e)
        }

        // バーチャラスティック制御権の獲得
        try {
            VirtualStickManager.getInstance().speedLevel = 1.0
            VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false)

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

        try {
            logFileWriter?.flush()
            logFileWriter?.close()
            logFileWriter = null
            Log.d(TAG, "AutoPilot CSV logger stopped")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to close CSV logger: $e", e)
        }

        Log.d(TAG, "AutoPilot stopped")
    }

    private val autoPilotTask = object : Runnable {
        override fun run() {
            if (isRunning.value != true) return

            val loopStartTime = System.currentTimeMillis()
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

                // 4. AI推論 ＆ デフォルト確実前進保証 (-660 〜 660)
                val rawSticks = if (currBitmap != null && refBitmap != null && model != null) {
                    runInference(currBitmap, refBitmap)
                } else {
                    lastInferenceTimeMs = 0L
                    intArrayOf(0, 0, 350, 0)
                }

                // 5. -660〜+660 のスティック値を直接使用
                val sendLV = rawSticks[0].coerceIn(-660, 660)
                val sendLH = rawSticks[1].coerceIn(-660, 660)
                val sendRV = rawSticks[2].coerceIn(-660, 660)
                val sendRH = rawSticks[3].coerceIn(-660, 660)

                // スタート初期 (Frame 15コマ / 約3秒) は回転せずにまっすぐ前進を最優先！
                val finalLH = if (currentFrameIndex < 15) 0 else sendLH

                predictedSticks.postValue(rawSticks)
                val attitude = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude))
                val totalLoopTimeMs = System.currentTimeMillis() - loopStartTime

                statusMessage.postValue(
                    "AutoPilot Active | Dist:${String.format(java.util.Locale.US, "%.1f", currentCumDistance)}m\n" +
                    "【送信命令(Stick)】LV:$sendLV LH:$finalLH RV:$sendRV RH:$sendRH\n" +
                    "【処理速度】Infer:${lastInferenceTimeMs}ms | Loop:${totalLoopTimeMs}ms\n" +
                    "【実機レスポンス】Pitch:${String.format(java.util.Locale.US, "%.1f", attitude?.pitch ?: 0.0)}°"
                )

                // 6. ドローンへスティック値 (-660〜660) を直接送信
                VirtualStickManager.getInstance().leftStick.verticalPosition = sendLV
                VirtualStickManager.getInstance().leftStick.horizontalPosition = finalLH
                VirtualStickManager.getInstance().rightStick.verticalPosition = sendRV
                VirtualStickManager.getInstance().rightStick.horizontalPosition = sendRH

                // 7. CSVログへの即時保存 (AI推論値, 送信スティック値[-660~660], 推論時間ms, ループ処理時間ms)
                try {
                    logFileWriter?.appendLine(
                        "$now,${String.format(java.util.Locale.US, "%.2f", currentCumDistance)}," +
                        "${rawSticks[0]},${rawSticks[1]},${rawSticks[2]},${rawSticks[3]}," +
                        "$sendLV,$finalLH,$sendRV,$sendRH," +
                        "$lastInferenceTimeMs,$totalLoopTimeMs,${refEntry?.imagePath ?: ""}," +
                        "${attitude?.pitch ?: 0.0},${attitude?.roll ?: 0.0},${attitude?.yaw ?: 0.0}"
                    )
                    logFileWriter?.flush()
                } catch (e: Throwable) {
                    Log.e(TAG, "Failed to write CSV log row: $e", e)
                }

                currentFrameIndex++
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
        val startInference = System.currentTimeMillis()
        val result = try {
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
        lastInferenceTimeMs = System.currentTimeMillis() - startInference
        return result
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
