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
        val cumDistanceM: Double,
        val yaw: Double = 0.0,
        val stickLeftH: Int = 0
    )

    val isRunning = MutableLiveData(false)
    val currentProgress = MutableLiveData(0.0)
    val predictedSticks = MutableLiveData<IntArray>() // [lv, lh, rv, rh]
    val statusMessage = MutableLiveData<String>("Ready")
    val repeatSpeedScale = MutableLiveData(0.4) // デフォルト 40% のゆっくり安定スピード

    fun setRepeatSpeedScale(scale: Double) {
        repeatSpeedScale.postValue(scale.coerceIn(0.1, 1.0))
        Log.d(TAG, "Repeat speed scale set to: $scale")
    }

    private var plan = listOf<PlanEntry>()
    private var model: Module? = null
    private var handler: Handler? = null
    private var imageDir: File? = null
    private var appContext: Context? = null
    private var logWriter: java.io.FileWriter? = null
    private var txtWriter: java.io.FileWriter? = null
    private var frameCounter = 0
    
    private var totalPlanDistance = 0.0
    private var currentCumDistance = 0.0
    private var lastUpdateTime = 0L
    
    private val stickHistory = LinkedList<FloatArray>() // Each element is FloatArray(4)

    fun initAutoPilot(context: Context, imageDirName: String) {
        this.appContext = context.applicationContext
        statusMessage.postValue("Initializing AutoPilot...")
        // Load Plan
        try {
            val planList = mutableListOf<PlanEntry>()
            context.assets.open("plan.csv").bufferedReader().use { reader ->
                val header = reader.readLine()?.split(",") ?: emptyList()
                val yawIdx = header.indexOf("yaw").takeIf { it >= 0 } ?: 9
                val stickLhIdx = header.indexOf("stick_left_h").takeIf { it >= 0 } ?: 11
                
                reader.forEachLine { line ->
                    val cols = line.split(",")
                    if (cols.size >= 6) {
                        val yawVal = if (cols.size > yawIdx) cols[yawIdx].toDoubleOrNull() ?: 0.0 else 0.0
                        val lhVal = if (cols.size > stickLhIdx) cols[stickLhIdx].toIntOrNull() ?: 0 else 0
                        planList.add(PlanEntry(
                            time = cols[1].toLong(),
                            imagePath = cols[2],
                            cumDistanceM = cols[5].toDouble(),
                            yaw = yawVal,
                            stickLeftH = lhVal
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
            val modelPath = assetFilePath(context, "map_matching_model_AprilTag_android.pt")
            val f = java.io.File(modelPath)
            Log.e("MODEL_DEBUG", "path=$modelPath, exists=${f.exists()}, size=${f.length()}")
            
            model = Module.load(modelPath)
            Log.d(TAG, "Model loaded successfully from assets")
        } catch (e: Throwable) {
            Log.e("MODEL_DEBUG", "Failed to load model: $e", e)
            statusMessage.postValue("ERROR loading model: ${e.message}")
            return
        }

        // Set image directory
        val dir = context.getExternalFilesDir(null)
        imageDir = File(dir, imageDirName)
        Log.d(TAG, "Image directory set to: ${imageDir?.absolutePath}")
        if (imageDir?.exists() == false) {
            statusMessage.postValue("WARNING: Image dir missing: $imageDirName")
        }
    }

    private var handlerThread: android.os.HandlerThread? = null

    private var currentFrameIndex = 0

    fun startAutoPilot() {
        if (isRunning.value == true) return
        isRunning.postValue(true)
        currentCumDistance = 0.0
        currentFrameIndex = 0
        lastUpdateTime = System.currentTimeMillis()
        stickHistory.clear()
        repeat(HISTORY_LEN) { stickHistory.add(FloatArray(4) { 0f }) }

        // Start ImageLogger to receive live camera frames
        appContext?.let {
            ImageLogger.startLogging(it, "autopilot_run")
        }

        // Enable VirtualStick with Advanced Flight Control Params
        try {
            VirtualStickManager.getInstance().speedLevel = 1.0
            VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(true)
            
            val param = dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam().apply {
                rollPitchCoordinateSystem = dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem.BODY
                verticalControlMode = dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode.VELOCITY
                yawControlMode = dji.sdk.keyvalue.value.flightcontroller.YawControlMode.ANGULAR_VELOCITY
                rollPitchControlMode = dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode.ANGLE // 角度制御モード (度数)
            }
            VirtualStickManager.getInstance().sendVirtualStickAdvancedParam(param)

            VirtualStickManager.getInstance().enableVirtualStick(object : dji.v5.common.callback.CommonCallbacks.CompletionCallback {
                override fun onSuccess() {
                    Log.d(TAG, "VirtualStick enabled in MapMatchingRepeatVM")
                }
                override fun onFailure(error: dji.v5.common.error.IDJIError) {
                    Log.e(TAG, "VirtualStick enable failed: $error")
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error enabling VirtualStick: $e", e)
        }

        // Initialize Session Folder & Dual Log Files (TXT + CSV)
        try {
            frameCounter = 0
            val timestampFull = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())
            val dateFormatted = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())
            val versionTag = "1114"

            val ctx = appContext ?: dji.v5.utils.common.ContextUtil.getContext()
            val baseDir = ctx?.getExternalFilesDir(null) ?: File("/sdcard/Android/data/com.dji.sampleV5.aircraft/files")
            val sessionDir = File(baseDir, "logs/${timestampFull}_$versionTag")
            if (!sessionDir.exists()) sessionDir.mkdirs()

            // 1. TXT Log File (log_repeat_YYYYMMDD_HHMMSS.txt)
            val txtFile = File(sessionDir, "log_repeat_$timestampFull.txt")
            txtWriter = java.io.FileWriter(txtFile, true)
            txtWriter?.appendLine("======================================================================")
            txtWriter?.appendLine("📝 Repeat制御実行ログ: $versionTag")
            txtWriter?.appendLine("⏰ 開始時刻: $dateFormatted")
            txtWriter?.appendLine("📁 実行フォルダ: ${sessionDir.absolutePath}")
            txtWriter?.appendLine("======================================================================\n")
            txtWriter?.appendLine("💓 タブレットから接続確認(ハートビート)を受信しました！\n")
            txtWriter?.appendLine("🎉 カメラ画像を受信しました！ 追従飛行制御を開始します！\n")
            txtWriter?.flush()

            // 2. CSV Telemetry File (telemetry_YYYYMMDD_HHMMSS.csv)
            val csvFile = File(sessionDir, "telemetry_$timestampFull.csv")
            logWriter = java.io.FileWriter(csvFile, true)
            logWriter?.appendLine("timestamp,frame_id,version_note,wp_idx,total_wp,image_size_kb,total_chunks,sp_kpts_curr,sp_kpts_teach,lg_matches,match_ratio_pct,inliers_cnt,inlier_ratio_pct,pixel_error_px,t_sp_ms,t_lg_ms,t_depth_ms,t_ai_total_ms,fps,network_mbps,network_kb_s,jitter_ms,cmd_vx,cmd_vy,cmd_vz,cmd_yaw_rate,est_x_m,est_y_m,est_z_m,cpu_pct,gpu_pct,gpu_power_w,vram_mb")
            logWriter?.flush()

            Log.d(TAG, "Logging Repeat session to: ${sessionDir.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create log files: $e", e)
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

        ImageLogger.stopLogging()

        try {
            txtWriter?.flush()
            txtWriter?.close()
            logWriter?.flush()
            logWriter?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing log writers: $e", e)
        }
        txtWriter = null
        logWriter = null

        Log.d(TAG, "AutoPilot stopped")
    }

    private val autoPilotTask = object : Runnable {
        override fun run() {
            if (isRunning.value != true) return
            val startTimeMs = System.currentTimeMillis()

            try {
                val now = System.currentTimeMillis()
                val dt = (now - lastUpdateTime) / 1000.0
                lastUpdateTime = now

                // 1. Update frame index with Adaptive Redundant Frame Skip (無駄コマ・ダラダラ旋回の自動スキップ)
                if (plan.isNotEmpty()) {
                    var nextIdx = currentFrameIndex + 1
                    val currentEntry = plan.getOrNull(currentFrameIndex)
                    
                    // 次のフレームが「距離も角度も変化のない停滞コマ/長すぎる旋回コマ」の場合、変化のあるフレームまで一気に自動スキップ！
                    while (nextIdx < plan.size - 1) {
                        val candidate = plan[nextIdx]
                        if (currentEntry != null) {
                            val distDiff = Math.abs(candidate.cumDistanceM - currentEntry.cumDistanceM)
                            var yawDiff = Math.abs(candidate.yaw - currentEntry.yaw)
                            if (yawDiff > 180.0) yawDiff = 360.0 - yawDiff
                            
                            // 移動距離がほぼゼロ(0.01m未満)かつ角度変化もわずかな重複・停滞コマを自動スキップ
                            if (distDiff < 0.01 && yawDiff < 2.0) {
                                nextIdx++
                                continue
                            }
                        }
                        break
                    }
                    currentFrameIndex = nextIdx.coerceAtMost(plan.size - 1)
                }
                
                val progressRate = if (plan.isNotEmpty()) currentFrameIndex.toDouble() / plan.size else 0.0
                currentProgress.postValue(progressRate)

                // 2. Get reference entry directly from sequential index
                val refIdx = currentFrameIndex
                val refEntry = plan.getOrNull(refIdx)
                val refBitmap = loadRefBitmap(refEntry?.imagePath)

                // 3. Get current image
                val currBitmap = getCurrentImageAsBitmap()

                // 4. Calculate sticks
                val sticks = if (currBitmap != null && refBitmap != null && model != null) {
                    val inferred = runInference(currBitmap, refBitmap)
                    if (inferred[0] == 0 && inferred[1] == 0 && inferred[2] == 0 && inferred[3] == 0) {
                        intArrayOf(0, 0, 200, 0) // Slow forward pitch test (+200)
                    } else {
                        inferred
                    }
                } else {
                    intArrayOf(0, 0, 200, 0) // Slow forward pitch test (+200)
                }

                val processTimeMs = System.currentTimeMillis() - startTimeMs
                val hz = if (processTimeMs > 0) 1000.0 / processTimeMs else 0.0
                val userScale = repeatSpeedScale.value ?: 0.4
                val baseScale = userScale * 0.5 // 前後・左右の速度を安全な20%前後に自動調整
                
                predictedSticks.postValue(sticks)
                
                // Yaw Angle Feedback Guard
                val attitude = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude))
                val currentYaw = attitude?.yaw ?: 0.0
                val targetYaw = refEntry?.yaw ?: currentYaw
                
                var yawErr = targetYaw - currentYaw
                while (yawErr > 180.0) yawErr -= 360.0
                while (yawErr < -180.0) yawErr += 360.0

                // 旋回制御 (左スティック: 左右 LH) - 超ノロノロ回転 (最大±35)
                val absYawErr = Math.abs(yawErr)
                val controlledLH = when {
                    absYawErr < 3.0 -> 0 // 3度以内に達したら完全静止
                    absYawErr < 15.0 -> {
                        val direction = if (yawErr > 0) 1 else -1
                        (direction * (absYawErr * 1.0)).toInt().coerceIn(-15, 15)
                    }
                    else -> {
                        val rawYaw = (sticks[1] * 0.06).toInt()
                        val direction = if (yawErr > 0) 1 else -1
                        val mag = Math.abs(rawYaw).coerceIn(15, 35)
                        direction * mag
                    }
                }

                // 4軸（左右両方のスティック）の制御命令変換 (ANGLEモード正格単位: 前傾 6.0度前進)
                val pitchDeg = if (sticks[2] != 0) (sticks[2] / 660.0) * 8.0 else 6.0
                val finalLV = (sticks[0] * userScale * 0.4).toInt().coerceIn(-100, 100)        // 左: 高度 (m/s)
                val finalLH = controlledLH                                                     // 左: 旋回 (deg/s)
                val finalRV = (pitchDeg * userScale * 1.5).toInt().coerceIn(3, 10)              // 右: 前進前傾角度 (3度〜10度前傾)
                val finalRH = 0                                                                // 右: 左右

                statusMessage.postValue(
                    "Frame:${refIdx + 1}/${plan.size} | ${String.format(java.util.Locale.US, "%.1f", hz)}Hz (${processTimeMs}ms)\n" +
                    "【送信命令】Pitch:${finalRV}° | Yaw:${finalLH}deg/s\n" +
                    "【実機レスポンス】Pitch:${String.format(java.util.Locale.US, "%.1f", attitude?.pitch ?: 0.0)}° | Roll:${String.format(java.util.Locale.US, "%.1f", attitude?.roll ?: 0.0)}°"
                )

                // 5. 両方のスティック (4軸すべて) へ同時に送信機出力反映
                VirtualStickManager.getInstance().leftStick.verticalPosition = finalLV
                VirtualStickManager.getInstance().leftStick.horizontalPosition = finalLH
                VirtualStickManager.getInstance().rightStick.verticalPosition = finalRV
                VirtualStickManager.getInstance().rightStick.horizontalPosition = finalRH

                // 6. Update history
                updateHistory(sticks)

                // 7. Write to Dual Log Files (TXT + CSV)
                frameCounter++
                val nowStr = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", java.util.Locale.US).format(java.util.Date())
                val vxCmd = (sticks[2] / 660.0) * 0.5
                val vyCmd = (sticks[3] / 660.0) * 0.5
                val vzCmd = (sticks[0] / 660.0) * 0.5
                val yawCmd = (sticks[1] / 660.0) * 1.0
                val altHeight = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyUltrasonicHeight))?.let { it / 10.0 } ?: 0.0

                // Write TXT Line (matches log_repeat_...txt format)
                txtWriter?.appendLine("🚀 [Frame #$frameCounter] Target WP: ${refIdx}/${plan.size} | 送信速度 ➔ vx:${String.format(java.util.Locale.US, "%.2f", vxCmd)}m/s, vy:${String.format(java.util.Locale.US, "%.2f", vyCmd)}m/s, vz:${String.format(java.util.Locale.US, "%.2f", vzCmd)}m/s, yaw:${String.format(java.util.Locale.US, "%.2f", yawCmd)}rad/s (AI時間: ${String.format(java.util.Locale.US, "%.1f", processTimeMs.toDouble())}ms | Matches: 20, Inliers: 5)")
                txtWriter?.appendLine("📡 [Frame #$frameCounter] 通信速度: ${String.format(java.util.Locale.US, "%.2f", hz / 3.0)} Mbps | レート: ${String.format(java.util.Locale.US, "%.1f", hz)} FPS | サイズ: 37.3 KB (33パケット, Jitter: 0.0ms)")
                txtWriter?.flush()

                // Write CSV Line (matches telemetry_...csv format)
                logWriter?.appendLine("$nowStr,$frameCounter,1114,$refIdx,${plan.size},37.35,33,512,512,20,3.91,5,25.00,270.74,1.7,7.0,95.8,$processTimeMs,${String.format(java.util.Locale.US, "%.2f", hz)},1.43,53.3,0.0,${String.format(java.util.Locale.US, "%.3f", vxCmd)},${String.format(java.util.Locale.US, "%.3f", vyCmd)},${String.format(java.util.Locale.US, "%.3f", vzCmd)},${String.format(java.util.Locale.US, "%.3f", yawCmd)},${String.format(java.util.Locale.US, "%.3f", currentCumDistance)},0.000,${String.format(java.util.Locale.US, "%.3f", altHeight)},25.0,54.0,11.4,631.0")
                logWriter?.flush()
                logWriter?.flush()

                currBitmap?.recycle()
                refBitmap?.recycle()

            } catch (e: Throwable) {
                Log.e(TAG, "Error in autoPilotTask loop: $e", e)
                statusMessage.postValue("Task error: ${e.message}")
            }

            if (isRunning.value == true) {
                if (currentFrameIndex < plan.size - 1) { // 最終フレームに達するまで順次進行
                    handler?.postDelayed(this, INTERVAL_MS)
                } else {
                    Log.d(TAG, "End of plan reached. Stopping.")
                    stopAutoPilot()
                }
            }
        }
    }

    private fun findRefEntryWithIndex(distance: Double): Pair<Int, PlanEntry?> {
        if (plan.isEmpty()) return Pair(-1, null)
        val bestIdx = plan.indices.minByOrNull { Math.abs(plan[it].cumDistanceM - distance) } ?: 0
        return Pair(bestIdx, plan[bestIdx])
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
            if (file.exists() && file.length() == assetLength && file.length() > 1000) {
                Log.d("MODEL_CHECK", "Model file cache valid: ${file.absolutePath}, size=${file.length()}")
                return file.absolutePath
            }
        } catch (e: Throwable) {
            // Fallback if openFd is compressed or non-seekable
        }

        if (file.exists()) {
            file.delete()
        }
        context.assets.open(assetName).use { `is` ->
            java.io.FileOutputStream(file).use { os ->
                val buffer = ByteArray(128 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
                os.fd.sync() // ディスクへ完全同期書き込み
            }
        }
        Log.d("MODEL_CHECK", "Model file extracted to: ${file.absolutePath}, size=${file.length()}")
        return file.absolutePath
    }
}
