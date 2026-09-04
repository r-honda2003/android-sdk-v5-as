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

        // Load Custom 2D-CNN Model
        try {
            val modelPath = assetFilePath(context, "map_matching_model_mobile.ptl")
            val modelFile = File(modelPath)
            if (modelFile.exists() && modelFile.length() > 0L) {
                model = Module.load(modelPath)
                Log.d(TAG, "Custom 2D-CNN model loaded successfully: $modelPath")
                statusMessage.postValue("Custom 2D-CNN Model Loaded. Ready!")
            } else {
                statusMessage.postValue("ERROR: map_matching_model_mobile.ptl missing")
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to load custom 2D-CNN model: $e", e)
            statusMessage.postValue("ERROR loading model: ${e.message}")
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

    fun startAutoPilot() {
        if (isRunning.value == true) return
        isRunning.postValue(true)
        currentCumDistance = 0.0
        lastUpdateTime = System.currentTimeMillis()
        stickHistory.clear()
        repeat(HISTORY_LEN) { stickHistory.add(FloatArray(4) { 0f }) }

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

                if (currBitmap != null && refBitmap != null && model != null) {
                    // 4. Run inference
                    val sticks = runInference(currBitmap, refBitmap)
                    predictedSticks.postValue(sticks)
                    statusMessage.postValue("Running: LV:${sticks[0]} LH:${sticks[1]} RV:${sticks[2]} RH:${sticks[3]}")

                    // 5. Apply sticks
                    VirtualStickManager.getInstance().leftStick.verticalPosition = sticks[0]
                    VirtualStickManager.getInstance().leftStick.horizontalPosition = sticks[1]
                    VirtualStickManager.getInstance().rightStick.verticalPosition = sticks[2]
                    VirtualStickManager.getInstance().rightStick.horizontalPosition = sticks[3]

                    // 6. Update history
                    updateHistory(sticks)
                }

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
