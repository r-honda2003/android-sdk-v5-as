package dji.sampleV5.aircraft.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.MutableLiveData
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
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

    private const val TAG = "MapMatchingRepeatVM"
    private const val INTERVAL_MS = 200L
    private const val HISTORY_LEN = 5

    data class PlanEntry(
        val time: Long,
        val imagePath: String,
        val cumDistanceM: Double
    )

    val isRunning = MutableLiveData(false)
    val currentProgress = MutableLiveData(0.0)
    val predictedSticks = MutableLiveData<IntArray>() // [lv, lh, rv, rh]

    private var plan = listOf<PlanEntry>()
    private var model: Module? = null
    private var handler: Handler? = null
    private var imageDir: File? = null
    
    private var totalPlanDistance = 0.0
    private var currentCumDistance = 0.0
    private var lastUpdateTime = 0L
    
    private val stickHistory = LinkedList<FloatArray>() // Each element is FloatArray(4)

    fun initAutoPilot(context: Context, imageDirName: String) {
        // Load Plan
        try {
            val planList = mutableListOf<PlanEntry>()
            context.assets.open("plan.csv").bufferedReader().use { reader ->
                val header = reader.readLine() // Skip header
                reader.forEachLine { line ->
                    val cols = line.split(",")
                    // plan.csv columns: keyframe_id, time, image_path, cum_north_m, cum_east_m, cum_distance_m, ...
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
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load plan.csv: $e")
        }

        // Load Model
        try {
            val modelPath = assetFilePath(context, "map_matching_model_mobile.ptl")
            model = LiteModuleLoader.load(modelPath)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $e")
        }

        // Set image directory
        val dir = context.getExternalFilesDir(null)
        imageDir = File(dir, imageDirName)
        Log.d(TAG, "Image directory set to: ${imageDir?.absolutePath}")
    }

    fun startAutoPilot() {
        if (isRunning.value == true) return
        isRunning.value = true
        currentCumDistance = 0.0
        lastUpdateTime = System.currentTimeMillis()
        stickHistory.clear()
        repeat(HISTORY_LEN) { stickHistory.add(FloatArray(4) { 0f }) }

        handler = Handler(Looper.getMainLooper())
        handler?.post(autoPilotTask)
        Log.d(TAG, "AutoPilot started")
    }

    fun stopAutoPilot() {
        isRunning.value = false
        handler?.removeCallbacksAndMessages(null)
        VirtualStickManager.getInstance().leftStick.horizontalPosition = 0
        VirtualStickManager.getInstance().leftStick.verticalPosition = 0
        VirtualStickManager.getInstance().rightStick.horizontalPosition = 0
        VirtualStickManager.getInstance().rightStick.verticalPosition = 0
        Log.d(TAG, "AutoPilot stopped")
    }

    private val autoPilotTask = object : Runnable {
        override fun run() {
            if (isRunning.value != true) return

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

            // 3. Get current image
            val currBitmap = getCurrentImageAsBitmap()

            if (currBitmap != null && refBitmap != null && model != null) {
                // 4. Run inference
                val sticks = runInference(currBitmap, refBitmap)
                predictedSticks.postValue(sticks)

                // 5. Apply sticks
                // Note: sticks are in range -660..660 (roughly, after multiplying by 660 in python, 
                // but the model output was Tanh [-1, 1] then scaled back to -1..1 in evaluate, 
                // and scaled by 660 for reporting. Our model output is Tanh [-1, 1]).
                // We need to scale them to -660..660.
                VirtualStickManager.getInstance().leftStick.verticalPosition = sticks[0]
                VirtualStickManager.getInstance().leftStick.horizontalPosition = sticks[1]
                VirtualStickManager.getInstance().rightStick.verticalPosition = sticks[2]
                VirtualStickManager.getInstance().rightStick.horizontalPosition = sticks[3]

                // 6. Update history
                updateHistory(sticks)
            }

            if (currentCumDistance < totalPlanDistance + 1.0) { // Keep running slightly past the end
                handler?.postDelayed(this, INTERVAL_MS)
            } else {
                Log.d(TAG, "End of plan reached. Stopping.")
                stopAutoPilot()
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
        
        val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun runInference(curr: Bitmap, ref: Bitmap): IntArray {
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
        return IntArray(4) { i ->
            (outData[i] * 660).toInt().coerceIn(-660, 660)
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        return TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
    }

    private fun updateHistory(sticks: IntArray) {
        stickHistory.removeFirst()
        stickHistory.addLast(FloatArray(4) { i -> sticks[i].toFloat() })
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { `is` ->
            file.outputStream().use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }
}
