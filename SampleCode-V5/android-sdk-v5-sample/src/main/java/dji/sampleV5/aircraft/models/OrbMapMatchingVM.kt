package dji.sampleV5.aircraft.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.lifecycle.MutableLiveData
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.features2d.DescriptorMatcher
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.*
import kotlin.math.hypot

class OrbMapMatchingVM : DJIViewModel() {

    companion object {
        private const val TAG = "OrbMapMatchingVM"
        private const val INTERVAL_MS = 200L
        private const val MAX_MATCHES = 50
        
        // Control Gains (Tune these)
        private const val ROLL_GAIN = 0.5f 
        private const val PITCH_SPEED = 150 // Increased for positive initial movement
    }

    data class PlanEntry(
        val time: Long,
        val imagePath: String,
        val cumDistanceM: Double
    )

    val isRunning = MutableLiveData(false)
    val currentProgress = MutableLiveData(0.0)
    val statusMessage = MutableLiveData<String>("Ready (ORB)")

    private var plan = listOf<PlanEntry>()
    private var handler: Handler? = null
    private var imageDir: File? = null
    
    private var totalPlanDistance = 0.0
    private var currentCumDistance = 0.0
    private var lastUpdateTime = 0L

    // OpenCV Objects
    private lateinit var orb: ORB
    private lateinit var matcher: BFMatcher

    init {
        try {
            if (!OpenCVLoader.initDebug()) {
                Log.e(TAG, "OpenCVLoader.initDebug() failed, trying loadLibrary...")
                System.loadLibrary("opencv_java4")
            }
            Log.d(TAG, "OpenCV initialized successfully.")
            orb = ORB.create(500)
            matcher = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, true)
            statusMessage.postValue("ORB System OK")
        } catch (e: Throwable) {
            Log.e(TAG, "OpenCV load failed: ${e.message}")
            statusMessage.postValue("ORB Load Error: ${e.message}")
        }
    }

    fun initAutoPilot(context: Context, imageDirName: String) {
        statusMessage.postValue("Initializing ORB TaR...")
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
            
            val dir = context.getExternalFilesDir(null)
            imageDir = File(dir, imageDirName)
            
            statusMessage.postValue("ORB Ready: ${plan.size} frames")
        } catch (e: Exception) {
            Log.e(TAG, "Init failed: $e")
            statusMessage.postValue("ERROR: ${e.message}")
        }
    }

    private var handlerThread: HandlerThread? = null

    fun startAutoPilot() {
        if (isRunning.value == true) return

        // 1. Set Virtual Stick Parameters & Speed Level (Crucial for MSDK V5)
        try {
            val vsm = VirtualStickManager.getInstance()
            vsm.init()
            vsm.setSpeedLevel(0.5) // Set to 50% speed for safety
            vsm.setVirtualStickAdvancedModeEnabled(true)

            val param = VirtualStickFlightControlParam()
            param.rollPitchCoordinateSystem = FlightCoordinateSystem.BODY
            param.verticalControlMode = VerticalControlMode.VELOCITY
            param.yawControlMode = YawControlMode.ANGULAR_VELOCITY
            param.rollPitchControlMode = RollPitchControlMode.ANGLE
            vsm.sendVirtualStickAdvancedParam(param)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup VSM: $e")
        }

        isRunning.postValue(true)
        currentCumDistance = 0.0
        lastUpdateTime = System.currentTimeMillis()

        handlerThread = HandlerThread("OrbAutoPilotThread").apply { start() }
        handler = Handler(handlerThread!!.looper)
        handler?.post(autoPilotTask)
        Log.d(TAG, "ORB AutoPilot started")
    }

    fun stopAutoPilot() {
        isRunning.postValue(false)
        handler?.removeCallbacksAndMessages(null)
        handlerThread?.quitSafely()
        handlerThread = null
        resetSticks()
        Log.d(TAG, "ORB AutoPilot stopped")
    }

    private fun resetSticks() {
        VirtualStickManager.getInstance().leftStick.horizontalPosition = 0
        VirtualStickManager.getInstance().leftStick.verticalPosition = 0
        VirtualStickManager.getInstance().rightStick.horizontalPosition = 0
        VirtualStickManager.getInstance().rightStick.verticalPosition = 0
    }

    private val autoPilotTask = object : Runnable {
        override fun run() {
            if (isRunning.value != true) return

            try {
                val now = System.currentTimeMillis()
                val dt = (now - lastUpdateTime) / 1000.0
                lastUpdateTime = now

                // 1. Progress estimation
                val velocity = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAircraftVelocity))
                if (velocity != null) {
                    val speed = hypot(velocity.x, velocity.y)
                    currentCumDistance += speed * dt
                }
                currentProgress.postValue(if (totalPlanDistance > 0) currentCumDistance / totalPlanDistance else 0.0)

                // 2. Load images
                val refEntry = findRefEntry(currentCumDistance)
                val refBitmap = loadRefBitmap(refEntry?.imagePath)
                val currBitmap = getCurrentImageAsBitmap()

                if (currBitmap != null && refBitmap != null) {
                    // 3. ORB Processing
                    val sticks = calculateSticksWithOrb(currBitmap, refBitmap)
                    
                    // 4. Apply sticks
                    VirtualStickManager.getInstance().rightStick.verticalPosition = sticks.pitch
                    VirtualStickManager.getInstance().rightStick.horizontalPosition = sticks.roll
                    VirtualStickManager.getInstance().leftStick.horizontalPosition = sticks.yaw
                    
                    statusMessage.postValue("ORB Run: R:${sticks.roll} P:${sticks.pitch}")
                } else {
                    if (currBitmap == null) statusMessage.postValue("Wait: No Camera Frame")
                    if (refBitmap == null) statusMessage.postValue("Wait: No Ref Image")
                }

                currBitmap?.recycle()
                refBitmap?.recycle()

            } catch (e: Exception) {
                Log.e(TAG, "Loop error: $e")
            }

            if (isRunning.value == true) {
                handler?.postDelayed(this, INTERVAL_MS)
            }
        }
    }

    data class StickCommand(val roll: Int, val pitch: Int, val yaw: Int)

    private fun calculateSticksWithOrb(curr: Bitmap, ref: Bitmap): StickCommand {
        val matCurr = Mat()
        val matRef = Mat()
        Utils.bitmapToMat(curr, matCurr)
        Utils.bitmapToMat(ref, matRef)

        val keypointsCurr = MatOfKeyPoint()
        val keypointsRef = MatOfKeyPoint()
        val descCurr = Mat()
        val descRef = Mat()

        orb.detectAndCompute(matCurr, Mat(), keypointsCurr, descCurr)
        orb.detectAndCompute(matRef, Mat(), keypointsRef, descRef)

        if (descCurr.empty() || descRef.empty()) {
            matCurr.release()
            matRef.release()
            return StickCommand(0, PITCH_SPEED, 0)
        }

        val matches = MatOfDMatch()
        matcher.match(descCurr, descRef, matches)

        val matchesArray = matches.toArray()
        val goodMatches = matchesArray.sortedBy { it.distance }.take(MAX_MATCHES)
        if (goodMatches.size < 10) {
            matCurr.release()
            matRef.release()
            return StickCommand(0, PITCH_SPEED, 0)
        }

        // Calculate average displacement
        var sumDx = 0.0
        val kpCurrArray = keypointsCurr.toArray()
        val kpRefArray = keypointsRef.toArray()

        for (match in goodMatches) {
            val ptCurr = kpCurrArray[match.queryIdx].pt
            val ptRef = kpRefArray[match.trainIdx].pt
            sumDx += (ptCurr.x - ptRef.x)
        }

        val avgDx = sumDx / goodMatches.size
        val normDx = avgDx / curr.width 
        
        val rollInput = (-normDx * 1000 * ROLL_GAIN).toInt().coerceIn(-300, 300)
        
        matCurr.release()
        matRef.release()
        
        return StickCommand(rollInput, PITCH_SPEED, 0)
    }

    private fun findRefEntry(distance: Double): PlanEntry? {
        if (plan.isEmpty()) return null
        return plan.minByOrNull { Math.abs(it.cumDistanceM - distance) }
    }

    private fun loadRefBitmap(imagePath: String?): Bitmap? {
        if (imagePath == null || imageDir == null) return null
        val file = File(imageDir, imagePath)
        return if (file.exists()) BitmapFactory.decodeFile(file.absolutePath) else null
    }

    private fun getCurrentImageAsBitmap(): Bitmap? {
        val data = ImageLogger.getLatestFrameData() ?: return null
        val width = ImageLogger.getLatestFrameWidth()
        val height = ImageLogger.getLatestFrameHeight()
        if (width <= 0 || height <= 0) return null
        
        return try {
            val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
            val bytes = out.toByteArray()
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            null
        }
    }
}
