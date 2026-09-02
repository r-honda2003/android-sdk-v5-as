package dji.sampleV5.aircraft.models

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

/**
 * CHANGED: ImageLoggerは自分自身のタイミングでは保存しない。
 * カメラフレームが来るたびに「最新フレーム」をメモリ上に保持するだけにし、
 * 実際の保存はTelemetryLogger側の周期タイマーが saveLatestFrame() を呼んだ時にだけ行う。
 * これにより、画像とテレメトリが同じタイムスタンプ・同じタイミングで記録され、
 * 撮影時点から同期済みのデータセットになる（後からの時刻マッチングが不要になる）。
 */
object ImageLogger {

    private const val TAG = "ImageLogger"
    private const val JPEG_QUALITY = 90

    private data class FrameSnapshot(
        val data: ByteArray,
        val width: Int,
        val height: Int
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as FrameSnapshot

            if (!data.contentEquals(other.data)) return false
            if (width != other.width) return false
            if (height != other.height) return false

            return true
        }

        override fun hashCode(): Int {
            var result = data.contentHashCode()
            result = 31 * result + width
            result = 31 * result + height
            return result
        }
    }

    private var isRecording = false
    private var imageDir: File? = null
    private var activeCameraIndex: ComponentIndexType? = null

    @Volatile
    private var latestFrame: FrameSnapshot? = null

    fun getLatestFrameData(): ByteArray? = latestFrame?.data
    fun getLatestFrameWidth(): Int = latestFrame?.width ?: 0
    fun getLatestFrameHeight(): Int = latestFrame?.height ?: 0

    private val frameListener = object : ICameraStreamManager.CameraFrameListener {
        override fun onFrame(
            frameData: ByteArray,
            offset: Int,
            length: Int,
            width: Int,
            height: Int,
            format: ICameraStreamManager.FrameFormat
        ) {
            if (!isRecording) return
            try {
                val dataCopy = if (offset == 0 && length == frameData.size) {
                    frameData.copyOf()
                } else if (offset + length <= frameData.size) {
                    frameData.copyOfRange(offset, offset + length)
                } else {
                    frameData.copyOf(length)
                }
                latestFrame = FrameSnapshot(dataCopy, width, height)
            } catch (e: Exception) {
                Log.e(TAG, "Error in onFrame: $e")
            }
        }
    }

    private val availableCameraListener = object : ICameraStreamManager.AvailableCameraUpdatedListener {
        override fun onAvailableCameraUpdated(availableCameraList: List<ComponentIndexType>) {
            if (availableCameraList.isNotEmpty() && activeCameraIndex == null) {
                val cameraIndex = availableCameraList[0]
                activeCameraIndex = cameraIndex
                MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
                    cameraIndex,
                    ICameraStreamManager.FrameFormat.NV21,
                    frameListener
                )
                Log.d(TAG, "Frame listener attached to camera: $cameraIndex")
            } else if (availableCameraList.isEmpty()) {
                Log.e(TAG, "No available cameras detected.")
            }
        }

        override fun onCameraStreamEnableUpdate(cameraStreamEnableMap: MutableMap<ComponentIndexType, Boolean>) {
            Log.d(TAG, "Camera stream enable update: $cameraStreamEnableMap")
        }
    }

    fun startLogging(context: Context, sessionId: String) {
        if (isRecording) return
        isRecording = true
        latestFrame = null

        val dir = context.getExternalFilesDir(null)
        imageDir = File(dir, "images_$sessionId")
        imageDir?.mkdirs()

        MediaDataCenter.getInstance().cameraStreamManager
            .addAvailableCameraUpdatedListener(availableCameraListener)
    }

    fun stopLogging() {
        isRecording = false
        activeCameraIndex?.let {
            MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(frameListener)
        }
        MediaDataCenter.getInstance().cameraStreamManager
            .removeAvailableCameraUpdatedListener(availableCameraListener)
        activeCameraIndex = null
        latestFrame = null
    }

    /**
     * NEW: TelemetryLoggerの周期タイマーから呼ばれる。
     * その瞬間の最新フレームを、指定されたタイムスタンプ(=CSVのtime列と同じ値)をファイル名にして保存する。
     * @return 保存したファイル名（例: "1787730696776.jpg"）。まだ1枚もフレームを受信していない場合は null。
     */
    fun saveLatestFrame(timestamp: Long): String? {
        val frame = latestFrame ?: return null
        val dir = imageDir ?: return null
        return try {
            val yuvImage = YuvImage(frame.data, ImageFormat.NV21, frame.width, frame.height, null)
            val out = ByteArrayOutputStream()
            val ok = yuvImage.compressToJpeg(Rect(0, 0, frame.width, frame.height), JPEG_QUALITY, out)
            if (!ok) {
                Log.e(TAG, "compressToJpeg failed")
                return null
            }
            val fileName = "$timestamp.jpg"
            val file = File(dir, fileName)
            FileOutputStream(file).use { stream ->
                stream.write(out.toByteArray())
            }
            fileName
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save frame: $e")
            null
        }
    }
}