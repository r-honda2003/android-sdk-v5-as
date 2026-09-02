package dji.sampleV5.aircraft.models

import android.content.Context
import android.os.Handler
import android.os.Looper
import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.RemoteControllerKey
import dji.sdk.keyvalue.value.common.Attitude
import dji.sdk.keyvalue.value.common.LocationCoordinate3D
import dji.sdk.keyvalue.value.common.Velocity3D
import dji.sdk.keyvalue.value.flightcontroller.GPSSignalLevel
import dji.v5.manager.KeyManager
import java.io.File
import java.io.FileWriter

object TelemetryLogger {

    private var fileWriter: FileWriter? = null
    private var handler: Handler? = null
    private var isRecording = false

    private var velocity: Velocity3D? = null
    private var attitude: Attitude? = null
    private var location: LocationCoordinate3D? = null
    private var gpsSignal: GPSSignalLevel? = null
    private var batteryPercent: Int? = null
    // NEW: GPSに依存しない対地高度（超音波センサー）
    // KeyUltrasonicHeight は Int を返す。単位はdm(0.1m)であることを較正テストで確認済み
    private var ultrasonicHeight: Int? = null

    // NEW: リモコンのスティック入力（範囲: -660〜660）
    // 左スティック: 一般的に上下=高度(スロットル)、左右=旋回(ヨー)
    // 右スティック: 一般的に上下=前後(ピッチ)、左右=左右(ロール)  ※モード設定により対応が変わる場合あり
    private var stickLeftVertical: Int? = null
    private var stickLeftHorizontal: Int? = null
    private var stickRightVertical: Int? = null
    private var stickRightHorizontal: Int? = null

    private val listenerId = this

    fun startLogging(context: Context, sessionId: String) {
        if (isRecording) return
        isRecording = true

        val dir = context.getExternalFilesDir(null)
        val fileName = "telemetry_$sessionId.csv"
        val file = File(dir, fileName)
        fileWriter = FileWriter(file, true)
        fileWriter?.appendLine("time,image_path,vx,vy,vz,pitch,roll,yaw,lat,lng,alt,gps_signal,battery_percent,stick_left_v,stick_left_h,stick_right_v,stick_right_h")

        subscribeKeys()
        startPeriodicWrite()
    }

    fun stopLogging() {
        isRecording = false
        handler?.removeCallbacksAndMessages(null)
        unsubscribeKeys()
        fileWriter?.flush()
        fileWriter?.close()
        fileWriter = null
    }

    private fun subscribeKeys() {
        val velocityKey = KeyTools.createKey(FlightControllerKey.KeyAircraftVelocity)
        KeyManager.getInstance().listen(velocityKey, listenerId) { _, newValue ->
            velocity = newValue
            android.util.Log.d("Telemetry", "Velocity updated: $newValue")
        }

        val attitudeKey = KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude)
        KeyManager.getInstance().listen(attitudeKey, listenerId) { _, newValue ->
            attitude = newValue
            android.util.Log.d("Telemetry", "Attitude updated: $newValue")
        }

        val locationKey = KeyTools.createKey(FlightControllerKey.KeyAircraftLocation3D)
        KeyManager.getInstance().listen(locationKey, listenerId) { _, newValue ->
            location = newValue
            android.util.Log.d("Telemetry", "Location updated: $newValue")
        }

        val gpsKey = KeyTools.createKey(FlightControllerKey.KeyGPSSignalLevel)
        KeyManager.getInstance().listen(gpsKey, listenerId) { _, newValue ->
            gpsSignal = newValue
            android.util.Log.d("Telemetry", "GPS updated: $newValue")
        }

        val batteryKey = KeyTools.createKey(BatteryKey.KeyChargeRemainingInPercent)
        KeyManager.getInstance().listen(batteryKey, listenerId) { _, newValue ->
            batteryPercent = newValue
            android.util.Log.d("Telemetry", "Battery updated: $newValue")
        }

        // NEW: 超音波高度（GPS不要、屋内でも更新されるはず）
        val heightKey = KeyTools.createKey(FlightControllerKey.KeyUltrasonicHeight)
        KeyManager.getInstance().listen(heightKey, listenerId) { _, newValue ->
            ultrasonicHeight = newValue
            android.util.Log.d("Telemetry", "UltrasonicHeight updated: $newValue")
        }

        // NEW: リモコンのスティック入力（4軸）
        val stickLeftVKey = KeyTools.createKey(RemoteControllerKey.KeyStickLeftVertical)
        KeyManager.getInstance().listen(stickLeftVKey, listenerId) { _, newValue ->
            stickLeftVertical = newValue
            android.util.Log.d("Telemetry", "StickLeftVertical updated: $newValue")
        }

        val stickLeftHKey = KeyTools.createKey(RemoteControllerKey.KeyStickLeftHorizontal)
        KeyManager.getInstance().listen(stickLeftHKey, listenerId) { _, newValue ->
            stickLeftHorizontal = newValue
            android.util.Log.d("Telemetry", "StickLeftHorizontal updated: $newValue")
        }

        val stickRightVKey = KeyTools.createKey(RemoteControllerKey.KeyStickRightVertical)
        KeyManager.getInstance().listen(stickRightVKey, listenerId) { _, newValue ->
            stickRightVertical = newValue
            android.util.Log.d("Telemetry", "StickRightVertical updated: $newValue")
        }

        val stickRightHKey = KeyTools.createKey(RemoteControllerKey.KeyStickRightHorizontal)
        KeyManager.getInstance().listen(stickRightHKey, listenerId) { _, newValue ->
            stickRightHorizontal = newValue
            android.util.Log.d("Telemetry", "StickRightHorizontal updated: $newValue")
        }
    }

    private fun unsubscribeKeys() {
        KeyManager.getInstance().cancelListen(listenerId)
    }

    private fun startPeriodicWrite() {
        handler = Handler(Looper.getMainLooper())
        val interval = 200L

        val writeTask = object : Runnable {
            override fun run() {
                if (!isRecording) return

                val now = System.currentTimeMillis()
                // NEW: この瞬間の最新カメラフレームを、同じタイムスタンプで同時に保存する。
                // これにより画像とテレメトリの時刻が構造的に一致し、後から同期する必要がなくなる。
                val imagePath = ImageLogger.saveLatestFrame(now) ?: ""
                val vx = velocity?.x ?: 0.0
                val vy = velocity?.y ?: 0.0
                val vz = velocity?.z ?: 0.0
                val pitch = attitude?.pitch ?: 0.0
                val roll = attitude?.roll ?: 0.0
                val yaw = attitude?.yaw ?: 0.0
                val lat = location?.latitude ?: 0.0
                val lng = location?.longitude ?: 0.0
                // CHANGED: location.altitude (GPS依存) ではなく ultrasonicHeight (GPS非依存) を使う
                // 実測0.5m/1.0m/1.5mでの較正テストにより、単位はdm(0.1m)であることを確認済み(2026/08/22)
                val alt = (ultrasonicHeight ?: 0) / 10.0
                val gps = gpsSignal?.name ?: "UNKNOWN"
                val battery = batteryPercent ?: -1
                val stickLV = stickLeftVertical ?: 0
                val stickLH = stickLeftHorizontal ?: 0
                val stickRV = stickRightVertical ?: 0
                val stickRH = stickRightHorizontal ?: 0

                fileWriter?.appendLine(
                    "$now,$imagePath,$vx,$vy,$vz,$pitch,$roll,$yaw,$lat,$lng,$alt,$gps,$battery,$stickLV,$stickLH,$stickRV,$stickRH"
                )
                fileWriter?.flush()

                handler?.postDelayed(this, interval)
            }
        }
        handler?.post(writeTask)
    }
}