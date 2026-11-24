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

    // DAgGER用: 人間による緊急介入（プロポ操作割り込み）中かどうか
    @Volatile
    var isHumanOverrideActive: Boolean = false

    // AutoPilot / AI 飛行時の制御値・推論時間データ保持用
    @Volatile
    private var autoPilotSendLV: Int? = null
    @Volatile
    private var autoPilotSendLH: Int? = null
    @Volatile
    private var autoPilotSendRV: Int? = null
    @Volatile
    private var autoPilotSendRH: Int? = null
    @Volatile
    private var autoPilotImgFetchMs: Long = 0L
    @Volatile
    private var autoPilotPreprocMs: Long = 0L
    @Volatile
    private var autoPilotAiTotalMs: Long = 0L
    @Volatile
    private var autoPilotCmdSendMs: Long = 0L
    @Volatile
    private var autoPilotTurnaroundMs: Long = 0L

    fun updateAutoPilotState(
        sendLV: Int,
        sendLH: Int,
        sendRV: Int,
        sendRH: Int,
        imgFetchTimeMs: Long,
        preprocTimeMs: Long,
        inferenceTimeMs: Long,
        cmdSendTimeMs: Long,
        loopTimeMs: Long
    ) {
        autoPilotSendLV = sendLV
        autoPilotSendLH = sendLH
        autoPilotSendRV = sendRV
        autoPilotSendRH = sendRH
        autoPilotImgFetchMs = imgFetchTimeMs
        autoPilotPreprocMs = preprocTimeMs
        autoPilotAiTotalMs = inferenceTimeMs
        autoPilotCmdSendMs = cmdSendTimeMs
        autoPilotTurnaroundMs = loopTimeMs
    }

    fun clearAutoPilotState() {
        autoPilotSendLV = null
        autoPilotSendLH = null
        autoPilotSendRV = null
        autoPilotSendRH = null
        autoPilotImgFetchMs = 0L
        autoPilotPreprocMs = 0L
        autoPilotAiTotalMs = 0L
        autoPilotCmdSendMs = 0L
        autoPilotTurnaroundMs = 0L
    }

    private var lastYaw: Double? = null
    private var lastTimestamp: Long? = null
    private var frameCounter: Long = 0

    private val listenerId = this

    fun startLogging(context: Context, sessionId: String) {
        if (isRecording) {
            stopLogging()
        }
        isRecording = true
        isHumanOverrideActive = false

        lastYaw = null
        lastTimestamp = null
        frameCounter = 0

        val dir = context.getExternalFilesDir(null)
        val fileName = "telemetry_$sessionId.csv"
        val file = File(dir, fileName)
        fileWriter = FileWriter(file, true)
        fileWriter?.appendLine("timestamp,frame_id,version_note,wp_idx,total_wp,target_alt_m,repeat_image_name,image_size_kb,total_chunks,sp_kpts_curr,sp_kpts_teach,lg_matches,match_ratio_pct,inliers_cnt,inlier_ratio_pct,mean_match_score,pixel_error_px,t_img_recv_ms,t_sp_ms,t_lg_ms,t_depth_ms,t_ai_total_ms,t_cmd_send_ms,t_turnaround_ms,fps,network_mbps,network_kb_s,jitter_ms,cmd_vx,cmd_vy,cmd_vz,cmd_yaw_rate,cmd_pitch_660,cmd_roll_660,cmd_throttle_660,cmd_yaw_660,drone_pitch,drone_roll,drone_yaw,drone_alt_m,est_x_m,est_y_m,est_z_m,cpu_pct,gpu_pct,gpu_power_w,vram_mb,lat,lng,gps_signal,battery_percent,vx,vy,vz,stick_left_v,stick_left_h,stick_right_v,stick_right_h,human_override_active,calc_stick_left_v,calc_stick_left_h,calc_stick_right_v,calc_stick_right_h")

        subscribeKeys()
        startPeriodicWrite()
    }

    fun stopLogging() {
        if (!isRecording) return
        isRecording = false
        isHumanOverrideActive = false
        clearAutoPilotState()
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

                // C~H (vx, vy, vz, pitch, roll, yaw) から -660~660 の制御量を計算
                val maxVz = 3.0
                val maxVxy = 5.0
                val maxYawRate = 25.6

                // 1. calc_stick_left_v (スロットル/昇降): 上昇速度は -vz (NED座標系)
                val calcStickLV = ((-vz / maxVz) * 660.0).toInt().coerceIn(-660, 660)

                // 2. calc_stick_left_h (ヨー/旋回): yaw変化率から算出
                var calcStickLH = 0
                val prevY = lastYaw
                val prevT = lastTimestamp
                if (prevY != null && prevT != null) {
                    val dt = (now - prevT) / 1000.0
                    if (dt > 0.001) {
                        var dYaw = yaw - prevY
                        while (dYaw > 180.0) dYaw -= 360.0
                        while (dYaw < -180.0) dYaw += 360.0
                        val yawRate = dYaw / dt
                        calcStickLH = ((yawRate / maxYawRate) * 660.0).toInt().coerceIn(-660, 660)
                    }
                }
                lastYaw = yaw
                lastTimestamp = now

                // 3. calc_stick_right_v (ピッチ/前後): 北方向速度 vx から算出
                val calcStickRV = ((vx / maxVxy) * 660.0).toInt().coerceIn(-660, 660)

                // 4. calc_stick_right_h (ロール/左右): 東方向速度 vy から算出
                val calcStickRH = ((vy / maxVxy) * 660.0).toInt().coerceIn(-660, 660)

                // 5. V~Y列 (send_stick_*): ドローンへ実際に送信される制御値 (-660~660)
                val sendStickLV = autoPilotSendLV ?: (if (stickLV != 0) stickLV else calcStickLV)
                val sendStickLH = autoPilotSendLH ?: (if (stickLH != 0) stickLH else calcStickLH)
                val sendStickRV = autoPilotSendRV ?: (if (stickRV != 0) stickRV else calcStickRV)
                val sendStickRH = autoPilotSendRH ?: (if (stickRH != 0) stickRH else calcStickRH)

                val overrideFlag = if (isHumanOverrideActive) 1 else 0
                frameCounter++

                val datetimeStr = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", java.util.Locale.getDefault()).format(java.util.Date(now))
                val targetAltStr = String.format(java.util.Locale.US, "%.2f", alt)
                val pitchStr = String.format(java.util.Locale.US, "%.2f", pitch)
                val rollStr = String.format(java.util.Locale.US, "%.2f", roll)
                val yawStr = String.format(java.util.Locale.US, "%.2f", yaw)

                val tImgRecvStr = String.format(java.util.Locale.US, "%.1f", autoPilotImgFetchMs.toDouble())
                val tSpStr = String.format(java.util.Locale.US, "%.1f", autoPilotPreprocMs.toDouble())
                val tAiTotalStr = String.format(java.util.Locale.US, "%.1f", autoPilotAiTotalMs.toDouble())
                val tCmdSendStr = String.format(java.util.Locale.US, "%.1f", autoPilotCmdSendMs.toDouble())
                val tTurnaroundStr = String.format(java.util.Locale.US, "%.1f", autoPilotTurnaroundMs.toDouble())

                // Unified CSV 63-column layout (47 standard + 16 extra telemetry fields)
                fileWriter?.appendLine(
                    "$datetimeStr,$frameCounter,v5_sample,0,0,$targetAltStr,$imagePath,0.0,0,0,0,0,0.0,0,0.0,0.0,0.0,$tImgRecvStr,$tSpStr,0.0,0.0,$tAiTotalStr,$tCmdSendStr,$tTurnaroundStr,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,$sendStickRV,$sendStickRH,$sendStickLV,$sendStickLH,$pitchStr,$rollStr,$yawStr,$targetAltStr,0.0,0.0,0.0,0.0,0.0,0.0,0.0,$lat,$lng,$gps,$battery,$vx,$vy,$vz,$stickLV,$stickLH,$stickRV,$stickRH,$overrideFlag,$calcStickLV,$calcStickLH,$calcStickRV,$calcStickRH"
                )
                fileWriter?.flush()

                handler?.postDelayed(this, interval)
            }
        }
        handler?.post(writeTask)
    }
}