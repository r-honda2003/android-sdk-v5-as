package dji.sampleV5.aircraft.models

import android.content.Context
import android.os.Handler
import android.os.Looper
import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.Attitude
import dji.sdk.keyvalue.value.common.LocationCoordinate3D
import dji.sdk.keyvalue.value.common.Velocity3D
import dji.sdk.keyvalue.value.flightcontroller.GPSSignalLevel
import dji.v5.manager.KeyManager
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object TelemetryLogger {

    private var fileWriter: FileWriter? = null
    private var handler: Handler? = null
    private var isRecording = false

    private var velocity: Velocity3D? = null
    private var attitude: Attitude? = null
    private var location: LocationCoordinate3D? = null
    private var gpsSignal: GPSSignalLevel? = null
    private var batteryPercent: Int? = null

    private val listenerId = this

    fun startLogging(context: Context) {
        if (isRecording) return
        isRecording = true

        val dir = context.getExternalFilesDir(null)
        val fileName = "telemetry_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.csv"
        val file = File(dir, fileName)
        fileWriter = FileWriter(file, true)
        fileWriter?.appendLine("time,vx,vy,vz,pitch,roll,yaw,lat,lng,alt,gps_signal,battery_percent")

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
                val vx = velocity?.x ?: 0.0
                val vy = velocity?.y ?: 0.0
                val vz = velocity?.z ?: 0.0
                val pitch = attitude?.pitch ?: 0.0
                val roll = attitude?.roll ?: 0.0
                val yaw = attitude?.yaw ?: 0.0
                val lat = location?.latitude ?: 0.0
                val lng = location?.longitude ?: 0.0
                val alt = location?.altitude ?: 0.0
                val gps = gpsSignal?.name ?: "UNKNOWN"
                val battery = batteryPercent ?: -1

                fileWriter?.appendLine(
                    "$now,$vx,$vy,$vz,$pitch,$roll,$yaw,$lat,$lng,$alt,$gps,$battery"
                )
                fileWriter?.flush()

                handler?.postDelayed(this, interval)
            }
        }
        handler?.post(writeTask)
    }
}