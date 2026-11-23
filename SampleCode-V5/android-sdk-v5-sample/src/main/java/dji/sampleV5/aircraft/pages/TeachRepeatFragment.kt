package dji.sampleV5.aircraft.pages

import android.os.Bundle
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.activityViewModels
import dji.sampleV5.aircraft.databinding.FragTeachRepeatPageBinding
import dji.sampleV5.aircraft.models.BasicAircraftControlVM
import dji.sampleV5.aircraft.models.MapMatchingRepeatVM
import dji.sampleV5.aircraft.models.TelemetryLogger
import dji.sampleV5.aircraft.models.VirtualStickVM
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.RemoteControllerKey
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.KeyManager
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import dji.v5.ux.core.util.ToastUtils

class TeachRepeatFragment : DJIFragment() {

    private var binding: FragTeachRepeatPageBinding? = null
    private val basicAircraftControlVM: BasicAircraftControlVM by activityViewModels()
    private val virtualStickVM: VirtualStickVM by activityViewModels()
    private val autoPilotVM: MapMatchingRepeatVM by activityViewModels()

    private var sessionIndex = 1
    private var isTeachRecording = false

    private var surface: Surface? = null
    private var surfaceWidth = -1
    private var surfaceHeight = -1
    private var activeCameraIndex: ComponentIndexType? = null

    private val cameraSurfaceCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) {
            surface = holder.surface
            attachCameraStream()
        }

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            surfaceWidth = width
            surfaceHeight = height
            attachCameraStream()
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            surface = null
            detachCameraStream()
        }
    }

    private val availableCameraListener = object : ICameraStreamManager.AvailableCameraUpdatedListener {
        override fun onAvailableCameraUpdated(availableCameraList: List<ComponentIndexType>) {
            if (availableCameraList.isNotEmpty() && activeCameraIndex == null) {
                activeCameraIndex = availableCameraList[0]
                attachCameraStream()
            }
        }

        override fun onCameraStreamEnableUpdate(cameraStreamEnableMap: MutableMap<ComponentIndexType, Boolean>) {
            // ignore
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragTeachRepeatPageBinding.inflate(inflater, container, false)
        return binding?.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding?.svCameraStream?.holder?.addCallback(cameraSurfaceCallback)

        MediaDataCenter.getInstance().cameraStreamManager
            .addAvailableCameraUpdatedListener(availableCameraListener)

        initBtnClickListener()
        subscribeRcSticks()

        autoPilotVM.statusMessage.observe(viewLifecycleOwner) { msg ->
            binding?.tvRepeatStatus?.text = msg
        }

        val appContext = requireContext().applicationContext
        java.util.concurrent.Executors.newSingleThreadExecutor().execute {
            autoPilotVM.initAutoPilot(appContext, "images_run1")
        }
    }

    private fun initBtnClickListener() {
        // Top Bar Back Button
        binding?.btnBack?.setOnClickListener {
            requireActivity().onBackPressed()
        }

        // Backup Files
        binding?.btnBackupFiles?.setOnClickListener {
            val dir = requireContext().getExternalFilesDir(null)
            val path = dir?.absolutePath ?: "N/A"
            android.app.AlertDialog.Builder(requireContext())
                .setTitle("ログ・画像バックアップ保存先")
                .setMessage("データ保存先フォルダ:\n$path\n\nPC接続時にこのフォルダ内の telemetry_*.csv および images_* フォルダをコピーしてください。")
                .setPositiveButton("OK", null)
                .show()
        }

        // Take Off
        binding?.btnTakeOff?.setOnClickListener {
            basicAircraftControlVM.startTakeOff(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) {
                    ToastUtils.showToast("離陸命令送信 成功")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("離陸失敗: ${error.description()}")
                }
            })
        }

        // Auto Landing
        binding?.btnAutoLanding?.setOnClickListener {
            basicAircraftControlVM.startLanding(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) {
                    ToastUtils.showToast("自動着陸命令送信 成功")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("自動着陸失敗: ${error.description()}")
                }
            })
        }

        // Climb to 1.7m
        binding?.btnClimbToTargetAltitude?.setOnClickListener {
            ToastUtils.showToast("指定高度 (1.7m) へ上昇制御を開始します...")
            virtualStickVM.climbToAltitude(1.7, object : CommonCallbacks.CompletionCallback {
                override fun onSuccess() {
                    mainHandler.post {
                        ToastUtils.showToast("指定高度 (1.7m) に到達しました！")
                    }
                }

                override fun onFailure(error: IDJIError) {
                    mainHandler.post {
                        ToastUtils.showToast("指定高度上昇完了/停止: ${error.description()}")
                    }
                }
            })
        }

        binding?.btnClimbToTargetAltitude?.setOnLongClickListener {
            val builder = android.app.AlertDialog.Builder(requireContext())
            builder.setTitle("目標高度の設定 (m)")
            val input = android.widget.EditText(requireContext())
            input.setText("1.7")
            input.setSelection(input.text.length)
            input.inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            builder.setView(input)
            builder.setPositiveButton("上昇開始") { _, _ ->
                val altStr = input.text.toString().trim()
                val targetAlt = altStr.toDoubleOrNull() ?: 1.7
                ToastUtils.showToast("指定高度 (${targetAlt}m) へ上昇制御を開始します...")
                virtualStickVM.climbToAltitude(targetAlt, object : CommonCallbacks.CompletionCallback {
                    override fun onSuccess() {
                        mainHandler.post {
                            ToastUtils.showToast("指定高度 (${targetAlt}m) に到達しました！")
                        }
                    }

                    override fun onFailure(error: IDJIError) {
                        mainHandler.post {
                            ToastUtils.showToast("高度制御完了/停止: ${error.description()}")
                        }
                    }
                })
            }
            builder.setNegativeButton("キャンセル", null)
            builder.show()
            true
        }

        // 1. Teach Mode
        binding?.btnStartTeach?.setOnClickListener {
            // プロポ（RC）物理スティックによる手動飛行権限を確定するため、VirtualStickを確実に無効化
            virtualStickVM.disableVirtualStick(object : CommonCallbacks.CompletionCallback {
                override fun onSuccess() {}
                override fun onFailure(error: IDJIError) {}
            })
            val dateStr = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.US).format(java.util.Date())
            val sessionId = "${dateStr}_run1_session_$sessionIndex"
            TelemetryLogger.startLogging(requireContext(), sessionId)
            dji.sampleV5.aircraft.models.ImageLogger.startLogging(requireContext(), sessionId)
            isTeachRecording = true
            binding?.tvTeachStatus?.text = "Teach Recording: Running (Session $sessionIndex/5)\nData saving to: $sessionId"
            ToastUtils.showToast("Teach Mode 録画開始 (Session $sessionIndex/5) - プロポ操作有効")
        }

        binding?.btnStopTeach?.setOnClickListener {
            TelemetryLogger.stopLogging()
            dji.sampleV5.aircraft.models.ImageLogger.stopLogging()
            isTeachRecording = false
            ToastUtils.showToast("Teach Mode 録画停止 (Session $sessionIndex/5)")
            if (sessionIndex < 5) {
                sessionIndex++
            }
            binding?.tvTeachStatus?.text = "Teach Recording: Stopped\nNext Session: $sessionIndex/5"
        }

        // 2. Repeat Mode
        binding?.btnStartRepeat?.setOnClickListener {
            val ip = binding?.etServerIp?.text?.toString()?.trim() ?: "192.168.12.60"
            if (ip.isNotEmpty()) {
                autoPilotVM.pcServerIp = ip
            }
            autoPilotVM.isRcOverrideEnabled = true
            autoPilotVM.isPcInferenceMode = true
            autoPilotVM.startAutoPilot(requireContext().applicationContext)
            ToastUtils.showToast("Repeat Mode 起動 (PC: ${autoPilotVM.pcServerIp})")
        }

        binding?.btnStopRepeat?.setOnClickListener {
            autoPilotVM.stopAutoPilot()
            TelemetryLogger.stopLogging()
            ToastUtils.showToast("Repeat Mode 停止")
            binding?.tvRepeatStatus?.text = "Repeat Flight: Stopped\nServer: ${autoPilotVM.pcServerIp}"
        }
    }

    private var rcRoll = 0
    private var rcPitch = 0
    private var rcYaw = 0
    private var rcThrottle = 0
    private var currentAltitude = 0.0

    private fun subscribeRcSticks() {
        val listenerId = this

        val altitudeKey = KeyTools.createKey(FlightControllerKey.KeyAltitude)
        KeyManager.getInstance().listen(altitudeKey, listenerId) { _, valAlt ->
            currentAltitude = valAlt ?: 0.0
            mainHandler.post {
                binding?.tvAircraftAltitude?.text = "現在高度 (Altitude): ${String.format(java.util.Locale.US, "%.2f", currentAltitude)} m"
            }
        }

        val stickLH = KeyTools.createKey(RemoteControllerKey.KeyStickLeftHorizontal)
        KeyManager.getInstance().listen(stickLH, listenerId) { _, valH ->
            rcYaw = valH ?: 0
            updateRcStatusText()
        }

        val stickLV = KeyTools.createKey(RemoteControllerKey.KeyStickLeftVertical)
        KeyManager.getInstance().listen(stickLV, listenerId) { _, valV ->
            rcThrottle = valV ?: 0
            updateRcStatusText()
        }

        val stickRH = KeyTools.createKey(RemoteControllerKey.KeyStickRightHorizontal)
        KeyManager.getInstance().listen(stickRH, listenerId) { _, valRH ->
            rcRoll = valRH ?: 0
            updateRcStatusText()
        }

        val stickRV = KeyTools.createKey(RemoteControllerKey.KeyStickRightVertical)
        KeyManager.getInstance().listen(stickRV, listenerId) { _, valRV ->
            rcPitch = valRV ?: 0
            updateRcStatusText()
        }
    }

    private fun updateRcStatusText() {
        mainHandler.post {
            binding?.tvRcStickStatus?.text = "Roll: $rcRoll | Pitch: $rcPitch | Yaw: $rcYaw | Throttle: $rcThrottle"
        }
    }

    private fun attachCameraStream() {
        val surf = surface ?: return
        val camIndex = activeCameraIndex ?: ComponentIndexType.LEFT_OR_MAIN
        try {
            MediaDataCenter.getInstance().cameraStreamManager
                .putCameraStreamSurface(camIndex, surf, surfaceWidth, surfaceHeight, ICameraStreamManager.ScaleType.CENTER_INSIDE)
        } catch (e: Throwable) {
            // ignore
        }
    }

    private fun detachCameraStream() {
        val surf = surface ?: return
        try {
            MediaDataCenter.getInstance().cameraStreamManager
                .removeCameraStreamSurface(surf)
        } catch (e: Throwable) {
            // ignore
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        binding?.svCameraStream?.holder?.removeCallback(cameraSurfaceCallback)
        MediaDataCenter.getInstance().cameraStreamManager
            .removeAvailableCameraUpdatedListener(availableCameraListener)
        KeyManager.getInstance().cancelListen(this)
        detachCameraStream()
    }
}