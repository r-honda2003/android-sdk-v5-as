package dji.sampleV5.aircraft.pages

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.activityViewModels
import dji.sampleV5.aircraft.databinding.FragVirtualStickPageBinding
import dji.sampleV5.aircraft.keyvalue.KeyValueDialogUtil
import dji.sampleV5.aircraft.models.BasicAircraftControlVM
import dji.sampleV5.aircraft.models.SimulatorVM
import dji.sampleV5.aircraft.models.VirtualStickVM
import dji.sampleV5.aircraft.models.MapMatchingRepeatVM
import dji.sampleV5.aircraft.util.Helper
import dji.v5.ux.core.util.ToastUtils
import dji.sampleV5.aircraft.virtualstick.OnScreenJoystick
import dji.sampleV5.aircraft.virtualstick.OnScreenJoystickListener
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.aircraft.virtualstick.Stick
import dji.v5.utils.common.JsonUtil
import kotlin.math.abs

/**
 * Class Description
 *
 * @author Hoker
 * @date 2021/5/11
 *
 * Copyright (c) 2021, DJI All Rights Reserved.
 */
class VirtualStickFragment : DJIFragment() {

    private val basicAircraftControlVM: BasicAircraftControlVM by activityViewModels()
    private val virtualStickVM: VirtualStickVM by activityViewModels()
    private val simulatorVM: SimulatorVM by activityViewModels()
    private val autoPilotVM: MapMatchingRepeatVM by activityViewModels()
    private var binding: FragVirtualStickPageBinding? = null
    private val deviation: Double = 0.02

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragVirtualStickPageBinding.inflate(inflater, container, false)
        return binding?.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding?.widgetHorizontalSituationIndicator?.setSimpleModeEnable(false)
        initBtnClickListener()
        initStickListener()
        virtualStickVM.listenRCStick()
        virtualStickVM.currentSpeedLevel.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        virtualStickVM.useRcStick.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        virtualStickVM.currentVirtualStickStateInfo.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        virtualStickVM.stickValue.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        virtualStickVM.virtualStickAdvancedParam.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }

        autoPilotVM.currentProgress.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        autoPilotVM.predictedSticks.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        autoPilotVM.statusMessage.observe(viewLifecycleOwner) {
            updateVirtualStickInfo()
        }
        simulatorVM.simulatorStateSb.observe(viewLifecycleOwner) {
            binding?.simulatorStateInfoTv?.text = it
        }

        // バックグラウンドで非同期にモデルと地図(plan.csv)を事前読み込み（TestingToolsActivity下の安全例外保護）
        try {
            val safeContext = context?.applicationContext ?: dji.v5.utils.common.ContextUtil.getContext()
            if (safeContext != null) {
                java.util.concurrent.Executors.newSingleThreadExecutor().execute {
                    try {
                        autoPilotVM.initAutoPilot(safeContext, "images_test1")
                    } catch (e: Throwable) {
                        mainHandler.post {
                            ToastUtils.showToast("Init Error: ${e.message}")
                        }
                    }
                }
            }
        } catch (e: Throwable) {
            ToastUtils.showToast("Context Error: ${e.message}")
        }
    }

    private fun initBtnClickListener() {
        binding?.btnEnableVirtualStick?.setOnClickListener {
            try {
                virtualStickVM.enableVirtualStick(object : CommonCallbacks.CompletionCallback {
                    override fun onSuccess() {
                        mainHandler.post {
                            ToastUtils.showToast("enableVirtualStick success.")
                        }
                    }

                    override fun onFailure(error: IDJIError) {
                        mainHandler.post {
                            ToastUtils.showToast("enableVirtualStick error,$error")
                        }
                    }
                })
            } catch (e: Throwable) {
                ToastUtils.showToast("Exception: ${e.message}")
            }
        }
        binding?.btnDisableVirtualStick?.setOnClickListener {
            try {
                virtualStickVM.disableVirtualStick(object : CommonCallbacks.CompletionCallback {
                    override fun onSuccess() {
                        mainHandler.post {
                            ToastUtils.showToast("disableVirtualStick success.")
                        }
                    }

                    override fun onFailure(error: IDJIError) {
                        mainHandler.post {
                            ToastUtils.showToast("disableVirtualStick error,$error")
                        }
                    }
                })
            } catch (e: Throwable) {
                ToastUtils.showToast("Exception: ${e.message}")
            }
        }
        binding?.btnSetVirtualStickSpeedLevel?.setOnClickListener {
            val speedLevels = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
            initPopupNumberPicker(Helper.makeList(speedLevels)) {
                val chosenSpeed = speedLevels[indexChosen[0]]
                virtualStickVM.setSpeedLevel(chosenSpeed)
                autoPilotVM.setRepeatSpeedScale(chosenSpeed)
                ToastUtils.showToast("Repeat Speed set to ${(chosenSpeed * 100).toInt()}%")
                resetIndex()
            }
        }
        binding?.btnTakeOff?.setOnClickListener {
            basicAircraftControlVM.startTakeOff(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) {
                    ToastUtils.showToast("start takeOff onSuccess.")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("start takeOff onFailure,$error")
                }
            })
        }
        binding?.btnLanding?.setOnClickListener {
            basicAircraftControlVM.startLanding(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) {
                    ToastUtils.showToast("start landing onSuccess.")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("start landing onFailure,$error")
                }
            })
        }

        binding?.btnEnableAutoPilot?.setOnClickListener {
            try {
                // 自動で VirtualStick 制御権を獲得してから AutoPilot を起動（一発全自動）
                virtualStickVM.enableVirtualStick(object : CommonCallbacks.CompletionCallback {
                    override fun onSuccess() {
                        mainHandler.post {
                            autoPilotVM.startAutoPilot()
                            ToastUtils.showToast("AutoPilot Started & VirtualStick ON")
                        }
                    }

                    override fun onFailure(error: IDJIError) {
                        mainHandler.post {
                            ToastUtils.showToast("Enable VirtualStick Failed: $error")
                        }
                    }
                })
            } catch (e: Throwable) {
                ToastUtils.showToast("AutoPilot Error: ${e.message}")
            }
        }

        binding?.btnDisableAutoPilot?.setOnClickListener {
            try {
                autoPilotVM.stopAutoPilot()
                ToastUtils.showToast("AutoPilot Disabled")
            } catch (e: Throwable) {
                ToastUtils.showToast("AutoPilot Stop Error: ${e.message}")
            }
        }
        binding?.btnUseRcStick?.setOnClickListener {
            virtualStickVM.useRcStick.value = virtualStickVM.useRcStick.value != true
            if (virtualStickVM.useRcStick.value == true) {
                ToastUtils.showToast(
                    "After it is turned on," +
                            "the joystick value of the RC will be used as the left/ right stick value"
                )
            }
        }
        binding?.btnSetVirtualStickAdvancedParam?.setOnClickListener {
            KeyValueDialogUtil.showInputDialog(
                activity, "Set Virtual Stick Advanced Param",
                JsonUtil.toJson(virtualStickVM.virtualStickAdvancedParam.value), "", false
            ) {
                it?.apply {
                    val param = JsonUtil.toBean(this, VirtualStickFlightControlParam::class.java)
                    if (param == null) {
                        ToastUtils.showToast("Value Parse Error")
                        return@showInputDialog
                    }
                    virtualStickVM.virtualStickAdvancedParam.postValue(param)
                }
            }
        }
        binding?.btnSendVirtualStickAdvancedParam?.setOnClickListener {
            virtualStickVM.virtualStickAdvancedParam.value?.let {
                virtualStickVM.sendVirtualStickAdvancedParam(it)
            }
        }
        binding?.btnEnableVirtualStickAdvancedMode?.setOnClickListener {
            virtualStickVM.enableVirtualStickAdvancedMode()
        }
        binding?.btnDisableVirtualStickAdvancedMode?.setOnClickListener {
            virtualStickVM.disableVirtualStickAdvancedMode()
        }
    }

    private fun initStickListener() {
        binding?.leftStickView?.setJoystickListener(object : OnScreenJoystickListener {
            override fun onTouch(joystick: OnScreenJoystick?, pX: Float, pY: Float) {
                var leftPx = 0F
                var leftPy = 0F

                if (abs(pX) >= deviation) {
                    leftPx = pX
                }

                if (abs(pY) >= deviation) {
                    leftPy = pY
                }

                virtualStickVM.setLeftPosition(
                    (leftPx * Stick.MAX_STICK_POSITION_ABS).toInt(),
                    (leftPy * Stick.MAX_STICK_POSITION_ABS).toInt()
                )
            }
        })
        binding?.rightStickView?.setJoystickListener(object : OnScreenJoystickListener {
            override fun onTouch(joystick: OnScreenJoystick?, pX: Float, pY: Float) {
                var rightPx = 0F
                var rightPy = 0F

                if (abs(pX) >= deviation) {
                    rightPx = pX
                }

                if (abs(pY) >= deviation) {
                    rightPy = pY
                }

                virtualStickVM.setRightPosition(
                    (rightPx * Stick.MAX_STICK_POSITION_ABS).toInt(),
                    (rightPy * Stick.MAX_STICK_POSITION_ABS).toInt()
                )
            }
        })
    }

    private fun updateVirtualStickInfo() {
        val builder = StringBuilder()
        builder.append("=== AutoPilot Monitor ===").append("\n")
        builder.append("Status: ").append(autoPilotVM.statusMessage.value ?: "Ready").append("\n")
        builder.append("Progress: ").append(String.format(java.util.Locale.US, "%.1f%%", (autoPilotVM.currentProgress.value ?: 0.0) * 100)).append("\n")
        val sticks = autoPilotVM.predictedSticks.value
        if (sticks != null && sticks.size == 4) {
            builder.append("Predicted Sticks: LV:${sticks[0]} LH:${sticks[1]} RV:${sticks[2]} RH:${sticks[3]}").append("\n")
        }
        builder.append("\n=== VirtualStick System ===\n")
        builder.append("Speed level:").append(virtualStickVM.currentSpeedLevel.value).append("\n")
        builder.append("Use rc stick as virtual stick:").append(virtualStickVM.useRcStick.value).append("\n")
        builder.append("Is virtual stick enable:").append(virtualStickVM.currentVirtualStickStateInfo.value?.state?.isVirtualStickEnable).append("\n")
        builder.append("Current control permission owner:").append(virtualStickVM.currentVirtualStickStateInfo.value?.state?.currentFlightControlAuthorityOwner).append("\n")
        builder.append("Change reason:").append(virtualStickVM.currentVirtualStickStateInfo.value?.reason).append("\n")
        builder.append("Rc stick value:").append(virtualStickVM.stickValue.value?.toString()).append("\n")

        mainHandler.post {
            binding?.virtualStickInfoTv?.text = builder.toString()
        }
    }
}