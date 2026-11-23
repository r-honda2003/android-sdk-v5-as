package dji.sampleV5.aircraft.models

import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.RemoteControllerKey
import dji.sdk.keyvalue.value.flightcontroller.*
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.et.create
import dji.v5.et.listen
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickState
import dji.v5.manager.aircraft.virtualstick.VirtualStickStateListener
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.abs

/**
 * Class Description
 *
 * @author Hoker
 * @date 2021/6/18
 *
 * Copyright (c) 2021, DJI All Rights Reserved.
 */
class VirtualStickVM : DJIViewModel() {

    val currentSpeedLevel = MutableLiveData(0.0)
    var useRcStick = MutableLiveData(false)
    val currentVirtualStickStateInfo = MutableLiveData(VirtualStickStateInfo())

    val virtualStickAdvancedParam = MutableLiveData(VirtualStickFlightControlParam()).apply {
        value?.rollPitchCoordinateSystem = FlightCoordinateSystem.BODY
        value?.verticalControlMode = VerticalControlMode.VELOCITY
        value?.yawControlMode = YawControlMode.ANGULAR_VELOCITY
        value?.rollPitchControlMode = RollPitchControlMode.ANGLE
    }

    // RC Stick Value
    var stickValue = MutableLiveData(RCStickValue(0, 0, 0, 0))

    init {
        VirtualStickManager.getInstance().init()
        currentSpeedLevel.value = VirtualStickManager.getInstance().speedLevel
        VirtualStickManager.getInstance().setVirtualStickStateListener(object :
            VirtualStickStateListener {
            override fun onVirtualStickStateUpdate(stickState: VirtualStickState) {
                val current = currentVirtualStickStateInfo.value ?: VirtualStickStateInfo()
                currentVirtualStickStateInfo.postValue(current.copy(state = stickState))
            }

            override fun onChangeReasonUpdate(reason: FlightControlAuthorityChangeReason) {
                val current = currentVirtualStickStateInfo.value ?: VirtualStickStateInfo()
                currentVirtualStickStateInfo.postValue(current.copy(reason = reason))
            }
        })
    }

    fun enableVirtualStick(callback: CommonCallbacks.CompletionCallback) {
        try {
            VirtualStickManager.getInstance().init()
            VirtualStickManager.getInstance().enableVirtualStick(callback)
        } catch (e: Throwable) {
            dji.v5.utils.common.LogUtils.e("VirtualStickVM", "enableVirtualStick Exception: " + e.message)
            callback.onFailure(dji.v5.common.error.DJICommonError.FACTORY.build(e.message))
        }
    }

    private var altitudeControlJob: Job? = null

    fun climbToAltitude(targetAlt: Double = 1.7, callback: CommonCallbacks.CompletionCallback? = null) {
        enableVirtualStick(object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                startAltitudeControl(targetAlt, callback)
            }

            override fun onFailure(error: IDJIError) {
                callback?.onFailure(error)
            }
        })
    }

    fun stopAltitudeControl() {
        altitudeControlJob?.cancel()
        altitudeControlJob = null
        try {
            VirtualStickManager.getInstance().leftStick.verticalPosition = 0
        } catch (e: Throwable) {
            // ignore
        }
    }

    private fun startAltitudeControl(targetAlt: Double, callback: CommonCallbacks.CompletionCallback?) {
        altitudeControlJob?.cancel()
        altitudeControlJob = viewModelScope.launch(Dispatchers.IO) {
            val startTime = System.currentTimeMillis()
            val timeoutMs = 25000L // 25秒タイムアウト
            var success = false

            while (isActive) {
                val currentAlt = KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyAltitude)) ?: 0.0
                val diff = targetAlt - currentAlt

                // 到達判定 (|diff| < 0.05m = 5cm以内)
                if (abs(diff) < 0.05) {
                    VirtualStickManager.getInstance().leftStick.verticalPosition = 0
                    success = true
                    break
                }

                if (System.currentTimeMillis() - startTime > timeoutMs) {
                    VirtualStickManager.getInstance().leftStick.verticalPosition = 0
                    break
                }

                // P制御による垂直速度スティック入力の計算 (-660 ~ 660)
                val rawSpeed = (diff * 350.0).toInt()
                val speed = if (diff > 0) {
                    rawSpeed.coerceIn(120, 350) // 上昇 (最小120, 最大350)
                } else {
                    rawSpeed.coerceIn(-300, -100) // 下降
                }

                VirtualStickManager.getInstance().leftStick.verticalPosition = speed
                delay(100L)
            }

            VirtualStickManager.getInstance().leftStick.verticalPosition = 0
            // 高度上昇制御完了後、プロポ（RC）手動操作権を即座に復元するためVirtualStickを無効化
            disableVirtualStick(object : CommonCallbacks.CompletionCallback {
                override fun onSuccess() {
                    if (success) {
                        callback?.onSuccess()
                    } else {
                        callback?.onFailure(dji.v5.common.error.DJICommonError.FACTORY.build("Altitude control stopped or timed out"))
                    }
                }

                override fun onFailure(error: IDJIError) {
                    if (success) {
                        callback?.onSuccess()
                    } else {
                        callback?.onFailure(error)
                    }
                }
            })
        }
    }

    fun disableVirtualStick(callback: CommonCallbacks.CompletionCallback) {
        stopAltitudeControl()
        try {
            VirtualStickManager.getInstance().disableVirtualStick(callback)
        } catch (e: Throwable) {
            dji.v5.utils.common.LogUtils.e("VirtualStickVM", "disableVirtualStick Exception: " + e.message)
            callback.onFailure(dji.v5.common.error.DJICommonError.FACTORY.build(e.message))
        }
    }

    fun setSpeedLevel(speedLevel: Double) {
        VirtualStickManager.getInstance().speedLevel = speedLevel
        currentSpeedLevel.value = speedLevel
    }

    fun setLeftPosition(horizontal: Int, vertical: Int) {
        VirtualStickManager.getInstance().leftStick.horizontalPosition = horizontal
        VirtualStickManager.getInstance().leftStick.verticalPosition = vertical
    }

    fun setRightPosition(horizontal: Int, vertical: Int) {
        VirtualStickManager.getInstance().rightStick.horizontalPosition = horizontal
        VirtualStickManager.getInstance().rightStick.verticalPosition = vertical
    }

    fun sendVirtualStickAdvancedParam(param: VirtualStickFlightControlParam) {
        VirtualStickManager.getInstance().sendVirtualStickAdvancedParam(param)
    }

    fun disableVirtualStickAdvancedMode() {
        VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(false)
    }

    fun enableVirtualStickAdvancedMode() {
        VirtualStickManager.getInstance().setVirtualStickAdvancedModeEnabled(true)
    }

    fun listenRCStick() {
        RemoteControllerKey.KeyStickLeftHorizontal.create().listen(this) {
            it?.let {
                stickValue.value?.leftHorizontal = it
            }
            tryUpdateVirtualStickByRc()
        }
        RemoteControllerKey.KeyStickLeftVertical.create().listen(this) {
            it?.let {
                stickValue.value?.leftVertical = it
            }
            tryUpdateVirtualStickByRc()
        }
        RemoteControllerKey.KeyStickRightHorizontal.create().listen(this) {
            it?.let {
                stickValue.value?.rightHorizontal = it
            }
            tryUpdateVirtualStickByRc()
        }
        RemoteControllerKey.KeyStickRightVertical.create().listen(this) {
            it?.let {
                stickValue.value?.rightVertical = it
            }
            tryUpdateVirtualStickByRc()
        }
    }

    private fun tryUpdateVirtualStickByRc() {
        stickValue.postValue(stickValue.value)
        if (useRcStick.value == true) {
            stickValue.value?.apply {
                setLeftPosition(leftHorizontal, leftVertical)
                setRightPosition(rightHorizontal, rightVertical)
            }
        }
    }

    override fun onCleared() {
        stopAltitudeControl()
        KeyManager.getInstance().cancelListen(this)
        VirtualStickManager.getInstance().clearAllVirtualStickStateListener()
        VirtualStickManager.getInstance().destroy()
    }

    data class VirtualStickStateInfo(
        var state: VirtualStickState = VirtualStickState(false, FlightControlAuthority.UNKNOWN, false),
        var reason: FlightControlAuthorityChangeReason = FlightControlAuthorityChangeReason.UNKNOWN
    )

    data class RCStickValue(
        var leftHorizontal: Int, var leftVertical:
        Int, var rightHorizontal: Int, var rightVertical: Int
    ) {
        override fun toString(): String {
            return "leftHorizontal=$leftHorizontal,leftVertical=$leftVertical,\n" +
                    "rightHorizontal=$rightHorizontal,rightVertical=$rightVertical"
        }
    }
}