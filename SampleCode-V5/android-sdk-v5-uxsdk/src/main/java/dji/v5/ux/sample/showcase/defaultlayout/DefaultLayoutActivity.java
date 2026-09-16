/*
 * Copyright (c) 2018-2020 DJI
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package dji.v5.ux.sample.showcase.defaultlayout;

import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Locale;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import dji.v5.manager.datacenter.MediaDataCenter;
import dji.v5.manager.interfaces.ICameraStreamManager;

import dji.v5.manager.aircraft.virtualstick.VirtualStickManager;
import dji.v5.manager.aircraft.virtualstick.VirtualStickState;
import dji.v5.manager.aircraft.virtualstick.VirtualStickStateListener;
import dji.sdk.keyvalue.value.flightcontroller.FlightControlAuthorityChangeReason;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.ux.core.util.ToastUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.lifecycle.ViewModel;
import androidx.lifecycle.ViewModelProvider;

import dji.sdk.keyvalue.value.common.CameraLensType;
import dji.sdk.keyvalue.value.common.ComponentIndexType;
import dji.v5.manager.datacenter.MediaDataCenter;
import dji.v5.manager.interfaces.ICameraStreamManager;
import dji.v5.network.DJINetworkManager;
import dji.v5.network.IDJINetworkStatusListener;
import dji.v5.utils.common.JsonUtil;
import dji.v5.utils.common.LogPath;
import dji.v5.utils.common.LogUtils;
import dji.v5.ux.R;
import dji.v5.ux.accessory.RTKStartServiceHelper;
import dji.v5.ux.cameracore.widget.autoexposurelock.AutoExposureLockWidget;
import dji.v5.ux.cameracore.widget.cameracontrols.CameraControlsWidget;
import dji.v5.ux.cameracore.widget.cameracontrols.lenscontrol.LensControlWidget;
import dji.v5.ux.cameracore.widget.focusexposureswitch.FocusExposureSwitchWidget;
import dji.v5.ux.cameracore.widget.focusmode.FocusModeWidget;
import dji.v5.ux.cameracore.widget.fpvinteraction.FPVInteractionWidget;
import dji.v5.ux.core.base.SchedulerProvider;
import dji.v5.ux.core.communication.BroadcastValues;
import dji.v5.ux.core.communication.GlobalPreferenceKeys;
import dji.v5.ux.core.communication.ObservableInMemoryKeyedStore;
import dji.v5.ux.core.communication.UXKeys;
import dji.v5.ux.core.extension.ViewExtensions;
import dji.v5.ux.core.panel.systemstatus.SystemStatusListPanelWidget;
import dji.v5.ux.core.panel.topbar.TopBarPanelWidget;
import dji.v5.ux.core.util.CameraUtil;
import dji.v5.ux.core.util.DataProcessor;
import dji.v5.ux.core.util.ViewUtil;
import dji.v5.ux.core.widget.fpv.FPVWidget;
import dji.v5.ux.core.widget.hsi.HorizontalSituationIndicatorWidget;
import dji.v5.ux.core.widget.hsi.PrimaryFlightDisplayWidget;
import dji.v5.ux.core.widget.setting.SettingWidget;
import dji.v5.ux.core.widget.simulator.SimulatorIndicatorWidget;
import dji.v5.ux.core.widget.systemstatus.SystemStatusWidget;
import dji.v5.ux.gimbal.GimbalFineTuneWidget;
import dji.v5.ux.map.MapWidget;
import dji.v5.ux.mapkit.core.maps.DJIUiSettings;
import dji.v5.ux.training.simulatorcontrol.SimulatorControlWidget;
import dji.v5.ux.visualcamera.CameraNDVIPanelWidget;
import dji.v5.ux.visualcamera.CameraVisiblePanelWidget;
import dji.v5.ux.visualcamera.zoom.FocalZoomWidget;
import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers;
import io.reactivex.rxjava3.disposables.CompositeDisposable;

/**
 * Displays a sample layout of widgets similar to that of the various DJI apps.
 */
public class DefaultLayoutActivity extends AppCompatActivity {

    //region Fields
    private final String TAG = LogUtils.getTag(this);

    protected FPVWidget primaryFpvWidget;
    protected FPVInteractionWidget fpvInteractionWidget;
    protected FPVWidget secondaryFPVWidget;
    protected SystemStatusListPanelWidget systemStatusListPanelWidget;
    protected SimulatorControlWidget simulatorControlWidget;
    protected LensControlWidget lensControlWidget;
    protected AutoExposureLockWidget autoExposureLockWidget;
    protected FocusModeWidget focusModeWidget;
    protected FocusExposureSwitchWidget focusExposureSwitchWidget;
    protected CameraControlsWidget cameraControlsWidget;
    protected HorizontalSituationIndicatorWidget horizontalSituationIndicatorWidget;
    protected PrimaryFlightDisplayWidget pfvFlightDisplayWidget;
    protected CameraNDVIPanelWidget ndviCameraPanel;
    protected CameraVisiblePanelWidget visualCameraPanel;
    protected FocalZoomWidget focalZoomWidget;
    protected SettingWidget settingWidget;
    protected MapWidget mapWidget;
    protected TopBarPanelWidget topBarPanel;
    protected ConstraintLayout fpvParentView;
    private DrawerLayout mDrawerLayout;
    private TextView gimbalAdjustDone;
    private GimbalFineTuneWidget gimbalFineTuneWidget;
    private ComponentIndexType lastDevicePosition = ComponentIndexType.UNKNOWN;
    private CameraLensType lastLensType = CameraLensType.UNKNOWN;


    private CompositeDisposable compositeDisposable;
    private final DataProcessor<CameraSource> cameraSourceProcessor = DataProcessor.create(new CameraSource(ComponentIndexType.UNKNOWN,
            CameraLensType.UNKNOWN));
    private final IDJINetworkStatusListener networkStatusListener = isNetworkAvailable -> {
        if (isNetworkAvailable) {
            LogUtils.d(TAG, "isNetworkAvailable=" + true);
            RTKStartServiceHelper.INSTANCE.startRtkService(false);
        }
    };
    private final ICameraStreamManager.AvailableCameraUpdatedListener availableCameraUpdatedListener = new ICameraStreamManager.AvailableCameraUpdatedListener() {
        @Override
        public void onAvailableCameraUpdated(@NonNull List<ComponentIndexType> availableCameraList) {
            runOnUiThread(() -> updateFPVWidgetSource(availableCameraList));
        }

        @Override
        public void onCameraStreamEnableUpdate(@NonNull Map<ComponentIndexType, Boolean> cameraStreamEnableMap) {
            //
        }
    };

    //endregion

    //region Lifecycle
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.uxsdk_activity_default_layout);
        fpvParentView = findViewById(R.id.fpv_holder);
        mDrawerLayout = findViewById(R.id.root_view);
        topBarPanel = findViewById(R.id.panel_top_bar);
        settingWidget = topBarPanel.getSettingWidget();
        primaryFpvWidget = findViewById(R.id.widget_primary_fpv);
        fpvInteractionWidget = findViewById(R.id.widget_fpv_interaction);
        secondaryFPVWidget = findViewById(R.id.widget_secondary_fpv);
        systemStatusListPanelWidget = findViewById(R.id.widget_panel_system_status_list);
        simulatorControlWidget = findViewById(R.id.widget_simulator_control);
        lensControlWidget = findViewById(R.id.widget_lens_control);
        ndviCameraPanel = findViewById(R.id.panel_ndvi_camera);
        visualCameraPanel = findViewById(R.id.panel_visual_camera);
        autoExposureLockWidget = findViewById(R.id.widget_auto_exposure_lock);
        focusModeWidget = findViewById(R.id.widget_focus_mode);
        focusExposureSwitchWidget = findViewById(R.id.widget_focus_exposure_switch);
        pfvFlightDisplayWidget = findViewById(R.id.widget_fpv_flight_display_widget);
        focalZoomWidget = findViewById(R.id.widget_focal_zoom);
        cameraControlsWidget = findViewById(R.id.widget_camera_controls);
        horizontalSituationIndicatorWidget = findViewById(R.id.widget_horizontal_situation_indicator);
        gimbalAdjustDone = findViewById(R.id.fpv_gimbal_ok_btn);
        gimbalFineTuneWidget = findViewById(R.id.setting_menu_gimbal_fine_tune);
        mapWidget = findViewById(R.id.widget_map);

        initClickListener();
        MediaDataCenter.getInstance().getCameraStreamManager().addAvailableCameraUpdatedListener(availableCameraUpdatedListener);
        primaryFpvWidget.setOnFPVStreamSourceListener((devicePosition, lensType) -> cameraSourceProcessor.onNext(new CameraSource(devicePosition, lensType)));

        //小surfaceView放置在顶部，避免被大的遮挡
        secondaryFPVWidget.setSurfaceViewZOrderOnTop(true);
        secondaryFPVWidget.setSurfaceViewZOrderMediaOverlay(true);


        mapWidget.initMapLibreMap(getApplicationContext(), map -> {
            DJIUiSettings uiSetting = map.getUiSettings();
            if (uiSetting != null) {
                uiSetting.setZoomControlsEnabled(false);//hide zoom widget
            }
        });
        mapWidget.onCreate(savedInstanceState);
        getWindow().setBackgroundDrawable(new ColorDrawable(Color.BLACK));

        //实现RTK监测网络，并自动重连机制
        DJINetworkManager.getInstance().addNetworkStatusListener(networkStatusListener);

        initVirtualStickControls();
    }

    private void isGimableAdjustClicked(BroadcastValues broadcastValues) {
        if (mDrawerLayout.isDrawerOpen(GravityCompat.END)) {
            mDrawerLayout.closeDrawers();
        }
        horizontalSituationIndicatorWidget.setVisibility(View.GONE);
        if (gimbalFineTuneWidget != null) {
            gimbalFineTuneWidget.setVisibility(View.VISIBLE);
        }
    }

    private void initClickListener() {
        secondaryFPVWidget.setOnClickListener(v -> swapVideoSource());

        if (settingWidget != null) {
            settingWidget.setOnClickListener(v -> toggleRightDrawer());
        }

        // Setup top bar state callbacks
        SystemStatusWidget systemStatusWidget = topBarPanel.getSystemStatusWidget();
        if (systemStatusWidget != null) {
            systemStatusWidget.setOnClickListener(v -> ViewExtensions.toggleVisibility(systemStatusListPanelWidget));
        }

        SimulatorIndicatorWidget simulatorIndicatorWidget = topBarPanel.getSimulatorIndicatorWidget();
        if (simulatorIndicatorWidget != null) {
            simulatorIndicatorWidget.setOnClickListener(v -> ViewExtensions.toggleVisibility(simulatorControlWidget));
        }
        if (gimbalAdjustDone != null) {
            gimbalAdjustDone.setOnClickListener(view -> {
                horizontalSituationIndicatorWidget.setVisibility(View.VISIBLE);
                if (gimbalFineTuneWidget != null) {
                    gimbalFineTuneWidget.setVisibility(View.GONE);
                }
            });
        }
    }

    private void toggleRightDrawer() {
        mDrawerLayout.openDrawer(GravityCompat.END);
    }

    protected Button btnVirtualStickToggle;
    protected Button btnAutoPilotToggle;
    protected TextView tvVsStatus;
    protected boolean isVirtualStickEnabled = false;

    protected boolean isAutoPilotRunning = false;
    protected boolean useOrbMode = false; // false: AI (2D-CNN), true: ORB
    protected Handler autoPilotHandler;
    protected Runnable autoPilotRunnable;

    protected Button btnSwitchPilotMode;
    protected Object orbAutoPilotVM; // Using Object to avoid direct compile-time dependency
    private static final String ORB_VM_CLASS = "dji.sampleV5.aircraft.models.OrbMapMatchingVM";

    protected List<PlanEntry> planList = new ArrayList<>();
    protected int currentPlanIndex = 0;

    protected static class PlanEntry {
        long time;
        double cumDistanceM;
        int stickLeftV;
        int stickLeftH;
        int stickRightV;
        int stickRightH;

        public PlanEntry(long time, double cumDistanceM, int stickLeftV, int stickLeftH, int stickRightV, int stickRightH) {
            this.time = time;
            this.cumDistanceM = cumDistanceM;
            this.stickLeftV = stickLeftV;
            this.stickLeftH = stickLeftH;
            this.stickRightV = stickRightV;
            this.stickRightH = stickRightH;
        }
    }

    private void loadPlanCsv() {
        planList.clear();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("plan.csv")))) {
            reader.readLine(); // Skip Header
            String line;
            while ((line = reader.readLine()) != null) {
                String[] cols = line.split(",");
                if (cols.length >= 14) {
                    long time = Long.parseLong(cols[1].trim());
                    double cumDist = Double.parseDouble(cols[5].trim());
                    int lv = Integer.parseInt(cols[10].trim());
                    int lh = Integer.parseInt(cols[11].trim());
                    int rv = Integer.parseInt(cols[12].trim());
                    int rh = Integer.parseInt(cols[13].trim());
                    planList.add(new PlanEntry(time, cumDist, lv, lh, rv, rh));
                }
            }
            LogUtils.d(TAG, "Plan CSV loaded successfully: " + planList.size() + " entries.");
        } catch (Throwable e) {
            LogUtils.e(TAG, "Failed to load plan.csv: " + e.getMessage());
        }
    }

    private void startAutoPilot() {
        if (!isVirtualStickEnabled) {
            ToastUtils.INSTANCE.showToast("Please enable Virtual Stick first!");
            return;
        }

        if (useOrbMode) {
            startOrbAutoPilot();
            return;
        }

        if (planList.isEmpty()) {
            loadPlanCsv();
        }

        if (planList.isEmpty()) {
            ToastUtils.INSTANCE.showToast("ERROR: plan.csv not loaded!");
            return;
        }

        isAutoPilotRunning = true;
        currentPlanIndex = 0;

        if (btnAutoPilotToggle != null) {
            btnAutoPilotToggle.setText("AutoPilot: ON");
            btnAutoPilotToggle.setBackgroundColor(Color.GREEN);
        }

        autoPilotHandler = new Handler(Looper.getMainLooper());
        autoPilotRunnable = new Runnable() {
            @Override
            public void run() {
                if (!isAutoPilotRunning || !isVirtualStickEnabled) {
                    stopAutoPilot();
                    return;
                }

                if (currentPlanIndex >= planList.size()) {
                    ToastUtils.INSTANCE.showToast("Plan AutoPilot Completed!");
                    stopAutoPilot();
                    return;
                }

                PlanEntry entry = planList.get(currentPlanIndex);
                double progress = ((double) (currentPlanIndex + 1) / planList.size()) * 100.0;

                if (tvVsStatus != null) {
                    tvVsStatus.setText(String.format("Plan: %d/%d (%.1f%%)\nLV:%d LH:%d RV:%d RH:%d",
                            currentPlanIndex + 1, planList.size(), progress,
                            entry.stickLeftV, entry.stickLeftH, entry.stickRightV, entry.stickRightH));
                }

                try {
                    VirtualStickManager.getInstance().getLeftStick().setVerticalPosition(entry.stickLeftV);
                    VirtualStickManager.getInstance().getLeftStick().setHorizontalPosition(entry.stickLeftH);
                    VirtualStickManager.getInstance().getRightStick().setVerticalPosition(entry.stickRightV);
                    VirtualStickManager.getInstance().getRightStick().setHorizontalPosition(entry.stickRightH);
                } catch (Throwable e) {
                    LogUtils.e(TAG, "AutoPilot stick send error: " + e.getMessage());
                }

                currentPlanIndex++;

                long delay = 200;
                if (currentPlanIndex < planList.size()) {
                    long timeDiff = planList.get(currentPlanIndex).time - entry.time;
                    if (timeDiff > 50 && timeDiff < 1000) {
                        delay = timeDiff;
                    }
                }

                autoPilotHandler.postDelayed(this, delay);
            }
        };

        autoPilotHandler.post(autoPilotRunnable);
        ToastUtils.INSTANCE.showToast("Plan AutoPilot Started (" + planList.size() + " frames)");
    }

    private void startOrbAutoPilot() {
        try {
            // Using reflection to call sample module's OrbMapMatchingVM
            ViewModelProvider provider = new ViewModelProvider(this);
            Class<?> vmClass = Class.forName(ORB_VM_CLASS);
            Object vmInstance = provider.get((Class<ViewModel>) vmClass);
            
            // Initialize
            vmClass.getMethod("initAutoPilot", Context.class, String.class)
                .invoke(vmInstance, getApplicationContext(), "images_20260914_130134");

            isAutoPilotRunning = true;
            if (btnAutoPilotToggle != null) {
                btnAutoPilotToggle.setText("AutoPilot: ON");
                btnAutoPilotToggle.setBackgroundColor(Color.BLUE);
            }
            if (tvVsStatus != null) {
                tvVsStatus.setText("VS Status: ORB Active");
            }

            vmClass.getMethod("startAutoPilot").invoke(vmInstance);
            ToastUtils.INSTANCE.showToast("ORB AutoPilot Started");

        } catch (Throwable e) {
            LogUtils.e(TAG, "Failed to start ORB: " + e.getMessage());
            ToastUtils.INSTANCE.showToast("ORB Start Error: " + e.getMessage());
        }
    }

    private void stopAutoPilot() {
        isAutoPilotRunning = false;
        if (autoPilotHandler != null && autoPilotRunnable != null) {
            autoPilotHandler.removeCallbacks(autoPilotRunnable);
        }

        if (useOrbMode) {
            try {
                ViewModelProvider provider = new ViewModelProvider(this);
                Class<?> vmClass = Class.forName(ORB_VM_CLASS);
                Object vmInstance = provider.get((Class<ViewModel>) vmClass);
                vmClass.getMethod("stopAutoPilot").invoke(vmInstance);
            } catch (Throwable e) {
                // ignore
            }
        }

        // スティック中立(0)リセット
        try {
            VirtualStickManager.getInstance().getLeftStick().setVerticalPosition(0);
            VirtualStickManager.getInstance().getLeftStick().setHorizontalPosition(0);
            VirtualStickManager.getInstance().getRightStick().setVerticalPosition(0);
            VirtualStickManager.getInstance().getRightStick().setHorizontalPosition(0);
        } catch (Throwable e) {
            // ignore
        }

        if (btnAutoPilotToggle != null) {
            btnAutoPilotToggle.setText("AutoPilot: OFF");
            btnAutoPilotToggle.setBackgroundColor(Color.DKGRAY);
        }
        if (tvVsStatus != null) {
            tvVsStatus.setText(isVirtualStickEnabled ? "VS Status: Enabled" : "VS Status: Disabled");
        }

        ToastUtils.INSTANCE.showToast("AutoPilot Stopped");
    }

    private void initVirtualStickControls() {
        btnVirtualStickToggle = findViewById(R.id.btn_virtual_stick_toggle);
        btnAutoPilotToggle = findViewById(R.id.btn_autopilot_toggle);
        tvVsStatus = findViewById(R.id.tv_vs_status);

        if (btnVirtualStickToggle == null) return;

        try {
            VirtualStickManager.getInstance().init();
            VirtualStickManager.getInstance().setVirtualStickStateListener(new VirtualStickStateListener() {
                @Override
                public void onVirtualStickStateUpdate(@NonNull VirtualStickState stickState) {
                    isVirtualStickEnabled = stickState.isVirtualStickEnable();
                    runOnUiThread(() -> {
                        if (isVirtualStickEnabled) {
                            btnVirtualStickToggle.setText("Virtual Stick: ON");
                            btnVirtualStickToggle.setBackgroundColor(Color.GREEN);
                            if (tvVsStatus != null && !isAutoPilotRunning) tvVsStatus.setText("VS Status: Enabled");
                        } else {
                            btnVirtualStickToggle.setText("Virtual Stick: OFF");
                            btnVirtualStickToggle.setBackgroundColor(Color.DKGRAY);
                            if (tvVsStatus != null) tvVsStatus.setText("VS Status: Disabled");
                            if (isAutoPilotRunning) stopAutoPilot();
                        }
                    });
                }

                @Override
                public void onChangeReasonUpdate(@NonNull FlightControlAuthorityChangeReason reason) {
                    //
                }
            });
        } catch (Throwable e) {
            LogUtils.e(TAG, "VirtualStickManager init error: " + e.getMessage());
        }

        btnVirtualStickToggle.setOnClickListener(v -> {
            if (!isVirtualStickEnabled) {
                VirtualStickManager.getInstance().enableVirtualStick(new CommonCallbacks.CompletionCallback() {
                    @Override
                    public void onSuccess() {
                        runOnUiThread(() -> ToastUtils.INSTANCE.showToast("Virtual Stick Enabled"));
                    }

                    @Override
                    public void onFailure(@NonNull IDJIError error) {
                        runOnUiThread(() -> ToastUtils.INSTANCE.showToast("Enable Failed: " + error));
                    }
                });
            } else {
                VirtualStickManager.getInstance().disableVirtualStick(new CommonCallbacks.CompletionCallback() {
                    @Override
                    public void onSuccess() {
                        runOnUiThread(() -> {
                            ToastUtils.INSTANCE.showToast("Virtual Stick Disabled");
                            stopAutoPilot();
                        });
                    }

                    @Override
                    public void onFailure(@NonNull IDJIError error) {
                        runOnUiThread(() -> ToastUtils.INSTANCE.showToast("Disable Failed: " + error));
                    }
                });
            }
        });

        if (btnAutoPilotToggle != null) {
            btnAutoPilotToggle.setOnClickListener(v -> {
                if (!isAutoPilotRunning) {
                    startAutoPilot();
                } else {
                    stopAutoPilot();
                }
            });
        }

        btnSwitchPilotMode = findViewById(R.id.btn_switch_pilot_mode);
        if (btnSwitchPilotMode != null) {
            btnSwitchPilotMode.setOnClickListener(v -> {
                useOrbMode = !useOrbMode;
                if (useOrbMode) {
                    btnSwitchPilotMode.setText("Mode: ORB (Geometric)");
                    btnSwitchPilotMode.setBackgroundColor(Color.BLUE);
                } else {
                    btnSwitchPilotMode.setText("Mode: AI (2D-CNN)");
                    btnSwitchPilotMode.setBackgroundColor(Color.rgb(85, 85, 85)); // 555555
                }
                ToastUtils.INSTANCE.showToast("Switched to " + (useOrbMode ? "ORB" : "AI") + " Mode");
            });
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopAutoPilot();
        mapWidget.onDestroy();
        MediaDataCenter.getInstance().getCameraStreamManager().removeAvailableCameraUpdatedListener(availableCameraUpdatedListener);
        DJINetworkManager.getInstance().removeNetworkStatusListener(networkStatusListener);
        VirtualStickManager.getInstance().clearAllVirtualStickStateListener();
        VirtualStickManager.getInstance().destroy();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mapWidget.onResume();
        compositeDisposable = new CompositeDisposable();
        compositeDisposable.add(systemStatusListPanelWidget.closeButtonPressed()
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(pressed -> {
                    if (pressed) {
                        ViewExtensions.hide(systemStatusListPanelWidget);
                    }
                }));
        compositeDisposable.add(simulatorControlWidget.getUIStateUpdates()
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(simulatorControlWidgetState -> {
                    if (simulatorControlWidgetState instanceof SimulatorControlWidget.UIState.VisibilityUpdated) {
                        if (((SimulatorControlWidget.UIState.VisibilityUpdated) simulatorControlWidgetState).isVisible()) {
                            hideOtherPanels(simulatorControlWidget);
                        }
                    }
                }));
        compositeDisposable.add(cameraSourceProcessor.toFlowable()
                .observeOn(SchedulerProvider.io())
                .throttleLast(500, TimeUnit.MILLISECONDS)
                .subscribeOn(SchedulerProvider.io())
                .subscribe(result -> runOnUiThread(() -> onCameraSourceUpdated(result.devicePosition, result.lensType)))
        );
        compositeDisposable.add(ObservableInMemoryKeyedStore.getInstance()
                .addObserver(UXKeys.create(GlobalPreferenceKeys.GIMBAL_ADJUST_CLICKED))
                .observeOn(SchedulerProvider.ui())
                .subscribe(this::isGimableAdjustClicked));
        ViewUtil.setKeepScreen(this, true);
    }

    @Override
    protected void onPause() {
        if (compositeDisposable != null) {
            compositeDisposable.dispose();
            compositeDisposable = null;
        }
        mapWidget.onPause();
        super.onPause();
        ViewUtil.setKeepScreen(this, false);
    }
    //endregion

    private void hideOtherPanels(@Nullable View widget) {
        View[] panels = {
                simulatorControlWidget
        };

        for (View panel : panels) {
            if (widget != panel) {
                panel.setVisibility(View.GONE);
            }
        }
    }

    private void updateFPVWidgetSource(List<ComponentIndexType> availableCameraList) {
        LogUtils.i(TAG, JsonUtil.toJson(availableCameraList));
        if (availableCameraList == null) {
            return;
        }

        ArrayList<ComponentIndexType> cameraList = new ArrayList<>(availableCameraList);

        //没有数据
        if (cameraList.isEmpty()) {
            secondaryFPVWidget.setVisibility(View.GONE);
            return;
        }

        //仅一路数据
        if (cameraList.size() == 1) {
            primaryFpvWidget.updateVideoSource(availableCameraList.get(0));
            secondaryFPVWidget.setVisibility(View.GONE);
            return;
        }

        //大于两路数据
        ComponentIndexType primarySource = getSuitableSource(cameraList, ComponentIndexType.LEFT_OR_MAIN);
        primaryFpvWidget.updateVideoSource(primarySource);
        cameraList.remove(primarySource);

        ComponentIndexType secondarySource = getSuitableSource(cameraList, ComponentIndexType.FPV);
        secondaryFPVWidget.updateVideoSource(secondarySource);

        secondaryFPVWidget.setVisibility(View.VISIBLE);
    }

    private ComponentIndexType getSuitableSource(List<ComponentIndexType> cameraList, ComponentIndexType defaultSource) {
        if (cameraList.contains(ComponentIndexType.LEFT_OR_MAIN)) {
            return ComponentIndexType.LEFT_OR_MAIN;
        } else if (cameraList.contains(ComponentIndexType.RIGHT)) {
            return ComponentIndexType.RIGHT;
        } else if (cameraList.contains(ComponentIndexType.UP)) {
            return ComponentIndexType.UP;
        } else if (cameraList.contains(ComponentIndexType.PORT_1)) {
            return ComponentIndexType.PORT_1;
        } else if (cameraList.contains(ComponentIndexType.PORT_2)) {
            return ComponentIndexType.PORT_2;
        } else if (cameraList.contains(ComponentIndexType.PORT_3)) {
            return ComponentIndexType.PORT_3;
        } else if (cameraList.contains(ComponentIndexType.PORT_4)) {
            return ComponentIndexType.PORT_4;
        } else if (cameraList.contains(ComponentIndexType.VISION_ASSIST)) {
            return ComponentIndexType.VISION_ASSIST;
        }
        return defaultSource;
    }

    private void onCameraSourceUpdated(ComponentIndexType devicePosition, CameraLensType lensType) {
        LogUtils.i(LogPath.SAMPLE, "onCameraSourceUpdated", devicePosition, lensType);
        if (devicePosition == lastDevicePosition && lensType == lastLensType) {
            return;
        }
        lastDevicePosition = devicePosition;
        lastLensType = lensType;
        updateViewVisibility(devicePosition, lensType);
        updateInteractionEnabled();
        //如果无需使能或者显示的，也就没有必要切换了。
        if (fpvInteractionWidget.isInteractionEnabled()) {
            fpvInteractionWidget.updateCameraSource(devicePosition, lensType);
        }
        if (lensControlWidget.getVisibility() == View.VISIBLE) {
            lensControlWidget.updateCameraSource(devicePosition, lensType);
        }
        if (ndviCameraPanel.getVisibility() == View.VISIBLE) {
            ndviCameraPanel.updateCameraSource(devicePosition, lensType);
        }
        if (visualCameraPanel.getVisibility() == View.VISIBLE) {
            visualCameraPanel.updateCameraSource(devicePosition, lensType);
        }
        if (autoExposureLockWidget.getVisibility() == View.VISIBLE) {
            autoExposureLockWidget.updateCameraSource(devicePosition, lensType);
        }
        if (focusModeWidget.getVisibility() == View.VISIBLE) {
            focusModeWidget.updateCameraSource(devicePosition, lensType);
        }
        if (focusExposureSwitchWidget.getVisibility() == View.VISIBLE) {
            focusExposureSwitchWidget.updateCameraSource(devicePosition, lensType);
        }
        if (cameraControlsWidget.getVisibility() == View.VISIBLE) {
            cameraControlsWidget.updateCameraSource(devicePosition, lensType);
        }
        if (focalZoomWidget.getVisibility() == View.VISIBLE) {
            focalZoomWidget.updateCameraSource(devicePosition, lensType);
        }
        if (horizontalSituationIndicatorWidget.getVisibility() == View.VISIBLE) {
            horizontalSituationIndicatorWidget.updateCameraSource(devicePosition, lensType);
        }
    }

    private void updateViewVisibility(ComponentIndexType devicePosition, CameraLensType lensType) {
        //只在fpv下显示
        pfvFlightDisplayWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.VISIBLE : View.INVISIBLE);

        //fpv下不显示
        lensControlWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        ndviCameraPanel.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        visualCameraPanel.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        autoExposureLockWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        focusModeWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        focusExposureSwitchWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        cameraControlsWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        focalZoomWidget.setVisibility(CameraUtil.isFPVTypeView(devicePosition) ? View.INVISIBLE : View.VISIBLE);
        horizontalSituationIndicatorWidget.setSimpleModeEnable(CameraUtil.isFPVTypeView(devicePosition));

        //只在部分len下显示
        ndviCameraPanel.setVisibility(CameraUtil.isSupportForNDVI(lensType) ? View.VISIBLE : View.INVISIBLE);
    }

    /**
     * Swap the video sources of the FPV and secondary FPV widgets.
     */
    private void swapVideoSource() {
        ComponentIndexType primarySource = primaryFpvWidget.getWidgetModel().getCameraIndex();
        ComponentIndexType secondarySource = secondaryFPVWidget.getWidgetModel().getCameraIndex();
        //两个source都存在的情况下才进行切换
        if (primarySource != ComponentIndexType.UNKNOWN && secondarySource != ComponentIndexType.UNKNOWN) {
            primaryFpvWidget.updateVideoSource(secondarySource);
            secondaryFPVWidget.updateVideoSource(primarySource);
        }
    }

    private void updateInteractionEnabled() {
        fpvInteractionWidget.setInteractionEnabled(!CameraUtil.isFPVTypeView(primaryFpvWidget.getWidgetModel().getCameraIndex()));
    }

    private static class CameraSource {
        ComponentIndexType devicePosition;
        CameraLensType lensType;

        public CameraSource(ComponentIndexType devicePosition, CameraLensType lensType) {
            this.devicePosition = devicePosition;
            this.lensType = lensType;
        }
    }

    @Override
    public void onBackPressed() {
        if (mDrawerLayout.isDrawerOpen(GravityCompat.END)) {
            mDrawerLayout.closeDrawers();
        } else {
            super.onBackPressed();
        }
    }
}
