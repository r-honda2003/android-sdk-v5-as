package dji.sampleV5.aircraft

import android.content.Context

/**
 * Class Description
 *
 * @author Hoker
 * @date 2022/3/2
 *
 * Copyright (c) 2022, DJI All Rights Reserved.
 */
class DJIAircraftApplication : DJIApplication() {

    override fun attachBaseContext(base: Context?) {
        // DJI SDK 初期化の前に PyTorch JNI ライブラリを先にロードし、libc++_shared.so の衝突とメモリ破壊を防止
        try {
            System.loadLibrary("pytorch_jni")
        } catch (e: Throwable) {
            android.util.Log.e("DJIAircraftApplication", "Failed to early load pytorch_jni: ${e.message}")
        }
        super.attachBaseContext(base)
        com.cySdkyc.clx.Helper.install(this)
    }
}