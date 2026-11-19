package dji.sampleV5.aircraft

import android.app.Application
import dji.sampleV5.aircraft.models.MSDKManagerVM
import dji.sampleV5.aircraft.models.globalViewModels

/**
 * Class Description
 *
 * @author Hoker
 * @date 2022/3/1
 *
 * Copyright (c) 2022, DJI All Rights Reserved.
 */
open class DJIApplication : Application() {

    private val msdkManagerVM: MSDKManagerVM by globalViewModels()

    override fun onCreate() {
        super.onCreate()

        // 1. グローバル・クラッシュハンドラーの設定 (crash_log.txt へ保存)
        initCrashLogger()

        // 2. バックグラウンド Logcat リダイレクト (app_logcat.txt へ連続記録)
        initLogcatCapture()

        // Ensure initialization is called first
        msdkManagerVM.initMobileSDK(this)
    }

    private fun initCrashLogger() {
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val logDir = getExternalFilesDir(null) ?: filesDir
                val crashFile = java.io.File(logDir, "crash_log.txt")
                val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())
                
                val sw = java.io.StringWriter()
                val pw = java.io.PrintWriter(sw)
                throwable.printStackTrace(pw)
                val stackTraceStr = sw.toString()

                val logContent = "========================================\n" +
                        "CRASH TIME: $timestamp\n" +
                        "THREAD: ${thread.name}\n" +
                        "EXCEPTION: ${throwable.javaClass.name}\n" +
                        "MESSAGE: ${throwable.message}\n" +
                        "STACK TRACE:\n$stackTraceStr\n" +
                        "========================================\n\n"

                // ファイルに追記
                java.io.FileWriter(crashFile, true).use { writer ->
                    writer.write(logContent)
                    writer.flush()
                }
                android.util.Log.e("DJIApplication", "Crash log written to ${crashFile.absolutePath}")
            } catch (e: Throwable) {
                android.util.Log.e("DJIApplication", "Failed to write crash log", e)
            } finally {
                defaultHandler?.uncaughtException(thread, throwable)
            }
        }
    }

    private fun initLogcatCapture() {
        try {
            val logDir = getExternalFilesDir(null) ?: filesDir
            val logcatFile = java.io.File(logDir, "app_logcat.txt")
            
            // logcatプロセスをバックグラウンド起動してファイルへ連続出力
            Runtime.getRuntime().exec("logcat -f ${logcatFile.absolutePath} -v time *:V")
            android.util.Log.d("DJIApplication", "Logcat capture started: ${logcatFile.absolutePath}")
        } catch (e: Throwable) {
            android.util.Log.e("DJIApplication", "Failed to start logcat capture", e)
        }
    }

}
