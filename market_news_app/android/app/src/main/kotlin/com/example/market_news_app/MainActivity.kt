package com.example.market_news_app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private lateinit var localAIService: LocalAIService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createMarketAlertsNotificationChannel()
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Initialize local AI service for Pixel 10
        localAIService = LocalAIService(this)
        localAIService.setupMethodChannel(flutterEngine)
    }

    private fun createMarketAlertsNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            "market_alerts",
            "Market Alerts",
            NotificationManager.IMPORTANCE_MAX
        ).apply {
            description = "GEX regime flips and other time-sensitive market alerts"
            setShowBadge(true)
            enableVibration(true)
            enableLights(true)
        }
        channel.lockscreenVisibility = android.app.Notification.VISIBILITY_PUBLIC
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            channel.setBypassDnd(true)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
}
