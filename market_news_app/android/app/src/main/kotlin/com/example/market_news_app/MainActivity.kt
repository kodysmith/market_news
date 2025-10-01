package com.example.market_news_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private lateinit var localAIService: LocalAIService
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        // Initialize local AI service for Pixel 10
        localAIService = LocalAIService(this)
        localAIService.setupMethodChannel(flutterEngine)
    }
}
