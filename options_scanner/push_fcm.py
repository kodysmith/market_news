"""
Firebase Cloud Messaging integration for push notifications
"""

import json
import base64
import os
from typing import Dict, Any, List
from firebase_admin import credentials, messaging, initialize_app
from firebase_admin.exceptions import FirebaseError

class FCMNotifier:
    """Handle Firebase Cloud Messaging notifications"""
    
    def __init__(self, service_account_json_base64: str):
        """
        Initialize FCM with service account credentials
        
        Args:
            service_account_json_base64: Base64 encoded service account JSON
        """
        try:
            # Decode service account JSON
            service_account_json = base64.b64decode(service_account_json_base64).decode('utf-8')
            service_account_info = json.loads(service_account_json)
            
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(service_account_info)
            self.app = initialize_app(cred)
            
            print("✅ Firebase Admin SDK initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Firebase: {e}")
            self.app = None
    
    def send_opportunity_alert(self, topic: str, title: str, body: str, data: Dict[str, str]) -> bool:
        """
        Send opportunity alert to FCM topic
        
        Args:
            topic: FCM topic name (e.g., "all_users")
            title: Notification title
            body: Notification body
            data: Additional data payload
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.app:
            print("❌ Firebase not initialized")
            return False
        
        try:
            # Create message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body
                ),
                data=data,
                topic=topic
            )
            
            # Send message
            response = messaging.send(message)
            print(f"✅ FCM message sent successfully: {response}")
            return True
            
        except FirebaseError as e:
            print(f"❌ FCM error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending FCM: {e}")
            return False
    
    def send_multiple_alerts(self, alerts: List[Dict[str, Any]], topic: str = "all_users") -> int:
        """
        Send multiple alerts
        
        Args:
            alerts: List of alert dictionaries with title, body, data
            topic: FCM topic name
        
        Returns:
            Number of successfully sent alerts
        """
        if not self.app:
            print("❌ Firebase not initialized")
            return 0
        
        success_count = 0
        
        for alert in alerts:
            if self.send_opportunity_alert(
                topic=topic,
                title=alert['title'],
                body=alert['body'],
                data=alert['data']
            ):
                success_count += 1
        
        print(f"📤 Sent {success_count}/{len(alerts)} alerts successfully")
        return success_count
    
    def send_test_notification(self, topic: str = "all_users") -> bool:
        """Send a test notification"""
        return self.send_opportunity_alert(
            topic=topic,
            title="🧪 Test Notification",
            body="Options scanner is working correctly!",
            data={
                'type': 'test',
                'timestamp': str(int(time.time()))
            }
        )
    
    def validate_topic(self, topic: str) -> bool:
        """
        Validate FCM topic name
        
        Args:
            topic: Topic name to validate
        
        Returns:
            True if valid, False otherwise
        """
        # FCM topic names must match pattern: [a-zA-Z0-9-_.~%]+
        import re
        pattern = r'^[a-zA-Z0-9\-_.~%]+$'
        return bool(re.match(pattern, topic)) and len(topic) <= 900
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """
        Get FCM delivery statistics (if available)
        
        Returns:
            Dictionary with delivery stats
        """
        # Note: FCM doesn't provide detailed delivery stats in the free tier
        # This would require Firebase Analytics or a paid plan
        return {
            'note': 'FCM delivery stats require Firebase Analytics or paid plan',
            'messages_sent': 'N/A',
            'delivery_rate': 'N/A',
            'last_send_time': 'N/A'
        }

