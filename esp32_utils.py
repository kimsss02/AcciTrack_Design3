import requests

ESP32_IP = "http://192.168.100.207"

def fetch_sensor_data():
    try:
        response = requests.get(f"{ESP32_IP}/sensor-data", timeout=3)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"ESP32 returned error: {response.status_code}")
    except Exception as e:
        print("❌ Failed to fetch from ESP32:", e)
    return None
