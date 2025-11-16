#include <WiFi.h>
#include <Wire.h>
#include <WebServer.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_TSL2591.h>

// WiFi
//const char* ssid = "PLDTHOMEFIBRf4468";
//const char* password = "PLDTWIFI2x2j5";

//const char* ssid = "S213 WIFI";
//const char* password = "VD4I3K8V";

const char* ssid = "HUAWEI-2.4G-2Ywa";
const char* password = "#ElysPlace@3F_2025!";

// I2C
#define SDA_PIN 21
#define SCL_PIN 22

// Sensors
Adafruit_BME680 bme;
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// Web server on port 80
WebServer server(80);

void setup() {
  Serial.begin(115200);

  // Connect WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // Init BME680
  if (!bme.begin()) {
    Serial.println("❌ BME680 not found");
    while (1);
  }
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150);

  // Init TSL2591
  if (!tsl.begin()) {
    Serial.println("❌ TSL2591 not found");
    while (1);
  }
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);

  // Define endpoint
  server.on("/sensor-data", HTTP_GET, []() {
    if (!bme.performReading()) {
      server.send(500, "text/plain", "Sensor error");
      return;
    }

    float temp = bme.temperature;
    float hum = bme.humidity;
    float pres = bme.pressure / 100.0;
    float gas = bme.gas_resistance / 1000.0;

    uint16_t visible = tsl.getLuminosity(TSL2591_VISIBLE);
    uint16_t ir = tsl.getLuminosity(TSL2591_INFRARED);
    uint16_t full = tsl.getLuminosity(TSL2591_FULLSPECTRUM);
    uint32_t lux = tsl.calculateLux(full, ir);

    String json = "{";
    json += "\"temperature\":" + String(temp) + ",";
    json += "\"humidity\":" + String(hum) + ",";
    json += "\"pressure\":" + String(pres) + ",";
    json += "\"gas\":" + String(gas) + ",";
    json += "\"lux\":" + String(lux) + ",";
    json += "\"visible\":" + String(visible);
    json += "}";

    server.send(200, "application/json", json);
  });

  server.begin();
  Serial.println("✅ Serial started");
  Serial.println("✅ Web server started");
}

void loop() {
  server.handleClient();
}