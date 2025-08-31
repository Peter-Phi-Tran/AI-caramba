#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid       = "Wokwi-GUEST";
const char* password   = "";
const char* serverURL  = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run/sensor-data";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    
    // Try with much longer timeout for slow Modal API
    http.begin(serverURL);
    http.setTimeout(180000);  // 3 minute timeout for cold starts and processing
    http.addHeader("Content-Type", "application/json");

    String json = "{\"plant_id\":\"my_plant_001\",\"soil_moisture\":25.0,\"light_level\":65.0,\"temperature\":22.0,\"humidity\":50.0}";

    Serial.println("Starting POST request...");
    Serial.println("Payload: " + json);
    unsigned long start = millis();
    
    int httpCode = http.POST(json);
    
    unsigned long elapsed = millis() - start;
    Serial.printf("Request completed in %lu ms\n", elapsed);
    Serial.printf("HTTP Code: %d\n", httpCode);

    if (httpCode > 0) {
      if (httpCode == 200) {
        String response = http.getString();
        Serial.println("SUCCESS! Response: " + response);
      } else {
        Serial.printf("HTTP Error Code: %d\n", httpCode);
        String errorResponse = http.getString();
        if (errorResponse.length() > 0) {
          Serial.println("Error Response: " + errorResponse);
        }
      }
    } else {
      Serial.printf("Connection Error: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
  } else {
    Serial.println("WiFi disconnected!");
  }

  Serial.println("Waiting 30 seconds...\n");
  delay(30000);
}
