#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Arduino_JSON.h>

#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>


const char* ssid = "";
const char* password = "";

const char*  serverURL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run/sensor-data";  

unsigned long lastPostTime = 0;
const unsigned long postInterval = 5 * 60 * 1000;  // 5 minutes in ms

//Soil Moisture sensor setup
#define MOISTMETER 12
int SoilMoistureValue;
int SoilMoisturePercent;
//max dry value
const int DryValue = 4095;
//max wet value 
const int WetValue = 0;

//DHT septup 
#define DHTPIN 13
#define DHTTYPE DHT11
DHT_Unified dht(DHTPIN, DHTTYPE);
float temperature;
float humidity;
uint32_t delayMS;

//https certificate 
const char* test_root_ca = \
"-----BEGIN CERTIFICATE----- \n"\
"MIIEVzCCAj+gAwIBAgIRAIOPbGPOsTmMYgZigxXJ/d4wDQYJKoZIhvcNAQELBQAw\n"\
"TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh\n"\
"cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMjQwMzEzMDAwMDAw\n"\
"WhcNMjcwMzEyMjM1OTU5WjAyMQswCQYDVQQGEwJVUzEWMBQGA1UEChMNTGV0J3Mg\n"\
"RW5jcnlwdDELMAkGA1UEAxMCRTUwdjAQBgcqhkjOPQIBBgUrgQQAIgNiAAQNCzqK\n"\
"a2GOtu/cX1jnxkJFVKtj9mZhSAouWXW0gQI3ULc/FnncmOyhKJdyIBwsz9V8UiBO\n"\
"VHhbhBRrwJCuhezAUUE8Wod/Bk3U/mDR+mwt4X2VEIiiCFQPmRpM5uoKrNijgfgw\n"\
"gfUwDgYDVR0PAQH/BAQDAgGGMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcD\n"\
"ATASBgNVHRMBAf8ECDAGAQH/AgEAMB0GA1UdDgQWBBSfK1/PPCFPnQS37SssxMZw\n"\
"i9LXDTAfBgNVHSMEGDAWgBR5tFnme7bl5AFzgAiIyBpY9umbbjAyBggrBgEFBQcB\n"\
"AQQmMCQwIgYIKwYBBQUHMAKGFmh0dHA6Ly94MS5pLmxlbmNyLm9yZy8wEwYDVR0g\n"\
"BAwwCjAIBgZngQwBAgEwJwYDVR0fBCAwHjAcoBqgGIYWaHR0cDovL3gxLmMubGVu\n"\
"Y3Iub3JnLzANBgkqhkiG9w0BAQsFAAOCAgEAH3KdNEVCQdqk0LKyuNImTKdRJY1C\n"\
"2uw2SJajuhqkyGPY8C+zzsufZ+mgnhnq1A2KVQOSykOEnUbx1cy637rBAihx97r+\n"\
"bcwbZM6sTDIaEriR/PLk6LKs9Be0uoVxgOKDcpG9svD33J+G9Lcfv1K9luDmSTgG\n"\
"6XNFIN5vfI5gs/lMPyojEMdIzK9blcl2/1vKxO8WGCcjvsQ1nJ/Pwt8LQZBfOFyV\n"\
"XP8ubAp/au3dc4EKWG9MO5zcx1qT9+NXRGdVWxGvmBFRAajciMfXME1ZuGmk3/GO\n"\
"koAM7ZkjZmleyokP1LGzmfJcUd9s7eeu1/9/eg5XlXd/55GtYjAM+C4DG5i7eaNq\n"\
"cm2F+yxYIPt6cbbtYVNJCGfHWqHEQ4FYStUyFnv8sjyqU8ypgZaNJ9aVcWSICLOI\n"\
"E1/Qv/7oKsnZCWJ926wU6RqG1OYPGOi1zuABhLw61cuPVDT28nQS/e6z95cJXq0e\n"\
"K1BcaJ6fJZsmbjRgD5p3mvEf5vdQM7MCEvU0tHbsx2I5mHHJoABHb8KVBgWp/lcX\n"\
"GWiWaeOyB7RP+OfDtvi2OsapxXiV7vNVs7fMlrRjY1joKaqmmycnBvAq14AEbtyL\n"\
"sVfOS66B8apkeFX2NY4XPEYV4ZSCe8VHPrdrERk2wILG3T/EGmSIkCYVUMSnjmJd\n"\
"VQD9F6Na/+zmXCc=\n"\
"-----END CERTIFICATE-----\n";

WiFiClientSecure client;
HTTPClient https;

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(115200);
  delay(100);

  Serial.print("Attempting to connect to SSID: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  // attempt to connect to Wifi network:
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    // wait 1 second for re-trying
    delay(1000);
  }

  Serial.print("Connected to ");
  Serial.println(ssid);
  
  Serial.println("\nStarting connection to server...");

  client.setCACert(test_root_ca);

    while (client.connected()) {
      String line = client.readStringUntil('\n');
      if (line == "\r") {
        Serial.println("headers received");
        break;
      }
    }
    // if there are incoming bytes available
    // from the server, read them and print them:
    while (client.available()) {
      char c = client.read();
      Serial.write(c);
    }

    client.stop();
    //reading for soil moisture sensor
    analogReadResolution(12);


    dht.begin();
    sensor_t sensor;
    dht.temperature().getSensor(&sensor);
    dht.humidity().getSensor(&sensor);
    delayMS = sensor.min_delay / 1000;
  }


void loop() {

  if (WiFi.status() == WL_CONNECTED) {

    delay(delayMS);
    sensors_event_t event;

    SoilMoistureValue = analogRead(12);

    SoilMoisturePercent = map(SoilMoistureValue, WetValue, DryValue, 0, 100);
    SoilMoisturePercent = constrain(SoilMoistureValue, 0, 100);
    Serial.println(SoilMoisturePercent);

    dht.temperature().getEvent(&event);
    if (!isnan(event.temperature)) {
      Serial.print("Temperature: ");
      Serial.println(event.temperature);
      temperature = event.temperature;
    }

    dht.humidity().getEvent(&event);
    if (!isnan(event.relative_humidity)) {
      Serial.print("Humidity: ");
      Serial.println(event.relative_humidity);
      humidity = event.relative_humidity;
    }

    
    // Try with much longer timeout for slow Modal API
    HTTPClient http;
    http.begin(client, serverURL);
    http.setTimeout(180000);  // 3 minute timeout for cold starts and processing
    http.addHeader("Content-Type", "application/json");

 String json = "{\n"
        "\t\"plant_id\": \"my_plant_001\",\n"
        "\t\"soil_moisture\": " + String(SoilMoisturePercent) + ",\n"
        "\t\"temperature\": " + String(temperature) + ",\n"
        "\t\"humidity\": " + String(humidity) + "\n"   // removed comma here
        "}";

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