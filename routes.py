from flask import Response, render_template, request, jsonify
from stream import generate_stream
from esp32_utils import fetch_sensor_data
import os
import json
from datetime import datetime
from inference_worker import get_incident_logs
import mysql.connector

# Path where detected images are stored
PENDING_DIR = "static/pending_incidents"
APPROVED_DIR = "static/approved_incidents"

# directories exist
os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(APPROVED_DIR, exist_ok=True)

def register_routes(app):
    @app.route('/monitor')
    def index():
        return render_template('monitor.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/sensor-readings')
    def sensor_readings():
        data = fetch_sensor_data()
        if data:
            return jsonify(data)
        else:
            return jsonify({"error": "Could not retrieve sensor data"}), 500
        
    @app.route('/incident_logs')
    def incident_logs_route():
        return jsonify(get_incident_logs())

    @app.route('/verify')
    def verify_tab():
        images = os.listdir(PENDING_DIR)
        images = sorted(images, reverse=True)
        return render_template("verify.html", images=images)

    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "08f_lala",
        "database": "acci_track"
    }

    @app.route('/verify/approve/<filename>', methods=['POST'])
    def approve_incident(filename):
        src_path = os.path.join(PENDING_DIR, filename)
        dest_path = os.path.join(APPROVED_DIR, filename)
        if os.path.exists(src_path):
            os.rename(src_path, dest_path)
            
            # Get sensor data
            sensor_data = fetch_sensor_data()
            
            log_entry = {
                "filename": filename,
                "approved_at": datetime.now().isoformat(),
                "sensor_data": sensor_data,
                "location": "Camera Location 1"
            }

            # Save log file (still optional)
            with open("incident_log.json", "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            
            # Insert into database
            try:
                conn = mysql.connector.connect(**db_config)
                cursor = conn.cursor()

                # Prepare values
                accident_type = "Vehicle Collision"  # or derive from filename/model
                temperature = float(sensor_data.get("temperature", 0))
                humidity = float(sensor_data.get("humidity", 0))
                time_of_day = datetime.now().time()
                weather = "Rainy" if sensor_data.get("rain_detected") else "Not Rainy"
                recorded_at = datetime.now()

                sql = """
                    INSERT INTO detection (accident_type, temperature, humidity, time_of_day, weather, recorded_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = (accident_type, temperature, humidity, time_of_day, weather, recorded_at)

                cursor.execute(sql, values)
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Database error: {err}")
            finally:
                cursor.close()
                conn.close()

            return jsonify({"status": "approved", "data": log_entry})

        return jsonify({"error": "File not found"}), 404


    @app.route('/verify/reject/<filename>', methods=['POST'])
    def reject_incident(filename):
        file_path = os.path.join(PENDING_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"status": "rejected", "filename": filename})
        return jsonify({"error": "File not found"}), 404
    
