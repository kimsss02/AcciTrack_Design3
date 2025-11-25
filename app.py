import joblib
from flask import Flask, flash, render_template, request, redirect, jsonify, session, url_for
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash
import mysql.connector
from flask_cors import CORS
from routes import register_routes
import numpy as np
from inference_worker import start_pipeline 
from threading import Thread
from datetime import datetime
import requests


app = Flask(__name__)

register_routes(app)


CORS(app, supports_credentials=True)
app.secret_key = "accitrack_2025_secret_key_for_sessions"

def run_detection_pipeline():
    try:
        print("🚦 Starting detection pipeline...")
        start_pipeline()
    except Exception as e:
        print(f"Detection pipeline error: {e}")

Thread(target=run_detection_pipeline, daemon=True).start()

#---------------- DBASE CONNECTION -------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="08f_lala",
    database="acci_track",
    ssl_disabled=True,  
    autocommit=True
)
cursor = db.cursor(dictionary=True)

# ---------------- AUTH ----------------
@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['role'] = user['role']
        return jsonify({'message': 'Login successful', 'role': user['role']})
    return jsonify({'error': 'Invalid email or password'}), 401

# ---------------- DASHBOARDS ----------------
@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect('/')

    user_id = session.get('user_id')
    cursor.execute("SELECT email FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    username = user['email'] if user else 'User'

    # Embed your Power BI report for user view
    powerbi_link = "https://app.powerbi.com/reportEmbed?reportId=b4a33197-c267-4aa0-8736-76821ada7b50&autoAuth=true&ctid=a515e7c4-8cf7-4f97-b23b-96491645f7f0"

    return render_template('user_dashboard.html', username=username, powerbi_link=powerbi_link)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/')
    powerbi_link = "https://app.powerbi.com/reportEmbed?reportId=b4a33197-c267-4aa0-8736-76821ada7b50&autoAuth=true&ctid=a515e7c4-8cf7-4f97-b23b-96491645f7f0"
    return render_template('admin_dashboard.html', powerbi_link=powerbi_link)

# admin_history 
@app.route('/history')
def history():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')

    cursor.execute("""
        SELECT id, accident_type, temperature, humidity, pressure, gas, light, time_of_day, weather, recorded_at
        FROM detection
        ORDER BY recorded_at DESC
    """)
    history = cursor.fetchall()
    return render_template('history.html', history=history)


# admin_profile
@app.route('/admin_profile', methods=['GET', 'POST'])
def admin_profile():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')

    admin_id = session['user_id']

    if request.method == 'POST':
        new_email = request.form['email']
        new_password = request.form['password']

        if new_password.strip():
            hashed_pw = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET email = %s, password = %s WHERE id = %s", (new_email, hashed_pw, admin_id))
        else:
            cursor.execute("UPDATE users SET email = %s WHERE id = %s", (new_email, admin_id))
        db.commit()
        return redirect('/admin_profile')

    cursor.execute("SELECT email FROM users WHERE id = %s", (admin_id,))
    admin = cursor.fetchone()
    return render_template('admin_profile.html', email=admin['email'])


@app.route('/monitor')
def monitor_page():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('monitor.html')

@app.route('/verify')
def verify_page():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('verify.html')

@app.route('/verify/<action>/<filename>', methods=['POST'])
def verify_incident(action, filename):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 401

    if action not in ["approve", "reject"]:
        return jsonify({"status": "error", "error": "Invalid action"}), 400

    if action == "approve":
        # Try fetching sensor data from ESP32
        try:
            resp = requests.get("http://192.168.100.207/sensor-data", timeout=3)
            resp.raise_for_status()
            sensor_data = resp.json()
        except Exception as e:
            print("❌ Failed to fetch from ESP32:", e)
            return jsonify({"status": "error", "error": "Failed to read sensor data"}), 500

        # Insert into MySQL
        approved_at = datetime.now()
        try:
            cursor.execute("""
                INSERT INTO detection
                (accident_type, temperature, humidity, pressure, gas, light, time_of_day, weather, recorded_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                sensor_data["accident_type"],
                sensor_data["temperature"],
                sensor_data["humidity"],
                sensor_data["pressure"],
                sensor_data["gas"],
                sensor_data["lux"],
                sensor_data["time_of_day"],
                sensor_data["weather"],
                approved_at
            ))
            db.commit()
        except Exception as e:
            print("❌ Database insert failed:", e)
            return jsonify({"status": "error", "error": "Database error"}), 500

        return jsonify({"status": "approved", "data": {**sensor_data, "approved_at": approved_at.strftime("%Y-%m-%d %H:%M:%S")}})

    elif action == "reject":
        # Remove file logic (optional)
        try:
            # Example: os.remove(f"static/pending_incidents/{filename}")
            pass
        except Exception as e:
            print("❌ Failed to delete file:", e)
        return jsonify({"status": "rejected"})

# user_history
@app.route('/user_history')
def user_history():
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect('/login')
    
    cursor.execute("""
        SELECT id, accident_type, temperature, humidity, pressure, gas, light, time_of_day, weather, recorded_at
        FROM detection
        ORDER BY recorded_at DESC
    """)
    history = cursor.fetchall()
    return render_template('user_history.html', history=history)


#user_profile
@app.route('/profile', methods=['GET', 'POST'])
def user_profile():
    if 'user_id' not in session or session.get('role') != 'user':
        return redirect('/login')

    user_id = session['user_id']

    if request.method == 'POST':
        new_email = request.form['email']
        new_password = request.form['password']

        if new_password.strip() != "":
            hashed_pw = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET email = %s, password = %s WHERE id = %s", (new_email, hashed_pw, user_id))
        else:
            cursor.execute("UPDATE users SET email = %s WHERE id = %s", (new_email, user_id))
        db.commit()
        return redirect('/profile')

    cursor.execute("SELECT email FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    return render_template('profile.html', email=user['email'])

# -------------------------------------------------------------------------------
@app.route('/users')
def manage_users():
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, email, role, status FROM users")
    users = cursor.fetchall()
    success = request.args.get('success')  
    return render_template('manage_users.html', users=users, success=success)

@app.route('/users/update/<int:user_id>', methods=['POST'])
def update_user(user_id):
    email = request.form['email']
    role = request.form['role']
    status = request.form['status']
    cursor = db.cursor()
    cursor.execute("UPDATE users SET email=%s, role=%s, status=%s WHERE id=%s",
                   (email, role, status, user_id))
    db.commit()
    return redirect(url_for('manage_users'))

@app.route('/users/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    cursor = db.cursor()
    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    db.commit()
    return redirect(url_for('manage_users'))

from werkzeug.security import generate_password_hash
@app.route('/users/add', methods=['POST'])
def add_user():
    email = request.form['email']
    password = request.form['password']
    role = request.form['role']
    status = request.form.get('status', 'active')

    hashed_pw = generate_password_hash(password)

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO users (email, password, role, status)
        VALUES (%s, %s, %s, %s)
    """, (email, hashed_pw, role, status))
    db.commit()
    return redirect(url_for('manage_users', success='User added successfully'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    app.run(debug=True)
