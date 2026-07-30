import sqlite3

# 1. Connect to the database
conn = sqlite3.connect("database/urbansight_logs.db")
cursor = conn.cursor()

# 2. Grab everything from the table
cursor.execute("SELECT * FROM geofence_alerts")
rows = cursor.fetchall()

# 3. Print it out beautifully
print("--- 🚦 UrbanSight Geofence Logs 🚦 ---")
if not rows:
    print("The database is currently empty.")
else:
    for row in rows:
        log_id = row[0]
        timestamp = row[1]
        vehicle_id = row[2]
        print(f"Log ID: {log_id} | Time: {timestamp} | Vehicle #{vehicle_id}")
print("--------------------------------------")

# 4. Close the connection
conn.close()
