import pandas as pd
import mysql.connector

# Connect to DB
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="08f_lala",
    database="acci_track"   # adjust db name
)

# Query the table
df = pd.read_sql("SELECT * FROM combined1", conn)

# Save to CSV
df.to_csv("combined.csv", index=False)

conn.close()
print("✅ Exported combined table to combined.csv")
