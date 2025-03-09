import pywhatkit
from datetime import datetime
import pandas as pd
import sqlite3

# Get current time
def get_person_info(person_id):
    # Load CSV file
    df = pd.read_csv("citizens.csv")

    # Search for the person with matching ID
    person = df[df["ID"] == person_id]

    if person.empty:
        return "No record found for this ID."

    # Extract values
    name = person.iloc[0]["Legal Name"]
    occupation = person.iloc[0]["Occupation"]
    phone = person.iloc[0]["Phone Number"]
    emergency = person.iloc[0]["Emergency Contact"]
    sin = person.iloc[0]["Social Insurance Number"]
    address = person.iloc[0]["Address"]

    # Format message
    message = f"""
🚨 Possible Suicide Attempt Detected 🚨


👤 Name: {name}

💼 Occupation: {occupation}

📞 Phone Number: {phone}

📟 Emergency Contact: {emergency}

🆔 Social Insurance Number: {sin}

🏠 Home Address: {address}


⚠️ Please take immediate action
"""

    with open("./alert.txt", "w", encoding="utf-8") as file:
        file.write(message)
