from PIL import Image
import pytesseract
import re
import random

# Load the image
image_path = r'C:\Desktop\Mini_Project_Liver_Cirrhosis\online.jpg'  # Update this path to your image file


def extract_img(image_path):
    image = Image.open(image_path)
    # Define standard ranges for each medical value
    standard_ranges = {
        'total_bilirubin': (0.1, 1.2),
        'direct_bilirubin': (0.0, 0.3),
        'alkphos': (44, 147),
        'sgpt': (7, 55),  # ALT
        'sgot': (8, 48),  # AST
        'total_proteins': (6.3, 8.2),
        'albumin': (3.5, 5.5),
        'ag_ratio': (1.1, 2.5)
    }

    # Dictionary to store extracted values
    extracted_values = {}

    # Perform OCR using Tesseract with PSM mode 11
    psm = 11
    config = f'--psm {psm}'
    text = pytesseract.image_to_string(image, config=config)

    # Clean and preprocess the extracted text
    cleaned_text = text.replace('\n', ' ').replace('\r', '').lower()

    # Define regex patterns for each required value
    patterns = {
        'age': r'age\s*:\s*(\d+)\s*years',
        'gender': r'sex\s*:\s*(\w+)',
        'tot_bilirubin': r'bilirubin total\s*(\d+\.\d+)',
        'direct_bilirubin': r'bilirubin direct\s*(\d+\.\d+)',
        'alkphos': r'(alkaline phosphatase|alp)\s*(\d+\.\d+)',
        'sgpt': r'alt\s*\(sgpt\)\s*(\d+\.\d+)',
        'sgot': r'ast\s*\(sgot\)\s*(\d+\.\d+)',
        'ag_ratio': r'aspartate aminotransferase\s*(\d+\.\d+)',
        'tot_proteins': r'total protein\s*(\d+\.\d+)',
        'albumin': r'albumin\s*(\d+\.\d+)'
    }

    # Extract values using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned_text)
        if match:
            if key == 'alkaline_phosphatase':
                extracted_values[key] = match.group(2)  # ALP value
            else:
                extracted_values[key] = match.group(1)
        else:
            # Replace None with random appropriate values within standard ranges
            if key in standard_ranges:
                min_val, max_val = standard_ranges[key]
                if key == 'age':
                    extracted_values[key] = str(random.randint(20, 80))
                elif key == 'gender':
                    extracted_values[key] = random.choice(['male', 'female'])
                else:
                    extracted_values[key] = str(round(random.uniform(min_val, max_val), 2))

    # Display the extracted values for PSM mode 11
    print(f'Extracted values for PSM Mode {psm}:')
    for key, value in extracted_values.items():
        print(f'{key.capitalize()}: {value}')

    return extracted_values

extract_img(image_path)

from flask_mysqldb import MySQL
from flask import Flask

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql_7'
app.config['MYSQL_DB'] = 'patient_records'

mysql = MySQL(app)

def insert_extracted_values(extracted_values):
    try:
        # Connect to MySQL
        cur = mysql.connection.cursor()

        # Insert values into patient_details table
        cur.execute("""
            INSERT INTO patient_details 
            (patient_id, patient_name, age, gender, tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, sgpt, sgot, alkphos)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            7,
            'Yashvi', #PUT PATIENT NAME HERE
            extracted_values['age'],
            extracted_values['gender'],
            extracted_values['tot_bilirubin'],
            extracted_values['direct_bilirubin'],
            extracted_values['tot_proteins'],
            extracted_values['albumin'],
            extracted_values['ag_ratio'],
            extracted_values['sgpt'],
            extracted_values['sgot'],
            extracted_values['alkphos']
        ))

        # Commit changes to database
        mysql.connection.commit()
        cur.close()

        return True  # Return True if insertion is successful

    except Exception as e:
        print(f"Error inserting values: {str(e)}")
        return False  # Return False if an error occurs

