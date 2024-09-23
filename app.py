import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
from extract import extract_img, insert_extracted_values
from sklearn.preprocessing import StandardScaler
import pandas as pd
from liver_disease_prediction import Model

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql_7'
app.config['MYSQL_DB'] = 'patient_records'

mysql = MySQL(app)

# Load the model
model = joblib.load('best_rf_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        patient_name = request.form['patient_name']
        age = request.form['age']
        gender = request.form['gender']
        tot_bilirubin = request.form['tot_bilirubin']
        direct_bilirubin = request.form['direct_bilirubin']
        tot_proteins = request.form['tot_proteins']
        albumin = request.form['albumin']
        ag_ratio = request.form['ag_ratio']
        sgpt = request.form['sgpt']
        sgot = request.form['sgot']
        alkphos = request.form['alkphos']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO patient_details (patient_id, patient_name, age, gender, tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, sgpt, sgot, alkphos) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (patient_id, patient_name, age, gender, tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, sgpt, sgot, alkphos))
        mysql.connection.commit()
        cur.close()

        # Prepare data for prediction
        patient_data = np.array([[float(age), float(tot_bilirubin), float(direct_bilirubin), float(tot_proteins), float(albumin), float(ag_ratio), float(sgpt), float(sgot), float(alkphos)]])
        scaler = StandardScaler()
        patient_data_scaled = scaler.fit_transform(patient_data)
        #prediction = model.predict(patient_data_scaled)
        prediction = Model(patient_data)
        if prediction == 1:
            print("The patient has liver disease")
        else:
            print("The patient does not have liver disease")
        print(prediction[0])

        return redirect(url_for('predict', patient_id = patient_id, patient_name = patient_name,prediction=prediction[0]))

    return render_template('form.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'imageUpload' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['imageUpload']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as e:
                flash('File upload failed: ' + str(e))
                return redirect(request.url)

            try:
                extracted_values = extract_img(filepath)
                # Prepare data for prediction
                patient_data = np.array([extracted_values])
                scaler = StandardScaler()
                patient_data_scaled = scaler.fit_transform(patient_data)
                #prediction = model.predict(patient_data_scaled)
                prediction = Model(patient_data)
                if prediction == 1:
                    print("The patient has liver disease")
                else:
                    print("The patient does not have liver disease")
                print(prediction[0])
                patient_id = extracted_values.get('patient_id')
                patient_name = extracted_values.get('patient_name')
                return redirect(url_for('predict', patient_id = '7', patient_name = 'Yashvi', prediction=prediction[0]))
            except Exception as e:
                flash('Extraction failed: ' + str(e))
                return redirect(request.url)

    return render_template('upload.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    # Assuming you have the logic to get prediction here
    patient_id = request.args.get('patient_id')
    patient_name = request.args.get('patient_name')
    prediction = request.args.get('prediction', None)
    plot_url = visualize(patient_id, patient_name)
    return render_template('predict.html', prediction=prediction, plot_url=plot_url)

@app.route('/visualize/<patient_id>/<patient_name>')
def visualize(patient_id, patient_name):
    # Fetch data from the database
    cur = mysql.connection.cursor()
    query = """
        SELECT 
    pd.date, 
    pd.tot_bilirubin, 
    pd.direct_bilirubin, 
    pd.tot_proteins, 
    pd.albumin, 
    pd.ag_ratio, 
    pd.sgpt, 
    pd.sgot, 
    pd.alkphos
FROM 
    patient_details pd
WHERE 
    pd.patient_id = '7'
ORDER BY 
    pd.date;


    """
    cur.execute(query, (patient_id))
    rows = cur.fetchall()
    cur.close()

    # Create DataFrame and convert 'date' column to datetime
    df = pd.DataFrame(rows, columns=['date','tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos'])
    df['date'] = pd.to_datetime(df['date'])

    # Plot the data
    plt.figure(figsize=(12, 4))
    for column in df.columns[1:]:
        plt.plot(df['date'], df[column], label=column)
    
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Liver Disease Trends Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free memory

    return plot_url



    return render_template('visualization.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
