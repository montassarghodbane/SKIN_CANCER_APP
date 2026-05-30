import os
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# TensorFlow / Keras et TensorFlow Hub pour le modèle ViT
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
# Clé secrète indispensable pour chiffrer les sessions de connexion
app.secret_key = 'super_secret_key_for_skin_cancer_app'

# =================================================================
# 1. CONFIGURATION DE LA BASE DE DONNÉES & DES DOSSIERS
# =================================================================

# Connexion à MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skin_cancer_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration du dossier d'upload
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# =================================================================
# 2. CHARGEMENT DU MODÈLE ViT (VISION TRANSFORMER)
# =================================================================

# Définition de la classe personnalisée pour charger le module TF Hub
class HubFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, feature_extractor_module, **kwargs):
        super(HubFeatureExtractor, self).__init__(**kwargs)
        self.feature_extractor_module = feature_extractor_module

    def call(self, inputs, training=False):
        return self.feature_extractor_module(inputs)

# Chemin vers vos poids téléchargés (Assurez-vous que le fichier est dans un dossier 'model')
MODEL_WEIGHTS_PATH = os.path.join('model', 'modele_poids.weights.h5')

try:
    print("Création de l'architecture du modèle ViT...")
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    feature_extractor_module = hub.load(vit_url)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = HubFeatureExtractor(feature_extractor_module, trainable=False)(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_weights(MODEL_WEIGHTS_PATH)
        print(f"-> SUCCÈS : Les connaissances de l'IA (poids) ont été chargées depuis '{MODEL_WEIGHTS_PATH}' !")
    else:
        print(f"⚠️ ERREUR CRITIQUE : Le fichier est introuvable au chemin : {MODEL_WEIGHTS_PATH}")
        model = None
except Exception as e:
    print(f"⚠️ ERREUR lors de l'initialisation du modèle : {e}")
    model = None

# =================================================================
# 3. STRUCTURE DES TABLES DE LA BASE DE DONNÉES (MODÈLES ORM)
# =================================================================

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    propability = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =================================================================
# 4. LOGIQUE MÉTIER & ROUTAGE DE L'APPLICATION
# =================================================================

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Identifiants incorrects. Veuillez réessayer.")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            return render_template('register.html', error="Les mots de passe ne correspondent pas.")
            
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Ce nom d'utilisateur est déjà pris.")
            
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        return render_template('login.html', success="Compte créé avec succès ! Connectez-vous.")
        
    return render_template('register.html')

# --- ROUTE : MOT DE PASSE OUBLIÉ ---
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    from flask import request, render_template
    
    if request.method == 'POST':
        username = request.form.get('username')
        message_succes = f"Si l'utilisateur '{username}' existe, les instructions ont été envoyées."
        return render_template('forgot_password.html', success=message_succes)
        
    return render_template('forgot_password.html')
    
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['GET'])
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/submit_analysis', methods=['POST'])
def submit_analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    file = request.files.get('lesion_image')
    
    if not file or not patient_name or not patient_age:
        return render_template('predict.html', error="Veuillez remplir tous les champs obligatoires.")
    
    if model is None:
        return render_template('predict.html', error="Le modèle d'IA n'est pas chargé sur le serveur.")

    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    try:
        # PRÉTRAITEMENT POUR LE MODÈLE ViT
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # PRÉDICTION
        prediction_score = model.predict(img_array)[0][0]
        
        if prediction_score > 0.5:
            result_label = "Malignant"
            confidence = float(prediction_score * 100)
        else:
            result_label = "Benign"
            confidence = float((1 - prediction_score) * 100)
            
        confidence = round(confidence, 2)

        new_patient = Patient(
            name=patient_name,
            age=int(patient_age),
            result=result_label,
            propability=confidence,
            image_path=f"/{file_path.replace(os.sep, '/')}" 
        )
        db.session.add(new_patient)
        db.session.commit()
        
        return redirect(url_for('result', patient_id=new_patient.id))

    except Exception as e:
        return render_template('predict.html', error=f"Erreur IA : {str(e)}")

@app.route('/result/<int:patient_id>')
def result(patient_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    patient = Patient.query.get_or_404(patient_id)
    return render_template('result.html', 
                           prediction=patient.result, 
                           confidence=patient.propability, 
                           image_url=patient.image_path)

@app.route('/patients_history')
def patients_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    db_patients = Patient.query.order_by(Patient.created_at.desc()).all()
    
    patients_list = []
    for p in db_patients:
        patients_list.append({
            'image_url': p.image_path,
            'name': p.name,
            'age': p.age,
            'date': p.created_at.strftime('%d/%m/%Y à %H:%M'),
            'prediction': p.result,
            'confidence': p.propability
        })
        
    return render_template('patients.html', patients=patients_list)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Crée automatiquement la table si elle a été supprimée
    app.run(debug=True)