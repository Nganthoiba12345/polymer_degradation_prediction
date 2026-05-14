"""
Flask backend for Polymer Degradation & Recyclability Web App
--------------------------------------------------------------
Handles CSV upload, ensemble training, prediction, and serves API endpoints.
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

# Import your existing modules (keep them as they are)
from config import ENSEMBLE_CONFIG, DEGRADATION_CURVE
from data_loader import prepare_data
from ensemble import train_ensemble_system, get_detailed_predictions

# ============================================================
# Degradation kinetic helpers (from original Streamlit app)
# ============================================================

T_REF = DEGRADATION_CURVE["reference_days"]

def compute_k_from_Dref(D_ref: float) -> float:
    """Compute kinetic constant from degradation percentage at reference time."""
    fraction = 1.0 - min(max(D_ref, 0.0), 99.9) / 100.0
    if fraction <= 0:
        fraction = 1e-6
    return - (1.0 / T_REF) * np.log(fraction)

def degradation_at_time(t_days: float, D_ref: float) -> float:
    """Predict degradation % after t_days given reference degradation D_ref at T_REF."""
    k = compute_k_from_Dref(D_ref)
    D_t = 100.0 * (1.0 - np.exp(-k * t_days))
    return float(np.clip(D_t, 0.0, 100.0))

def degradation_curve(D_ref: float, max_days: float = None, num_points: int = 60):
    """Generate full degradation curve from 0 to max_days."""
    if max_days is None:
        max_days = T_REF
    days = np.linspace(0, max_days, num_points)
    k = compute_k_from_Dref(D_ref)
    D_t = 100.0 * (1.0 - np.exp(-k * days))
    return days.tolist(), np.clip(D_t, 0.0, 100.0).tolist()

def time_to_reach_degradation(D_target: float, D_ref: float) -> float:
    """Days needed to reach a target degradation %."""
    D_target = max(0.1, min(D_target, 99.9))
    D_ref = max(0.0, min(D_ref, 99.9))
    k = compute_k_from_Dref(D_ref)
    frac_target = 1.0 - D_target / 100.0
    if frac_target <= 0:
        frac_target = 1e-6
    return float(max(-np.log(frac_target) / k, 0.0))

# ============================================================
# Flask app initialization
# ============================================================

app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for trained model & data
trained_model = None
trained_data = None   # contains display_df, feature_cols, metrics, X_test

# ============================================================
# Main route
# ============================================================

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

# ============================================================
# API: Upload CSV
# ============================================================

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload a CSV, compute Degradation_Percent if missing, return preview."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    session['csv_path'] = filepath

    # Compute Degradation_Percent if needed (70% CO2 + 30% WaterAbsorb)
    # Exactly as done in the Streamlit app
    if 'Degradation_Percent' not in df.columns:
        if 'CO2_ppm' in df.columns and 'WaterAbsorb' in df.columns:
            # Convert WaterAbsorb column (handling '-', etc.) to numeric
            water = pd.to_numeric(
                df["WaterAbsorb"].replace("-", np.nan),
                errors="coerce"
            ).fillna(0)
            max_co2 = df['CO2_ppm'].max()
            max_wa = water.max() if water.max() != 0 else 1.0
            df['Degradation_Percent'] = (
                0.7 * (df['CO2_ppm'] / max_co2) * 100.0 +
                0.3 * (water / max_wa) * 100.0
            ).round(2)
        else:
            return jsonify({'error': 'Missing Degradation_Percent, CO2_ppm or WaterAbsorb'}), 400

    # Save processed dataframe
    df.to_csv(filepath, index=False)
    session['csv_path'] = filepath

    preview = df.head(25).to_dict(orient='records')
    return jsonify({
        'message': 'File uploaded successfully',
        'preview': preview,
        'columns': list(df.columns),
        'rows': len(df)
    })

# ============================================================
# API: Train ensemble model
# ============================================================

@app.route('/api/train', methods=['POST'])
def train():
    """Train the stacking ensemble and return metrics + predictions preview."""
    global trained_model, trained_data
    csv_path = session.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'error': 'No CSV uploaded yet'}), 400

    df = pd.read_csv(csv_path)
    target_col = 'Degradation_Percent'

    # IMPORTANT: Do NOT filter out 'Control' rows.
    # The Streamlit app trains on all rows, so metrics match.
    # (Removed the control‑filtering block that was here before)

    # Train ensemble (RandomForest + SVM/LightGBM + Stacking)
    try:
        model, X_test, y_test, metrics = train_ensemble_system(df, target_col)
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

    # Get detailed predictions from base models
    detailed = get_detailed_predictions(model, X_test)

    # Build display dataframe with input features and predictions
    feature_df = df.drop(columns=[target_col])
    feature_subset = feature_df.loc[X_test.index].reset_index(drop=True)
    display_df = pd.concat([feature_subset, detailed.reset_index(drop=True)], axis=1)

    # Final degradation column
    if 'FinalPrediction' in display_df.columns:
        display_df['Final Degradation (%)'] = display_df['FinalPrediction'].clip(0, 100).round(2)
    else:
        display_df['Final Degradation (%)'] = model.predict(X_test).clip(0, 100).round(2)

    # Add individual model predictions if present
    for col in ['RF Prediction', 'SVM Prediction', 'LGBM Prediction']:
        if col in display_df.columns:
            new_col = col.replace('Prediction', 'Degradation (%)')
            display_df[new_col] = display_df[col].clip(0, 100).round(2)

    # Confidence based on disagreement among base models
    base_pred_cols = [c for c in ['RF Prediction', 'SVM Prediction', 'LGBM Prediction'] if c in display_df.columns]
    if len(base_pred_cols) >= 2:
        disagreement = display_df[base_pred_cols].std(axis=1)
        max_disagreement = disagreement.max() if disagreement.max() > 0 else 1
        confidence = (1.0 - disagreement / max_disagreement) * 100.0
        display_df['Confidence (%)'] = np.clip(confidence, 0, 100).round(1)
    else:
        display_df['Confidence (%)'] = 100.0

    # Quality Score & Recyclability
    if 'CO2_ppm' in display_df.columns:
        display_df['CO2_norm'] = display_df['CO2_ppm'] / (display_df['CO2_ppm'].max() + 1e-6)
    else:
        display_df['CO2_norm'] = 0
    if 'WaterAbsorb' in display_df.columns:
        display_df['Water_norm'] = pd.to_numeric(display_df['WaterAbsorb'], errors='coerce').fillna(0)
        display_df['Water_norm'] = display_df['Water_norm'] / (display_df['Water_norm'].max() + 1e-6)
    else:
        display_df['Water_norm'] = 0

    display_df['Degradation_norm'] = display_df['Final Degradation (%)'] / 100.0
    display_df['Quality_Score'] = (
        0.6 * (1 - display_df['Degradation_norm']) +
        0.2 * (1 - display_df['CO2_norm']) +
        0.2 * (1 - display_df['Water_norm'])
    )
    display_df['Recyclability (%)'] = (display_df['Quality_Score'] * 100).round(2)
    display_df.loc[display_df['Final Degradation (%)'] > 70, 'Recyclability (%)'] = 0
    display_df['Recyclability (%)'] = display_df['Recyclability (%)'].clip(0, 100)

    # Recyclability classification
    def classify(r):
        if r >= 60: return 'Strong'
        elif r >= 45: return 'Moderate'
        else: return 'Weak'
    display_df['Recyclability Class'] = display_df['Recyclability (%)'].apply(classify)
    display_df['Recyclability Rank'] = display_df['Recyclability (%)'].rank(ascending=False)

    # Store in global variables
    trained_model = model
    trained_data = {
        'display_df': display_df,
        'feature_cols': list(feature_subset.columns),
        'metrics': metrics,
        'X_test': X_test
    }

    # Return metrics and first 50 rows as preview
    return jsonify({
        'metrics': metrics,
        'predictions_preview': display_df.head(50).to_dict(orient='records'),
        'feature_cols': trained_data['feature_cols']
    })

# ----------------------------------------------------------------------
# The remaining endpoints (/api/polymer_list, /api/polymer_details,
# /api/degradation_curve, /api/time_to_degradation,
# /api/recyclability_at_time) stay exactly as in your original Flask app.
# (They are already correct and do not affect the metrics.)
# ----------------------------------------------------------------------

# ============================================================
# API: Get list of polymers for dropdown
# ============================================================

@app.route('/api/polymer_list', methods=['GET'])
def polymer_list():
    """Return unique polymer identifiers (IDs) from trained data."""
    if trained_data is None or 'display_df' not in trained_data:
        return jsonify({'error': 'Model not trained yet'}), 400
    df = trained_data['display_df']
    if 'ID' in df.columns:
        polymers = [str(p) for p in df['ID'].unique() if not str(p).lower().startswith('control')]
    elif 'Polymer_Type' in df.columns:
        polymers = [str(p) for p in df['Polymer_Type'].unique() if str(p).lower() != 'control']
    else:
        polymers = list(range(len(df)))
    return jsonify({'polymers': polymers})

# ============================================================
# API: Get detailed information for a selected polymer
# ============================================================

@app.route('/api/polymer_details', methods=['POST'])
def polymer_details():
    data = request.get_json()
    polymer_id = data.get('polymer_id')
    if trained_data is None:
        return jsonify({'error': 'Model not trained'}), 400
    df = trained_data['display_df']

    # Locate the row
    if 'ID' in df.columns:
        row = df[df['ID'].astype(str) == str(polymer_id)]
    elif 'Polymer_Type' in df.columns:
        row = df[df['Polymer_Type'].astype(str) == str(polymer_id)]
    else:
        row = df.iloc[[0]]

    if row.empty:
        return jsonify({'error': 'Polymer not found'}), 404

    row_dict = row.iloc[0].to_dict()

    # ----- Extract core polymer code (e.g., from "Fiber_PET" -> "PET") -----
    polymer = row_dict.get('Polymer', row_dict.get('Polymer_name', 'Unknown'))
    core_polymer = polymer.replace("Fiber_", "").replace("Film_", "").replace("Foam_", "")
    if '-' in core_polymer:
        core_polymer = core_polymer.split('-')[0]
    base_polymer = core_polymer

    # ----- 1. Sample image (polymers_pic) -----
    actual_filenames = {
        "Film_PE": "film_PE.jpg",
        "Film_PET": "film_PET1.jpg",
        "Film_PP": "film_PP.jpg",
        "Fiber_PA": "fibre_PA.jpg",
        "Fiber_PET": "fibre_PET1.jpg",
        "Fiber_PP": "fibre_PP.jpg",
        "Foam_PS": "foam_PS.jpg",
        "Foam_PU": "foam_PU.jpg",
    }
    label_to_category = {
        "LDPE": "Film_PE",
        "PET": "Film_PET",
        "PA": "Fiber_PA",
        "PP": "Film_PP",
        "PU": "Foam_PU",
        "PS": "Foam_PS"
    }
    category = label_to_category.get(base_polymer, base_polymer)
    sample_filename = actual_filenames.get(category, f"{category}.jpg")
    sample_path = f'/static/images/polymers_pic/{sample_filename}'

    # ----- 2. Structure image -----
    structure_filename = {
        "LDPE": "ldpe.jpg", "PET": "PET.jpg", "PA": "PA.jpg",
        "PP": "PP.jpg", "PS": "PS.jpg", "PU": "PU.jpg"
    }.get(base_polymer, f"{base_polymer}.jpg")
    struct_path = f'/static/images/polymers_structure/{structure_filename}'

    # ----- 3. UV degradation image -----
    uv_filename = f"{base_polymer}_uv.png"
    uv_path = f'/static/images/polymers_uv/{uv_filename}'

    image_urls = {
        'sample': sample_path,
        'structure': struct_path,
        'degradation': uv_path
    }

    # ----- Polymer info -----
    polymer_info = {
        "LDPE": {"full_name": "Low-Density Polyethylene", "type": "Film_PE", 
                 "examples": ["Plastic carry bags", "Cling wraps", "Squeeze bottles"], 
                 "uses": ["Packaging films", "Grocery bags", "Food wrapping"]},
        "PET": {"full_name": "Polyethylene Terephthalate", "type": "Film_PET", 
                "examples": ["Water bottles", "Soda bottles", "Polyester film"], 
                "uses": ["Beverage bottles", "Food containers", "Textile fibers"]},
        "PA": {"full_name": "Polyamide / Nylon", "type": "Fiber_PA", 
               "examples": ["Nylon rope", "Fishing nets", "Sportswear"], 
               "uses": ["Textile fibers", "Parachutes", "Ropes"]},
        "PP": {"full_name": "Polypropylene", "type": "Film_PP", 
               "examples": ["Snack wrappers", "Yogurt containers", "Reusable plastic boxes"], 
               "uses": ["Food packaging", "Automotive parts", "Medical containers"]},
        "PU": {"full_name": "Polyurethane", "type": "Foam_PU", 
               "examples": ["Sofa foam", "Mattress foam", "Insulation foam"], 
               "uses": ["Furniture cushioning", "Thermal insulation", "Footwear soles"]},
        "PS": {"full_name": "Polystyrene", "type": "Foam_PS", 
               "examples": ["Thermocol cups", "Packaging foam", "Disposable plates"], 
               "uses": ["Protective packaging", "Disposable utensils", "Insulation panels"]}
    }
    info = polymer_info.get(base_polymer, {"full_name": base_polymer, "type": "Unknown", "examples": [], "uses": []})

    feature_cols = trained_data['feature_cols']
    input_params = {col: row_dict.get(col, 'N/A') for col in feature_cols if col in row_dict}

    return jsonify({
        'polymer': polymer,
        'base_polymer': base_polymer,
        'info': info,
        'images': image_urls,
        'degradation': row_dict.get('Final Degradation (%)', 0),
        'recyclability': row_dict.get('Recyclability (%)', 0),
        'recyclability_class': row_dict.get('Recyclability Class', 'Unknown'),
        'confidence': row_dict.get('Confidence (%)', 0),
        'input_params': input_params
    })

# ============================================================
# API: Degradation curve (time vs degradation %)
# ============================================================

@app.route('/api/degradation_curve', methods=['POST'])
def degradation_curve_api():
    """Return days and corresponding degradation % for the selected polymer."""
    data = request.get_json()
    polymer_id = data.get('polymer_id')
    if trained_data is None:
        return jsonify({'error': 'Model not trained'}), 400
    df = trained_data['display_df']

    if 'ID' in df.columns:
        row = df[df['ID'].astype(str) == str(polymer_id)]
    elif 'Polymer_Type' in df.columns:
        row = df[df['Polymer_Type'].astype(str) == str(polymer_id)]
    else:
        row = df.iloc[[0]]

    if row.empty:
        return jsonify({'error': 'Polymer not found'}), 404

    D_ref = float(row.iloc[0]['Final Degradation (%)'])
    days, deg_values = degradation_curve(D_ref)
    return jsonify({'days': days, 'degradation': deg_values})

# ============================================================
# API: Time required to reach a target degradation %
# ============================================================

@app.route('/api/time_to_degradation', methods=['POST'])
def time_to_degradation_api():
    """Calculate days needed to reach a specified degradation %."""
    data = request.get_json()
    polymer_id = data.get('polymer_id')
    target_percent = data.get('target_percent', 50)
    if trained_data is None:
        return jsonify({'error': 'Model not trained'}), 400
    df = trained_data['display_df']

    if 'ID' in df.columns:
        row = df[df['ID'].astype(str) == str(polymer_id)]
    elif 'Polymer_Type' in df.columns:
        row = df[df['Polymer_Type'].astype(str) == str(polymer_id)]
    else:
        row = df.iloc[[0]]

    if row.empty:
        return jsonify({'error': 'Polymer not found'}), 404

    D_ref = float(row.iloc[0]['Final Degradation (%)'])
    t_days = time_to_reach_degradation(target_percent, D_ref)
    return jsonify({'days': t_days})

# ============================================================
# API: Recyclability at a specific time (days)
# ============================================================

@app.route('/api/recyclability_at_time', methods=['POST'])
def recyclability_at_time():
    """Compute degradation % and recyclability % after a given number of days."""
    data = request.get_json()
    polymer_id = data.get('polymer_id')
    time_days = data.get('time_days', 30)
    if trained_data is None:
        return jsonify({'error': 'Model not trained'}), 400
    df = trained_data['display_df']

    if 'ID' in df.columns:
        row = df[df['ID'].astype(str) == str(polymer_id)]
    elif 'Polymer_Type' in df.columns:
        row = df[df['Polymer_Type'].astype(str) == str(polymer_id)]
    else:
        row = df.iloc[[0]]

    if row.empty:
        return jsonify({'error': 'Polymer not found'}), 404

    D_ref = float(row.iloc[0]['Final Degradation (%)'])
    D_time = degradation_at_time(time_days, D_ref)
    quality = row.iloc[0].get('Quality_Score', 0.5)
    if pd.isna(quality):
        quality = 0.5
    recyclability = (1 - D_time/100) * 100 * (0.7 + 0.3 * quality)
    if D_time > 70:
        recyclability = 0
    recyclability = max(0, min(100, recyclability))
    return jsonify({
        'degradation_percent': round(D_time, 2),
        'recyclability_percent': round(recyclability, 2)
    })

# ============================================================
# Run the app
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)