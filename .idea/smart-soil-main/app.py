import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

import colorsys
from PIL import Image


def is_soil_color_image(file_path):
    """Check if image has soil-like colors (browns, grays, tans)"""
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((100, 100))  # Resize for faster processing

        pixels = np.array(img)
        pixels = pixels.reshape(-1, 3)

        soil_color_pixels = 0
        total_pixels = len(pixels)

        for pixel in pixels:
            r, g, b = pixel
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            # Soil colors typically fall in these HSV ranges:
            # Brown: H: 0.05-0.15, S: 0.3-0.8, V: 0.2-0.6
            # Tan: H: 0.08-0.18, S: 0.2-0.6, V: 0.4-0.8
            # Gray: S < 0.3, V: 0.2-0.7

            is_soil_color = (
                    (0.05 <= h <= 0.18 and 0.2 <= s <= 0.8 and 0.2 <= v <= 0.8) or  # Browns/Tans
                    (s < 0.3 and 0.2 <= v <= 0.7)  # Grays
            )

            if is_soil_color:
                soil_color_pixels += 1

        soil_ratio = soil_color_pixels / total_pixels
        return soil_ratio > 0.6  # At least 60% should be soil colors

    except Exception as e:
        print(f"Color check error: {e}")
        return False


# ------------------------------------------------------------
# Flask app setup
# ------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ------------------------------------------------------------
# Load trained model once
# ------------------------------------------------------------
MODEL_PATH = 'soil_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match your training folder names)
CLASS_LABELS = ['Alluvial', 'Black', 'Cinder', 'Clay', 'Laterite', 'Peat', 'Red', 'Yellow']


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_recommendations_with_ph(soil_type, ph_value):
    """Path A: When farmer knows pH value - find closest match"""
    df = pd.read_csv('Dataset/phDataset.csv')
    df['Soil_Type'] = df['Soil_Type'].astype(str).str.lower().str.strip()
    df['pH_Value'] = pd.to_numeric(df['pH_Value'], errors='coerce')

    # First try: Find in the specific soil type
    soil_matches = df[df['Soil_Type'].str.contains(soil_type.lower(), na=False)]

    if not soil_matches.empty:
        # Find the closest pH value within THIS soil type
        soil_matches['diff'] = abs(soil_matches['pH_Value'] - ph_value)
        best_match_in_soil = soil_matches.loc[soil_matches['diff'].idxmin()]

        # If we found a reasonable match in the same soil type, use it
        if abs(best_match_in_soil['pH_Value'] - ph_value) <= 0.5:
            return best_match_in_soil, best_match_in_soil['pH_Value'], "exact_same_soil", soil_type

    # Second try: Find in ANY soil type (global search)
    df['diff'] = abs(df['pH_Value'] - ph_value)
    best_match_global = df.loc[df['diff'].idxmin()]
    original_soil_type = best_match_global['Soil_Type'].title()

    return best_match_global, best_match_global['pH_Value'], "exact_other_soil", original_soil_type


def get_recommendations_without_ph(soil_type):
    """Path B: When farmer doesn't know pH value - use aggregated dataset"""
    df_agg = pd.read_csv('Dataset/phAggregatedDataset.csv')
    df_agg['Soil_Type'] = df_agg['Soil_Type'].astype(str).str.lower().str.strip()

    # Find soil type in aggregated dataset
    soil_match = df_agg[df_agg['Soil_Type'].str.contains(soil_type.lower(), na=False)]

    if not soil_match.empty:
        best_match = soil_match.iloc[0]  # Take first match
        return best_match, best_match['Average_pH'], "aggregated", soil_type

    return None, None, "no_match", soil_type


def is_valid_soil_image(confidence, pred):
    """Relaxed validation for soil images - accept most predictions"""
    print(f"DEBUG: Checking validation - Confidence: {confidence:.2f}%")

    # Accept if confidence is reasonable
    if confidence > 50.0:
        return True

    # For very low confidence, do basic check
    sorted_probs = sorted(pred[0], reverse=True)
    if len(sorted_probs) > 1:
        if sorted_probs[0] - sorted_probs[1] > 0.15:  # Clear winner
            return True

    return False


def get_ph_warning(ph_value):
    """Get warning message for extreme pH values"""
    if ph_value < 4.0:
        return "<p style='background:#FFEBEE;padding:10px;border-radius:5px;color:#C62828;border-left:4px solid #C62828;'>🚨 <b>Extreme pH Warning:</b> pH value below 4.0 indicates strongly acidic soil that may require immediate remediation. Consult with agricultural expert.</p>"
    elif ph_value > 8.0:
        return "<p style='background:#FFEBEE;padding:10px;border-radius:5px;color:#C62828;border-left:4px solid #C62828;'>🚨 <b>Extreme pH Warning:</b> pH value above 8.0 indicates strongly alkaline soil that may affect nutrient availability. Consult with agricultural expert.</p>"
    return ""


def create_result_html(predicted_label, confidence, best_match, ph_used, match_type, original_soil_type, ph_value=None):
    """Create HTML result for both paths"""

    # pH extreme warning
    ph_warning = ""
    if ph_value:
        ph_warning = get_ph_warning(ph_value)

    if match_type == "exact_same_soil":
        ph_note = f"Submitted pH Value: <b>{ph_value}</b> | Matched in {predicted_label}: {ph_used}"

    elif match_type == "exact_other_soil":
        ph_note = f"Submitted pH Value: <b>{ph_value}</b> | Data sourced from {original_soil_type} (pH {ph_used})"

    elif match_type == "aggregated":
        ph_note = f"🌡️ <b>Using Average pH for {predicted_label}: {ph_used}</b> (based on soil type analysis)"

    else:
        return f"""
        <div class="prediction-result error">
            <p>⚠️ No recommendations found for {predicted_label}.</p>
        </div>
        """

    # Determine which columns to use based on dataset
    if match_type == "aggregated":
        best_crops = best_match['Best_Crops']
        optional_crops = best_match['Optional_Crops']
        not_recommended_crops = best_match['Not_Recommended_Crops']
        best_fertilizer = best_match['Best_Fertilizers']
        optional_fertilizer = best_match['Optional_Fertilizers']
        not_recommended_fertilizer = best_match['Not_Recommended_Fertilizers']
    else:
        best_crops = best_match['Best_Crop_Recommendation']
        optional_crops = best_match['Optional_Crop_Recommendation']
        not_recommended_crops = best_match['Crop_Not_Recommended']
        best_fertilizer = best_match['Best_Fertilizer']
        optional_fertilizer = best_match['Optional_Fertilizer']
        not_recommended_fertilizer = best_match['Fertilizer_Not_Recommended']

    return f"""
    <div class="prediction-result">
        <div class="result-header">
            <h3>🧠 Predicted Soil Type: <b>{predicted_label}</b> 
            <span style='font-size:0.9em;color:#666;'>(Confidence: {confidence:.2f}%)</span></h3>
            <p class="ph-note">{ph_note}</p>
            {ph_warning}
        </div>

        <div class="result-grid">
            <div class="result-card highlight">
                <h4>🌾 Best Crop Recommendation</h4>
                <p>{best_crops}</p>
            </div>
            <div class="result-card">
                <h4>🌿 Optional Crops</h4>
                <p>{optional_crops}</p>
            </div>
            <div class="result-card">
                <h4>🚫 Crops Not Recommended</h4>
                <p>{not_recommended_crops}</p>
            </div>
        </div>

        <div class="additional-info">
            <h4>🧪 Fertilizer Recommendations</h4>
            <div class="info-grid">
                <div class="info-item"><span class="icon">🌱</span><b>Best Fertilizer:</b> {best_fertilizer}</div>
                <div class="info-item"><span class="icon">🧴</span><b>Optional Fertilizer:</b> {optional_fertilizer}</div>
                <div class="info-item"><span class="icon">🚫</span><b>Not Recommended:</b> {not_recommended_fertilizer}</div>
            </div>
        </div>
    </div>
    """


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html', show_result=False)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from form
        soil_image = request.files['soil_image']
        phone_number = request.form.get('phone_number')

        # Check if pH value is provided (Path A vs Path B)
        ph_input = request.form.get('ph_value', '').strip()
        has_ph_value = bool(ph_input)

        if has_ph_value:
            ph_value = round(float(ph_input), 1)
        else:
            ph_value = None

        # Save uploaded image
        filename = secure_filename(soil_image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        soil_image.save(file_path)

        # Preprocess image for prediction
        img = image.load_img(file_path, target_size=(150, 150))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # ORIGINAL SIMPLE PREDICTION - WORKING VERSION
        pred = model.predict(x)
        predicted_index = np.argmax(pred)
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = pred[0][predicted_index] * 100

        # STRICT VALIDATION: Check if image is actually soil
        color_check = is_soil_color_image(file_path)
        confidence_check = is_valid_soil_image(confidence, pred)

        if not color_check or not confidence_check:
            # Get image info for debugging
            img_pil = Image.open(file_path)
            img_size = img_pil.size

            return render_template(
                'index.html',
                show_result=True,
                result=f"""
                <div class='prediction-result error'>
                    <h3>🚫 Invalid Image Detected</h3>
                    <p><b>This doesn't appear to be a soil image.</b></p>

                    <div style='background:#FFEBEE;padding:15px;border-radius:8px;margin:15px 0;'>
                        <h4 style='color:#C62828;margin-top:0;'>Detection Results:</h4>
                        <ul>
                            <li>📊 Color Analysis: <b>{'FAILED' if not color_check else 'PASSED'}</b></li>
                            <li>🤖 AI Confidence: <b>{confidence:.2f}%</b> ({predicted_label})</li>
                            <li>🖼️ Image Size: <b>{img_size[0]}x{img_size[1]} pixels</b></li>
                        </ul>
                    </div>

                    <p><b>❌ Why this was rejected:</b></p>
                    <ul>
                        <li>Image doesn't contain typical soil colors (browns, tans, grays)</li>
                        <li>Low confidence in soil detection</li>
                        <li>May contain text, objects, or non-soil elements</li>
                    </ul>

                    <p><b>✅ Upload a proper soil image:</b></p>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:15px 0;'>
                        <div style='background:#E8F5E9;padding:15px;border-radius:8px;'>
                            <h4 style='color:#2E7D32;'>✔️ DO Upload</h4>
                            <ul>
                                <li>Close-up of bare soil</li>
                                <li>Natural soil colors</li>
                                <li>Good lighting</li>
                                <li>Clear focus</li>
                            </ul>
                        </div>
                        <div style='background:#FFEBEE;padding:15px;border-radius:8px;'>
                            <h4 style='color:#C62828;'>❌ DON'T Upload</h4>
                            <ul>
                                <li>Documents/text</li>
                                <li>People/objects</li>
                                <li>Screenshots</li>
                                <li>Blurry images</li>
                            </ul>
                        </div>
                    </div>

                    <p><em>Please upload a clear image of actual soil for accurate analysis.</em></p>
                </div>
                """
            )

        # ------------------------------------------------------------
        # Get recommendations based on whether pH is provided
        # ------------------------------------------------------------
        if has_ph_value:
            # Path A: Farmer knows pH
            best_match, ph_used, match_type, original_soil_type = get_recommendations_with_ph(predicted_label, ph_value)
        else:
            # Path B: Farmer doesn't know pH
            best_match, ph_used, match_type, original_soil_type = get_recommendations_without_ph(predicted_label)

        # Create result HTML
        if best_match is not None:
            result_html = create_result_html(predicted_label, confidence, best_match, ph_used, match_type,
                                             original_soil_type, ph_value)
        else:
            result_html = f"""
            <div class="prediction-result error">
                <p>⚠️ No data found for analysis.</p>
            </div>
            """

        # Optional SMS message
        sms_message = None
        if phone_number:
            sms_message = f"Results sent to {phone_number} (demo only)."

        # Render back to index.html
        return render_template(
            'index.html',
            show_result=True,
            image_url=filename,
            result=result_html,
            sms_message=sms_message
        )

    except Exception as e:
        return render_template(
            'index.html',
            show_result=True,
            result=f"<div class='prediction-result error'><p>❌ Error: {str(e)}</p></div>"
        )


# ------------------------------------------------------------
# Run app
# ------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)