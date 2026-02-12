from flask import Flask, render_template, request, jsonify, session
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
import os
from flask import send_file
import matplotlib.pyplot as plt
from io import StringIO

app = Flask(__name__)
app.secret_key = "supersecretkey"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CLASS NAMES (MUST MATCH TRAINING ORDER EXACTLY) =====
class_names = [
    'Pepper_Bacterial_spot',
    'Pepper_healthy',
    'Potato_Late_blight',
    'Potato___Early_blight',
    'Potato_healthy',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# ===== TREATMENT DATABASE =====
treatment_dict = {
    "Pepper_Bacterial_spot": {
        "explanation": "Small dark water-soaked spots on leaves.",
        "treatment": "Use copper-based bactericides and remove infected leaves."
    },
    "Pepper_healthy": {
        "explanation": "Leaf appears healthy with no disease symptoms.",
        "treatment": "No treatment required."
    },
    "Potato_Late_blight": {
        "explanation": "Large dark lesions with mold under the leaf.",
        "treatment": "Apply fungicides like chlorothalonil immediately."
    },
    "Potato___Early_blight": {
        "explanation": "Brown spots with concentric rings on potato leaves.",
        "treatment": "Remove infected leaves and apply fungicide."
    },
    "Potato_healthy": {
        "explanation": "Healthy potato leaf.",
        "treatment": "Maintain proper irrigation and nutrition."
    },
    "Tomato_Early_blight": {
        "explanation": "Brown circular spots with concentric rings.",
        "treatment": "Use fungicides and remove infected leaves."
    },
    "Tomato_Late_blight": {
        "explanation": "Dark water-soaked lesions spreading rapidly.",
        "treatment": "Apply copper-based fungicides."
    },
    "Tomato_Leaf_Mold": {
        "explanation": "Yellow patches with mold growth under leaf surface.",
        "treatment": "Improve ventilation and apply suitable fungicide."
    },
    "Tomato_healthy": {
        "explanation": "Leaf is healthy.",
        "treatment": "No treatment required."
    }
}

# ===== LOAD MODEL =====
model = models.mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 9)

model.load_state_dict(
    torch.load("model/plant_disease_model_9class.pth", map_location=device)
)

model = model.to(device)
model.eval()

import cv2
import numpy as np

def generate_heatmap(model, image_tensor, predicted_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0][predicted_class]
    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    handle_f.remove()
    handle_b.remove()

    return cam

# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    import os

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_value = float(confidence.item() * 100)
    # ===== SEVERITY LOGIC =====

    if "healthy" in predicted_class.lower():
        severity = "Low"
    elif confidence_value > 90:
        severity = "High"
    elif confidence_value > 75:
        severity = "Moderate"
    else:
        severity = "Low"

    heatmap = generate_heatmap(model, image, predicted.item())
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224,224))

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    heatmap_path = image_path.rsplit(".", 1)[0] + "_heatmap.jpg"
    cv2.imwrite(heatmap_path, superimposed_img)
    # Get treatment info
    info = treatment_dict.get(predicted_class, {
        "explanation": "No explanation available.",
        "treatment": "Consult agricultural expert."
    })

    # Confidence safeguard
    if confidence_value < 60:
        prob_list = probabilities.squeeze().tolist()
        session["prediction_data"] = {
            "prediction": "Uncertain – Please upload supported crop types.",
            "confidence": round(confidence_value, 2),
            "probabilities_labels": class_names,
            "probabilities_values": [round(p * 100, 2) for p in prob_list],
            "explanation": "Model confidence is low for this image.",
            "treatment": "Please upload a clearer image.",
            "heatmap_path": heatmap_path,
            "image_path": image_path,
            "severity": severity
        }
        print(session["prediction_data"])
    else:
        prob_list = probabilities.squeeze().tolist()
        session["prediction_data"] = {
            "prediction": predicted_class,
            "confidence": round(confidence_value, 2),
            "probabilities_labels": class_names,
            "probabilities_values": [round(p * 100, 2) for p in prob_list],
            "explanation": info["explanation"],
            "treatment": info["treatment"],
            "heatmap_path": heatmap_path,
            "image_path": image_path,
            "severity": severity
        }
        print(session["prediction_data"])
    return jsonify({"redirect": "/report"})

@app.route("/explainability")
def explainability():
    data = session.get("prediction_data", None)
    return render_template("explainability.html", data=data)


@app.route("/report")
def report():
    data = session.get("prediction_data", None)
    return render_template("report.html", data=data)

@app.route("/insights")
def insights():
    # Replace with your actual training losses
    train_losses = [138.79, 58.70, 43.04, 39.48, 34.07]

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(train_losses)+1), train_losses,
             marker='o', color="#4ade80", linewidth=2)

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    loss_path = "static/images/loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    return render_template("insights.html")

@app.route("/chatbot")
def chatbot():
    data = session.get("prediction_data", {})
    confidence = data.get("confidence", 0)

    if confidence > 90:
        severity = "high"
    elif confidence > 75:
        severity = "moderate"
    else:
        severity = "low"

    return render_template("chatbot.html", data=data, severity=severity)


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower()
    data = session.get("prediction_data", {})

    detected_disease = data.get("prediction", "Unknown Condition")
    treatment = data.get("treatment", "Consult agricultural expert.")
    explanation = data.get("explanation", "No explanation available.")
    confidence = data.get("confidence", 0)

    # Determine severity level
    if confidence > 90:
        severity = "High"
        urgency_line = "Immediate intervention is strongly recommended."
    elif confidence > 75:
        severity = "Moderate"
        urgency_line = "Early treatment will help prevent spread."
    else:
        severity = "Low Confidence"
        urgency_line = "Please upload a clearer image for better diagnosis."

    # -------- ADVANCED RESPONSE LOGIC --------
    if any(word in user_message for word in ["treatment", "what should i do", "solution", "cure"]):
        reply = f"""
            1. Disease Summary:
            The detected condition is {detected_disease}. {explanation}

            2. Immediate Action Plan:
            • Remove visibly infected leaves.
            • Apply appropriate fungicide or bactericide.
            • Avoid overhead irrigation to prevent disease spread.
            • Monitor neighboring plants closely.

            3. Chemical Treatment:
            • Use copper-based fungicides or chlorothalonil (as applicable).
            • Follow recommended dosage strictly.

            4. Organic Alternatives:
            • Neem oil spray (weekly application).
            • Improve soil microbial balance using compost.

            5. Severity Assessment:
            Confidence: {confidence}%
            Risk Level: {severity}
            {urgency_line}
            """

    elif any(word in user_message for word in ["prevent", "avoid", "future", "precaution"]):
        reply = f"""
            1. Prevention Strategy for {detected_disease}:

            • Practice crop rotation every season.
            • Maintain adequate plant spacing.
            • Ensure proper sunlight exposure.
            • Avoid excess nitrogen fertilization.

            2. Field Monitoring:
            • Inspect plants weekly.
            • Remove early infected leaves immediately.

            3. Environmental Control:
            • Improve drainage.
            • Avoid prolonged leaf wetness.

            4. Long-Term Sustainability:
            • Use resistant crop varieties.
            • Implement Integrated Pest Management (IPM).
            """

    elif any(word in user_message for word in ["fertilizer", "soil", "nutrient", "npk"]):
        reply = f"""
        1. Soil Health Recommendation:

        • Conduct soil testing before fertilizer application.
        • Maintain soil pH between 6.0–7.0.
        • Apply balanced NPK fertilizers.

        2. Nutrient Focus:
        • Potassium improves disease resistance.
        • Avoid overuse of nitrogen.

        3. Organic Boost:
        • Add compost or vermicompost.
        • Neem cake improves pest resistance.
        """

    elif any(word in user_message for word in ["severity", "serious", "danger", "spread"]):
        reply = f"""
            1. Severity Analysis:

            Model Confidence: {confidence}%

            Risk Level: {severity}

            2. Spread Risk:
            Higher humidity and dense planting may increase spread rate.

            3. Recommended Response:
            {urgency_line}
            """
    elif "immediate" in user_message or "emergency" in user_message:
        reply = f"""
            🚨 Emergency Action Plan:

            • Isolate infected plants immediately.
            • Apply recommended fungicide within 24 hours.
            • Avoid field irrigation for 1–2 days.
            • Remove severely damaged leaves.

            Risk Level: {severity}
            Early aggressive action can significantly reduce crop loss.
            """

    elif "uncertain" in user_message or "clearer image" in user_message:
        reply = """
            Image Capture Recommendations:

            • Ensure good natural lighting.
            • Avoid shadows on leaf surface.
            • Capture a close-up of infected region.
            • Avoid blurry images.
            • Upload single leaf clearly visible.

            This improves model diagnostic accuracy.
            """

    else:
        reply = f"""
            I can assist you with:

            • Detailed treatment steps
            • Prevention strategies
            • Soil and fertilizer guidance
            • Severity analysis
            • Organic solutions

            Detected Condition: {detected_disease}
            Confidence: {confidence}%

            Please ask your question in more detail.
            """

    return jsonify({"reply": reply})
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as RLImage
from reportlab.platypus import HRFlowable
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from datetime import datetime
from flask import send_file
import os

@app.route("/download_report")
def download_report():
    data = session.get("prediction_data")

    if not data:
        return "No report available."

    file_path = "diagnosis_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()

    # Custom Styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        textColor=colors.HexColor("#14532d") , # deep forest green
        fontSize=22,
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#1D6236"),
        spaceAfter=6
    )

    normal_style = styles["Normal"]

    # ===== HEADER =====
    elements.append(Paragraph("AgroDetect AI", header_style))
    elements.append(Paragraph("AI-Powered Plant Disease Diagnostic Report", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(HRFlowable(width="100%", thickness=2,
                               color=colors.HexColor("#4ade80")))
    elements.append(Spacer(1, 0.3 * inch))

    # ===== METADATA =====
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Generated On:</b> {current_time}", normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # ===== IMAGE =====
    if "image_path" in data and os.path.exists(data["image_path"]):
        img = RLImage(data["image_path"], width=4*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.4 * inch))

    # ===== DIAGNOSIS SECTION =====
    elements.append(Paragraph("Diagnosis Summary", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#2f8f83")))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"<b>Disease Detected:</b> {data['prediction']}", normal_style))
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph(f"<b>Model Confidence:</b> {data['confidence']}%", normal_style))
    elements.append(Spacer(1, 0.15 * inch))

    # ===== SEVERITY =====
    confidence = data['confidence']
    if confidence > 90:
        severity = "High"
    elif confidence > 70:
        severity = "Medium"
    else:
        severity = "Low"

    elements.append(Paragraph(f"<b>Severity Level:</b> {severity}", normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # ===== EXPLANATION =====
    elements.append(Paragraph("Condition Analysis", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#2f8f83")))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(data['explanation'], normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # ===== TREATMENT =====
    elements.append(Paragraph("Recommended Treatment Plan", section_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#2f8f83")))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(data['treatment'], normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    # ===== CONFIDENCE CHART =====
    labels = data.get("probabilities_labels", [])
    values = data.get("probabilities_values", [])

    if labels and values:
        plt.figure(figsize=(6,4))
        plt.bar(labels, values, color="#4ade80")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Confidence (%)")
        plt.title("Model Confidence Distribution")
        plt.tight_layout()

        chart_path = "confidence_chart.png"
        plt.savefig(chart_path)
        plt.close()

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Model Confidence Distribution", section_style))
        elements.append(Spacer(1, 0.2 * inch))

        chart_img = RLImage(chart_path, width=5*inch, height=3*inch)
        elements.append(chart_img)
        elements.append(Spacer(1, 0.4 * inch))
    # ===== FOOTER =====
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#4ade80")))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Project by Amogha Vyshnavi Aiman", styles["Italic"]))
    
    doc.build(elements)

    return send_file(file_path, as_attachment=True)

@app.route("/architecture")
def architecture():
    from torchvision import models
    import torch.nn as nn

    # Recreate model exactly as trained
    model = models.mobilenet_v2(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 9)

    # Capture model summary as string
    buffer = StringIO()
    print(model, file=buffer)
    model_summary = buffer.getvalue()

    return render_template("architecture.html", summary=model_summary)

if __name__ == "__main__":
    app.run(debug=True)
