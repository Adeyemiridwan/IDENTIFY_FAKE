from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("app/model/model.pkl", "rb"))
vectorizer = pickle.load(open("app/model/vectorizer.pkl", "rb"))


# =========================
# EXPLAIN ENGINE
# =========================
def explain_text(text):
    reasons = []
    text_lower = text.lower()

    fake_keywords = ["win", "click", "urgent", "now", "reward", "prize", "free", "money"]

    for word in fake_keywords:
        if word in text_lower:
            reasons.append(f"Suspicious keyword: {word}")

    if "!" in text:
        reasons.append("Excessive exclamation marks")

    if len(text.split()) < 5:
        reasons.append("Very short message (common scam pattern)")

    return reasons


# =========================
# RISK ENGINE
# =========================
def calculate_risk(text, confidence):
    score = 0
    text_lower = text.lower()

    risky_words = ["win", "click", "urgent", "now", "reward", "prize", "free", "money"]

    for word in risky_words:
        if word in text_lower:
            score += 10

    if "!" in text:
        score += 10

    if len(text.split()) < 5:
        score += 10

    score += confidence * 0.5

    return min(round(score, 2), 100)


def get_risk_level(score):
    if score < 30:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    return "HIGH"


# =========================
# WORD HIGHLIGHTING
# =========================
def highlight_words(text):
    risky_words = ["win", "click", "urgent", "now", "reward", "prize", "free", "money"]

    for word in risky_words:
        if word in text.lower():
            text = text.replace(word, f"[{word.upper()}]")

    return text


# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    # SAFE DEFAULTS
    prediction = None
    confidence = None
    reasons = []
    risk_score = None
    risk_level = None
    user_text = ""
    highlighted_text = ""

    if request.method == "POST":
        user_text = request.form["text"]

        if user_text.strip():

            # VECTORIZE
            text_vector = vectorizer.transform([user_text])

            # PREDICT
            result = model.predict(text_vector)
            probability = model.predict_proba(text_vector)

            prediction_raw = result[0]
            confidence = round(max(probability[0]) * 100, 2)

            # UNCERTAINTY
            is_uncertain = confidence < 60

            # RISK
            risk_score = calculate_risk(user_text, confidence)
            risk_level = get_risk_level(risk_score)

            # HIGHLIGHT
            highlighted_text = highlight_words(user_text)

            # DECISION LOGIC
            if is_uncertain:
                prediction = "⚠️ UNCERTAIN"
                reasons = ["Low confidence prediction"] + explain_text(user_text)

            elif prediction_raw == "fake":
                prediction = "⚠️ FAKE"
                reasons = explain_text(user_text)

            else:
                prediction = "✅ REAL"
                reasons = ["No strong scam signals detected"]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        reasons=reasons,
        risk_score=risk_score,
        risk_level=risk_level,
        user_text=user_text,
        highlighted_text=highlighted_text
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)