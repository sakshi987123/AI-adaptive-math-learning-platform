from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import os
import shap
from gpt_lesson_generator import GPTLessonGenerator
from user_profile import upsert_user, get_user, save_quiz_session, get_full_profile

app = Flask(__name__)

# Get the parent directory of the backend folder
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)

# Load the classification model and scaler
model = load(os.path.join(parent_dir, 'best_model.pkl'))
scaler = load(os.path.join(parent_dir, 'scaler.pkl'))

# Initialize SHAP explainer
explainer = shap.Explainer(model)

# Initialize GPT lesson generator
gpt_generator = GPTLessonGenerator()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Expected features: math_score, reading_score, writing_score, score_percentage
    features = [
        data['math_score'],
        data['reading_score'],
        data['writing_score'],
        data['score_percentage']
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict
    prediction_num = model.predict(features_scaled)[0]
    # Assuming 0: Slow, 1: Advanced
    level = 'Advanced Learner' if prediction_num == 1 else 'Slow Learner'

    # Compute SHAP values for explanation
    shap_values = explainer.shap_values(features_scaled)
    feature_names = ['Math Score', 'Reading Score', 'Writing Score', 'Score Percentage']
    
    # For binary classification, shap_values is a list of arrays for each class
    if isinstance(shap_values, list):
        shap_vals = shap_values[prediction_num]  # For the predicted class
    else:
        shap_vals = shap_values[0]  # For single output
    
    # Generate explanation text
    explanation = f"You were classified as {level} based on your quiz performance. "
    explanation += "The key factors influencing this classification are:\n"
    for i, (name, val) in enumerate(zip(feature_names, shap_vals[0])):
        impact = "increased" if val > 0 else "decreased"
        explanation += f"- {name}: {abs(val):.3f} impact ({impact} likelihood of {level})\n"
    explanation += "\nThis analysis uses SHAP (SHapley Additive exPlanations) to explain the model's decision."

    return jsonify({'level': level, 'explanation': explanation})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/generate_lesson', methods=['GET'])
def generate_lesson():
    level = request.args.get('level', 'Slow').strip()
    
    # Use Gemini to generate lesson for the specific learner level
    script, source = gpt_generator.generate(level=level)
    
    # Split into natural sentence chunks (split on ". " to keep sentences intact)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group into chunks of 3 sentences each for smoother TTS playback
    slides = []
    chunk_size = 3
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk:
            slides.append(chunk)
    
    return jsonify({
        'level': level.title(),
        'script': script,
        'slides': slides,
        'source': source
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '').strip()
    level = data.get('level', 'slow').strip().lower()

    if not question:
        return jsonify({'answer': 'Please ask a question.', 'source': 'error'})

    level_context = (
        "The student is a slow learner. Use very simple language and short sentences. Be patient and encouraging."
        if 'slow' in level else
        "The student is an advanced learner. You can use precise language and include deeper insights."
    )

    prompt = f"""You are a helpful, friendly AI assistant. {level_context}

The student asks: "{question}"

Rules:
- Answer ANY question — math, science, history, general knowledge, or anything else.
- Give a clear, complete, helpful answer.
- Keep it concise: 2 to 5 sentences maximum.
- Write in plain spoken English only — absolutely no markdown, no bullet points, no asterisks, no hash symbols, no numbered lists.
- Speak naturally as if talking to the student directly."""

    import requests as req
    import re
    import time
    from gpt_lesson_generator import GEMINI_API_KEY

    # Use gemini-flash-lite-latest for chat (separate quota)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 600
        }
    }

    last_error = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2)
            r = req.post(url, json=payload, timeout=20)
            r.raise_for_status()
            answer = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            answer = re.sub(r'\*+', '', answer)
            answer = re.sub(r'#+\s*', '', answer)
            answer = re.sub(r'\n+', ' ', answer).strip()
            print(f"[Chat] ✅ Answer: {answer[:80]}...")
            return jsonify({'answer': answer, 'source': 'api'})
        except Exception as e:
            last_error = e
            print(f"[Chat] Attempt {attempt+1} failed: {e}")

    print(f"[Chat] All attempts failed: {last_error}")
    return jsonify({'answer': 'Sorry, the AI is busy right now. Please wait a moment and try again.', 'source': 'error'})



@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyzes quiz answers using:
    1. Rule-based system — maps questions to topics, identifies weak areas
    2. Explainable AI — calculates topic impact scores (SHAP-style)
    3. Generative AI — produces personalized improvement plan
    """
    import requests as req
    import re
    from gpt_lesson_generator import GEMINI_API_KEY

    data = request.json
    answers = data.get('answers', [])
    level = data.get('level', 'slow').lower()

    if not answers:
        return jsonify({'error': 'No answers provided'}), 400

    # ── 1. RULE-BASED: Topic tagging ──
    topic_map = {
        "theorem": ["theorem", "pythagoras", "state", "formula", "relationship"],
        "hypotenuse": ["hypotenuse", "longest", "opposite", "right angle", "name"],
        "calculation": ["squared", "compute", "calculate", "sum", "add", "leg", "3", "4", "6", "8"],
        "square_root": ["square root", "value of c", "c squared equals", "√"],
        "application": ["ladder", "wall", "real", "problem", "distance", "height", "triple"]
    }

    topic_results = {t: {'correct': 0, 'wrong': 0, 'questions': []} for t in topic_map}

    for ans in answers:
        q_text = ans.get('text', '').lower()
        matched_topic = 'calculation'  # default
        for topic, keywords in topic_map.items():
            if any(kw in q_text for kw in keywords):
                matched_topic = topic
                break
        topic_results[matched_topic]['questions'].append(ans.get('text', ''))
        if ans.get('correct'):
            topic_results[matched_topic]['correct'] += 1
        else:
            topic_results[matched_topic]['wrong'] += 1

    # ── 2. EXPLAINABLE AI: Impact scores ──
    total_questions = len(answers)
    topic_impacts = {}
    weak_topics = []
    strong_topics = []

    for topic, res in topic_results.items():
        total = res['correct'] + res['wrong']
        if total == 0:
            continue
        accuracy = res['correct'] / total
        impact = round(res['wrong'] / total_questions, 3)  # contribution to total score loss
        topic_impacts[topic] = {
            'accuracy': round(accuracy * 100),
            'impact': impact,
            'correct': res['correct'],
            'wrong': res['wrong'],
            'total': total
        }
        if accuracy < 0.6:
            weak_topics.append(topic)
        elif accuracy >= 0.8:
            strong_topics.append(topic)

    # Sort by impact (highest loss first)
    sorted_impacts = sorted(topic_impacts.items(), key=lambda x: x[1]['impact'], reverse=True)

    topic_labels = {
        'theorem': 'Understanding the Theorem',
        'hypotenuse': 'Identifying the Hypotenuse',
        'calculation': 'Squaring and Adding',
        'square_root': 'Finding Square Roots',
        'application': 'Real-World Application'
    }

    # ── 3. GENERATIVE AI: Personalized improvement plan ──
    weak_labels = [topic_labels.get(t, t) for t in weak_topics]
    strong_labels = [topic_labels.get(t, t) for t in strong_topics]
    score = sum(1 for a in answers if a.get('correct'))
    pct = round(score / total_questions * 100)

    level_context = "The student is a slow learner — use very simple language." if 'slow' in level else "The student is an advanced learner — use precise language."

    prompt = f"""You are an AI tutor analyzing a blind student's quiz performance on the Pythagoras theorem.
{level_context}

Quiz results:
- Score: {score} out of {total_questions} ({pct}%)
- Weak topics: {', '.join(weak_labels) if weak_labels else 'None'}
- Strong topics: {', '.join(strong_labels) if strong_labels else 'None'}

Generate a short, encouraging improvement plan with:
1. One sentence acknowledging their performance
2. The top 1-2 areas to focus on (based on weak topics)
3. One specific action they should take next
4. One motivational closing sentence

Write in plain spoken English only — no markdown, no bullet points, no asterisks. Keep it under 5 sentences total. Make it sound warm and personal."""

    improvement_text = "Keep practicing the topics you found difficult. Review the lesson and try the quiz again."
    ai_source = 'rule-based'

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300}
        }
        r = req.post(url, json=payload, timeout=20)
        r.raise_for_status()
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        improvement_text = re.sub(r'\*+', '', raw)
        improvement_text = re.sub(r'#+\s*', '', improvement_text)
        improvement_text = re.sub(r'\n+', ' ', improvement_text).strip()
        ai_source = 'gemini'
    except Exception as e:
        print(f"[Analyze] Gemini error: {e}")

    return jsonify({
        'score': score,
        'total': total_questions,
        'percentage': pct,
        'topic_impacts': dict(sorted_impacts),
        'topic_labels': topic_labels,
        'weak_topics': weak_topics,
        'strong_topics': strong_topics,
        'improvement_text': improvement_text,
        'ai_source': ai_source
    })



# ── USER PROFILE APIs ────────────────────────────────────────────────────────

@app.route('/user/save', methods=['POST'])
def user_save():
    """Save or update user profile from Firebase auth data."""
    data = request.json
    uid = data.get('uid', '').strip()
    if not uid:
        return jsonify({'error': 'uid required'}), 400
    upsert_user(
        uid=uid,
        name=data.get('name', ''),
        email=data.get('email', ''),
        photo_url=data.get('photo_url', ''),
        disability_type=data.get('disability_type', 'blind')
    )
    return jsonify({'status': 'ok', 'uid': uid})


@app.route('/user/profile', methods=['GET'])
def user_profile():
    """Get full user profile + learning history + progress."""
    uid = request.args.get('uid', '').strip()
    if not uid:
        return jsonify({'error': 'uid required'}), 400
    profile = get_full_profile(uid)
    if not profile:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(profile)


@app.route('/user/save_result', methods=['POST'])
def user_save_result():
    """
    Save a quiz result to the user's learning history.
    Combines Rule-based + XAI + ANN + LSTM analysis.
    """
    data = request.json
    uid = data.get('uid', '').strip()
    if not uid:
        return jsonify({'error': 'uid required'}), 400

    session_data = {
        'quiz_type':        data.get('quiz_type', 'evaluation'),
        'disability':       data.get('disability', 'blind'),
        'level':            data.get('level', 'slow'),
        'pre_score':        data.get('pre_score', 0),
        'accuracy':         data.get('accuracy', 0),
        'total_questions':  data.get('total_questions', 5),
        'correct':          data.get('correct', 0),
        'wrong':            data.get('wrong', 0),
        'avg_response_time': data.get('avg_response_time', 0),
        'max_streak':       data.get('max_streak', 0),
        'weak_topics':      data.get('weak_topics', []),
        'strong_topics':    data.get('strong_topics', []),
        'topic_impacts':    data.get('topic_impacts', {}),
        'improvement_text': data.get('improvement_text', ''),
        'answers_detail':   data.get('answers_detail', []),
    }

    result = save_quiz_session(uid, session_data)
    return jsonify({'status': 'saved', 'behavior': result['behavior'], 'overall_behavior': result['overall_behavior']})


@app.route('/user/history', methods=['GET'])
def user_history():
    """Get user's score history and behavior trend."""
    uid = request.args.get('uid', '').strip()
    if not uid:
        return jsonify({'error': 'uid required'}), 400
    profile = get_full_profile(uid)
    if not profile:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'score_history':    profile['progress'].get('score_history', []),
        'behavior_history': profile['progress'].get('behavior_history', []),
        'overall_behavior': profile['progress'].get('overall_behavior', 'stable'),
        'weighted_score':   profile['progress'].get('weighted_score', 0),
        'overall_accuracy': profile['progress'].get('overall_accuracy', 0),
        'persistent_weak_topics': profile['progress'].get('persistent_weak_topics', []),
        'recent_sessions':  profile['recent_sessions']
    })


if __name__ == '__main__':
    app.run(debug=True)
