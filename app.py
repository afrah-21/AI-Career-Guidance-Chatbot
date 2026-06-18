from flask import Flask, render_template, request, jsonify
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Load FAQ data
df = pd.read_csv("faqs.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_questions = [clean_text(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(cleaned_questions)

def get_answer(user_question):
    user_question_clean = clean_text(user_question)
    user_vector = vectorizer.transform([user_question_clean])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score >= 0.25:
        return answers[best_match_index]

    if any(word in user_question_clean for word in ["degree", "qualification", "education"]):
        return "For AI careers, a degree in Artificial Intelligence, Computer Science, Data Science, Software Engineering, or a related field is useful."

    elif any(word in user_question_clean for word in ["career", "future", "scope", "choose"]):
        return "AI is a strong career choice because it has high demand, good salary opportunities, and applications in many industries."

    elif any(word in user_question_clean for word in ["skill", "skills", "learn"]):
        return "Important AI skills include Python, Machine Learning, Deep Learning, NLP, Computer Vision, Mathematics, Statistics, and problem solving."

    elif any(word in user_question_clean for word in ["project", "projects", "portfolio"]):
        return "Good AI projects include chatbots, recommendation systems, image classification, sentiment analysis, disease prediction, and object detection."

    elif any(word in user_question_clean for word in ["internship", "job", "work"]):
        return "To get an AI internship or job, build projects, improve your resume, apply on LinkedIn and company websites, and prepare for interviews."

    elif any(word in user_question_clean for word in ["resume", "cv"]):
        return "Your AI resume should include skills, projects, internships, certifications, education, and achievements."

    elif any(word in user_question_clean for word in ["interview", "questions"]):
        return "For AI interviews, prepare Python, ML, DL, NLP, Computer Vision, evaluation metrics, and your own project explanation."

    else:
        return "I can answer questions about AI careers, degrees, skills, jobs, internships, salaries, projects, resumes, and interview preparation."
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json["message"]
    bot_response = get_answer(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)