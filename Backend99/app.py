rom flask import Flask, request, jsonify, render_template
from model import hybrid_recommend

app = Flask(_name_)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id', 1)
    movie_title = data.get('movie_title', '')
    review_text = data.get('review_text', '')
    recommendations = hybrid_recommend(user_id, movie_title, review_text)
    return jsonify(recommendations)

if _name_ == '_main_':
    app.run(debug=True)