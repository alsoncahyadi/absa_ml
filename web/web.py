from flask import Flask, render_template, request, redirect, flash, Markup
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sys
import os

load_dotenv()
sys.path.insert(0, '..')

import utils
import main
from constants import Const

UPLOAD_FOLDER = Const.ROOT + "reviews/upload"
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_bolded_tokens(data):
    bolded_tokens = []
    for i, sent in enumerate(data[1]):
        new_sentence = []
        tokens = data[0][i].split()
        if len(sent) != len(tokens):
            print(i, len(sent), len(tokens))
            print(" ".join(tokens))
        for j, tag in enumerate(sent):
            try:
                token = tokens[j]
            except:
                pass
            if tag[:-2] == "ASPECT":
                token = "<b class='aspect'>" + token + "</b>"
            new_sentence.append(token)
        bolded_tokens.append(" ".join(new_sentence))
    return bolded_tokens

def get_processed_data(data):
    new_data = {}

    # BAR CHART
    new_data['bar_chart'] = {}
    new_data['bar_chart']['positive'] = []
    new_data['bar_chart']['negative'] = []

    for sentiment in Const.SENTIMENTS:
        for category in Const.CATEGORIES:
            tuples = data[4]['tuples']
            count = len(tuples[category][sentiment])
            new_data['bar_chart'][sentiment].append(count)

    # PIE CHART
    new_data['pie_chart'] = []
    for i in range(4):
        count = data[2][:,i].sum()
        new_data['pie_chart'].append(count)

    # STARS
    new_data['stars'] = dict([(category, [0, 0, 0]) for category in Const.CATEGORIES])
    for category in new_data['stars']:
        rating = data[5][category][1]
        for i in range(5):
            """
                index:
                0 ==> empty
                1 ==> half star
                2 ==> full star
            """
            if rating >= 1.:
                new_data['stars'][category][2] += 1
            elif rating >= 0.5:
                new_data['stars'][category][1] += 1
            else:
                new_data['stars'][category][0] += 1
            rating -= 1

    # MARKUP SENTENCE
    new_data['markup_sents'] = [Markup(sent) for sent in get_bolded_tokens(data)]

    # COLORED CATEGORIES & SENTIMENT COLORS
    colored_categories = []
    sentiment_colors = []

    for i, category_mats in enumerate(data[2]):
        sent_categories = []
        sentiment_color = []
        for j, category in enumerate(Const.CATEGORIES):
            if category_mats[j] == 1.: # if category exists in sentence
                if data[3][j][i] == 1.: # if sentiment is positive
                    sent_categories.append("<b class='neg'>" + category.title() + "</b>")
                    sentiment_color.append(Markup('#28a745'))
                else:
                    sent_categories.append("<b class='pos'>" + category.title() + "</b>")
                    sentiment_color.append(Markup('#dc3545'))
            else:
                sentiment_color.append(Markup('rgba(0,0,0,0.1)'))
        colored_categories.append(", ".join(sent_categories))
        sentiment_colors.append(sentiment_color)

    new_data['colored_categories'] = [Markup(colored_category) for colored_category in colored_categories]
    new_data['sentiment_colors'] = sentiment_colors

    # PRETTY_TUPLES
    pretty_tuples = {}
    for category in data[4]['tuples_unique']:
        pretty_tuples[category] = {'positive':[], 'negative':[]}
        for k, v in data[4]['tuples_unique'][category].items():
            pretty_tuples[category][k] = [Markup("<span class='aspect-details-{}'>".format(k) + aspect + "</span>") for aspect in v]
    new_data['pretty_tuples'] = pretty_tuples

    # CONVERT TO JAVASCRIPT VARIABLES
    data_into_javascript = ['pie_chart', 'bar_chart']
    for k in data_into_javascript:
        new_data[k] = str(new_data[k]).replace("'", "")

    return new_data

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'GET':
        data = utils.load_object("data.pkl")
    elif request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']

        if file.filename == '':
            print('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            raw_reviews_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(raw_reviews_path)
            data = main.main(raw_reviews_path)
            utils.save_object(data, "data.pkl")

    new_data = get_processed_data(data)
    return render_template('results.html',
        bar_chart=new_data['bar_chart'],
        pie_chart=new_data['pie_chart'],
        ratings=data[5],
        stars=new_data['stars'],
        table=zip(new_data['markup_sents'], new_data['sentiment_colors']),
        categories=Const.CATEGORIES,
        tuples=new_data['pretty_tuples'],
    )

@app.route('/charts', methods=['GET'])
def charts():
    return render_template('charts.html')

if __name__ == '__main__':
    app.config['SESSION_TYPE'] = 'filesystem'
    app.secret_key = 'super secret key'

    app.debug = True
    app.run(host='0.0.0.0')
