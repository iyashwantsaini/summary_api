import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from transformers import pipeline
from transformers import pipeline
import nltk
from flask_cors import CORS, cross_origin
nltk.download('punkt')
#from flask_ngrok import run_with_ngrok
summarizer = pipeline("summarization")

app = Flask(__name__)
#CORS(app)
#run_with_ngrok(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def summarization(document,number):
    total=document.split()
    list_of_paras=[]
    overall_summ=''
    sent = ''
    count = 0
    for sentence in nltk.sent_tokenize(document):
      l=sentence.split()
      count=count+len(l)
      if count < 500:
        sent=sent+sentence
      else:
        list_of_paras.append(sent)
        sent = ''
        count = 0
    if sent:
      list_of_paras.append(sent)
    #print(len(list_of_paras))
    for j in list_of_paras:
      ok=summarizer(j,min_length=int(number))
      summ=ok[0]['summary_text']
      overall_summ=overall_summ+summ
      
    return overall_summ
  
@app.route('/')
def index():
    return 'hello'
    
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def predict():
    content = request.json
    print(content)
    msg=content['text']
    number=content['number']
    m=msg.split()
    if len(m)<number:
        return jsonify('try again')
    summary=summarization(msg,number)
    print(summary)
    return jsonify(summary)



if __name__ == "__main__":
    app.run()
