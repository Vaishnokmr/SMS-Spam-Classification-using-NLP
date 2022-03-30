from django.shortcuts import render
import pickle
import nltk
nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps  = PorterStemmer()
from nltk.tokenize import word_tokenize
import sklearn



# Create your views here.
from django.http import HttpResponse, request
from django.shortcuts import render
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def home(request):
    return render (request,"index.html")


def predict(request):
    def transform_text(text):
        text = nltk.word_tokenize(text)
        
        y = []
        
        for i in text:
            y.append(i.lower())
        text = y[:]
        y.clear()


        for i in text:
            if i.isalnum():
                y.append(i)

        text =  y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text =  y[:]
        y.clear()
        
        for i in text:
        
            y.append(ps.stem(i)) 
            
        text = (" ".join(y))
        
        return text

    text = str(request.POST.get('name')) 


      
    transformed_sms = transform_text(text)
    
    vector_input = tfidf.transform([transformed_sms]).toarray()
    
    prediction = model.predict(vector_input)

    
    if prediction == 1:
        prediction = "Spam"
        

        return render (request, 'index.html',{ "prediction" : prediction})
    else:
        prediction = "Safe"
        return render (request, 'index.html',{ "prediction" : prediction})





    


    


