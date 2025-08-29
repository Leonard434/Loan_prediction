    #from flask import Flask, render_template, request
    #import flask
    #import numpy as np
    #import pickle


    #app = Flask(__name__)

    #model = pickle.load(open("best_model.pkl", "rb"))
    #@app.route('/')
    #def Home():
     #   return render_template('index.html')

    #@app.route('/predict', methods = ['POST'])
    #def predict():
     #   float_features = [float[x] for x in request.form.values()]
      #  features = np.array([float_features])
       # prediction = model.predict(features)[0]
    #
     #   return render_template("index.html", prediction_text=prediction)
    #
    #if __name__ == '__main__':
     #   app.run(debug=True)

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("best_model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        form_values = list(request.form.values())
        # Convert all inputs to float
        float_features = [float(x) for x in form_values]

        # Make it a 2D numpy array
        features = np.array([float_features])

        prediction = model.predict(features)[0]

        # Interpret result
        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)




