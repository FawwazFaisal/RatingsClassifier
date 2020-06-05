from datetime import datetime
import datetime as dt
import numpy as np
import flask
import pickle


app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = pickle.load(open("classifier.pkl", 'rb'))

def predict(diff):
	diff = np.float(diff)
	rating = model.predict([[diff]])
	result = str(rating[0])
	return result

@app.route('/prediction', methods=['GET','POST'])
def prediction():
	diff = flask.request.args.get("diff")
	res = predict(diff)
	return res

if __name__ == "__main__":
    app.run(use_reloader=False, port = 5005)