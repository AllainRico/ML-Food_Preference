from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(age, food, juice, dessert):
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		food = 1 if food == 'Western Food' else 0
		juice = 1 if juice == 'Carbonated drinks' else 0
		dessert = 1 if dessert == 'No' else 0
		output = model.predict_proba([[age, food, juice, dessert]])
		result_str = "Female " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "Male " + "{:.4f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	age = request.form.get('Age')
	food = request.form.get('Food')
	juice = request.form.get('Juice')
	dessert = request.form.get('Dessert')
	age = int(age)
	the_output = the_model(age, food, juice, dessert)

	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='1422', host='0.0.0.0', use_reloader=True)