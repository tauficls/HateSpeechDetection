from flask import Flask, redirect, url_for, request, render_template
import hatespeech as hs
import features_extraction_IS10_dataTest as FEConfig
import arffToCsv as converter
import os
app = Flask(__name__)

script_dir = os.path.dirname(__file__)

@app.route('/success/<name>')
def success(name):
	return 'welcome %s' % name

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':

		text = request.form['txtHateSpeech']
		if (request.files):
			audio = request.files['audioHateSpeech']
			audio.save(os.path.join(script_dir, "static/audio/" + audio.filename))
		else:
			audio = None
		if (text != '' or audio != None):
			if (text != ''):
				if (audio != None):
					# do both
					FEConfig.config(os.path.join(script_dir, "static/audio/"), audio.filename)
					files = [arff for arff in os.listdir('static/arff/') if arff.endswith(".arff")]
					file_akustik = []
					for file in files:
						with open('static/arff/' + file , "r") as inFile:
							content = inFile.readlines()
							name,ext = os.path.splitext(inFile.name)
							new = converter.toCsv(content)
						with open(name + ".csv", "w") as outFile:
							outFile.writelines(new)
							file_akustik.append(name+".csv")

					y_pred = hs.fuse(text, filename=file_akustik, model_fuse="is10_ES"
						, model_cbow="cbow_200")

					# if (y_pred[0] == 0):
					# 	y_pred = "No Hate Speech"
					# else:
					# 	y_pred = "Hate Speech"

					return redirect(url_for('success', name = y_pred))
				else:
					# do text only
					y_pred = hs.text(text, cbow_file="cbow_200", model_file="WE_adam")
					if (y_pred[0] == 0):
						y_pred = "No Hate Speech"
					else:
						y_pred = "Hate Speech"
					return redirect(url_for('success', name = y_pred))

			else:
				# do audio only

				# Main loop for reading and writing files
				FEConfig.config(os.path.join(script_dir, "static/audio/"), audio.filename)
				files = [arff for arff in os.listdir('static/arff/') if arff.endswith(".arff")]
				file_akustik = []
				for file in files:
					with open('static/arff/' + file , "r") as inFile:
						content = inFile.readlines()
						name,ext = os.path.splitext(inFile.name)
						new = converter.toCsv(content)
					with open(name + ".csv", "w") as outFile:
						outFile.writelines(new)
						file_akustik.append(name+".csv")
				y_pred = hs.akustik(file_akustik, "IS10_2")
				return redirect(url_for('success', name = y_pred))

		else:
			return redirect(url_for('success', name = 'fail'))

# @app.route('/main/<name>')
# def hatespeech(name):
#    return 'Hello %s!' % name
# # app.add_url_rule('/', 'main', hatespeech)
# @app.route('/')
# def index():
#    return render_template('login.html')

@app.route('/hello/<user>')
def hello_name(user):
	return render_template('hello.html', name = user)

@app.route('/')
def hatespeech():
	return render_template('main.html')

if __name__ == '__main__':
   app.run(debug=True)