from flask import Flask, redirect, url_for, request, render_template
import hatespeech as f
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
	return 'welcome %s' % name

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':

		text = request.form['txtHateSpeech']
		if (request.files):
			audio = request.files['audioHateSpeech']
			audio.save(audio.filename)
		else:
			audio = None
		if (text != '' or audio != None):
			if (text != ''):
				if (audio != None):
					# do both
					return redirect(url_for('success', name = text + " " + audio.filename))
				else:
					# do text only
					y_pred = f.text(text, cbow_file="cbow_200", model_file="WE_adam")
					if (y_pred[0] == 0):
						y_pred = "No Hate Speech"
					else:
						y_pred = "Hate Speech"
					return redirect(url_for('success', name = y_pred))

			else:
				# do audio only

				akustik()
				return redirect(url_for('success', name = audio.filename))

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