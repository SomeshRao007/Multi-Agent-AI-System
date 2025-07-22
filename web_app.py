from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template_string('<h1>Welcome Home!</h1>')

if __name__ == '__main__':
    app.run(debug=True)