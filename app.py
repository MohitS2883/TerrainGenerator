from flask import *
from generate import gen

app = Flask(__name__)

@app.route('/gen', methods=['GET', 'POST'])
def terrain():
    if request.method == 'POST':
        path = gen()
        if not path:
            return render_template('error.html')
        return render_template('generator.html', path=path)
    
    return render_template('generator.html')

@app.route('/')
def homeredirect():
    return redirect('/gen')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)