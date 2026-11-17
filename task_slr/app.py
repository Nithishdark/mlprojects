import pickle

from flask import Flask,render_template,request
import numpy as np

app=Flask(__name__)
with open("task_slr.pkl","rb") as file:
 model=pickle.load(file)

@app.route("/")
def default_site():
    return render_template("Height to weight predictor.html")
@app.route("/predict",methods=["GET","POST"])
def prediction():
    value=[float(i )for i in request.form.values()]
    value=np.array([value])
    result=model.predict(value)[0]
    return render_template("Height to weight predictor.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
