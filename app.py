from flask import Flask, render_template, jsonify , request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData


app = Flask(__name__)



@app.route('/', methods = ["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else: 
        data = CustomData(
            carat = float(request.form.get("carat")), 
            depth = float(request.form.get("depth")),
            table = float(request.form.get("table")), 
            x = float(request.form.get("x")),
            y = float(request.form.get("y")),
            z = float(request.form.get("z")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity"),     
        )
        df = data.getDataAsDataFrame()
        predictions = PredictionPipeline().predict(df)
        preds = round(predictions[0], 2)
        return render_template("result.html", final_result=preds)


if __name__ == "__main__":
    app.run(debug=True)