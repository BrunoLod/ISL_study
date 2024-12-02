import pickle

from flask import Flask, jsonify, request

artefact = pickle.load(open("endpoints/regressao_linear_1.pkl.sav", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "ML APP in Production..."

@app.route("/cotacao/", methods=["POST"])
def cotacao() -> str:

      """
      Função provisiona o modelo de regressão linear, 
      para o endpoint de cotação. Nela há uma camada de 
      processamento dos dados de entrada para array numpy
      e a execução da predição do modelo de regressão linear
      utilizado. 
      """

      request_data = request.get_json()
      columns = ["crim", "zn", 
                 "chas", "nox", 
                 "rm", "dis", 
                 "rad", "tax", 
                 "ptratio", "lstat"]
      
      # Extrair dados e preparar o formato de entrada
      input_data = ([request_data[column] for column in columns])
       
      price = artefact.predict(input_data)

      return jsonify(price=float(price[0]))

# Passando dentro do método o parâmetro informado, 
# para assim que atualize a API a alteração seja
# refletida no front. 
app.run(debug=True)
