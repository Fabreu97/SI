from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
import joblib

MAX_LAYERS = 5
MAX_PERCEPTRON_BY_LAYER = 5

path_4000v  = "../datasets/data_4000v/env_vital_signals.txt"
path_800v   = "../datasets/data_800v/env_vital_signals.txt"

sample = []
y = []

# Lê os dados das vítimas
with open(path_4000v, "r") as file:
    for line in file:
        datas = line.strip('\n').split(',')
        id      = int(datas[0])
        pSist   = float(datas[1])
        pDiast  = float(datas[2])
        qPA     = float(datas[3])
        pulso   = float(datas[4])
        freq_resp = float(datas[5])
        grav    = float(datas[6])
        class_grav = int(datas[7])
        sample.append([qPA, pulso, freq_resp])
        y.append(grav)

configs = []
activate_functions = ['tanh']
solvers = ['adam']
max_iter = 15000
alp = 0.001

for n_camadas in range(1, MAX_LAYERS + 1):
    for result in product(range(1, MAX_PERCEPTRON_BY_LAYER + 1), repeat=n_camadas):
        configs.append(tuple(result))

best_config = None
best_activate = None
best_mse = float('inf')
best_r2 = 0.0
best_model = None

x_train, x_test, y_train, y_test = train_test_split(sample, y, test_size=0.2, random_state=42)

activ = activate_functions[0]
for config in configs:
    network = MLPRegressor(hidden_layer_sizes=config, activation=activ, alpha=alp, max_iter=max_iter)
    network.fit(x_train, y_train)
    y_pred = network.predict(x_test)
    mse = mean_squared_error(y_pred = y_pred, y_true=y_test)
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    if best_mse > mse:
        best_config = config
        best_activate = activ
        best_mse = mse
        best_r2 = r2
        best_model = network

joblib.dump(best_model, 'network/modelo_completo_treinado2.joblib')

with open("./network/config2.txt", "w") as file:
    file.write(f"Mean Squared Error: {best_mse}\n")
    file.write(f"R2: {best_r2}\n")
    file.write(f"BEST_HIDDEN_LAYER_SIZES: {best_config}\n")
    file.write(f"BEST_ACTIVATE_FUNCTION: {best_activate}\n")
    file.write(f"BEST_ALPHA: {alp}\n")
    file.write(f"BEST_ALPHA: {best_model.get_params()}\n")