from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

MAX_LAYERS = 4
MAX_PERCEPTRON_BY_LAYER = 4
RS = 42

path_to_training_file = "../datasets/data_4000v/env_vital_signals.txt"

victim = {}

samples = []
y_grav = []
y_classe = []

# Lendo os dados
with open(path_to_training_file, "r") as file:
    for line in file:
        data_line = str(line).strip('\n').split(',')
        id = int(data_line[0])
        pSist = float(data_line[1])
        pDiast = float(data_line[2])
        qPA = float(data_line[3])
        pulso = float(data_line[4])
        freq_resp = float(data_line[5])
        grav = float(data_line[6])
        classe = int(data_line[7])
        
        victim[id] = (id, pSist, pDiast, qPA, pulso, freq_resp, grav, classe)
        samples.append([qPA, pulso, freq_resp])
        y_grav.append(grav)
        y_classe.append(classe)

# Possibilidades de Perceptron
configs = []
for n_camadas in range(1, MAX_LAYERS + 1):
    for result in product(range(1, MAX_PERCEPTRON_BY_LAYER + 1), repeat=n_camadas):
        configs.append(tuple(result))
            
activations = ['relu', 'identity', 'logistic', 'tanh']

# Dividir treino e teste
x_train, x_test, y_train, y_test = train_test_split(samples, y_grav, test_size=0.20, random_state=RS)

best_activation: str = ''
best_config = None
best_accurracy = 0.0
try:
    for config in configs:
        for activ in activations:
            network = MLPRegressor(hidden_layer_sizes=config, activation=activ, max_iter=10000, early_stopping=True)
            network.fit(x_train, y_train)
            y_pred = network.predict(x_test)
            # Calcule as métricas de regressão
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            if(best_accurracy < r2):
                best_accurracy = r2
                best_activation = activ
                best_config = config
except Exception as e:
    with open("result.txt", "w") as file:
        file.write(f"{best_accurracy}\n")
        file.write(f"{best_activation}\n")
        file.write(f"{best_config}\n")

print(f"ACCURRACY: {best_accurracy:10.2f}")
print(f"ACTIVATION: {best_activation:>11}")
print(f"CONFIG: {best_config}")

with open("result.txt", "w") as file:
    file.write(f"{best_accurracy}\n")
    file.write(f"{best_activation}\n")
    file.write(f"{best_config}\n")


