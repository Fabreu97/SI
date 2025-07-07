# Bibliotecas para Rede Neural
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import joblib

# Bibliotecas para Árvore de Decisão
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV

MAX_LAYERS = 4
MAX_PERCEPTRON_BY_LAYER = 5
RS = 42

path_to_training_file = "../datasets/data_4000v/env_vital_signals.txt"
path_to_casualty_severity_prediction_file = "../datasets/data_800v/env_vital_signals.txt"

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
solvers = ['adam']

# Dividir treino e teste
x_train, x_test, y_train, y_test = train_test_split(samples, y_grav, test_size=0.20, random_state=RS)

"""
# Descobrir uma configuração boa para RS = 42
best_activation: str = ''
best_config = None
best_accurracy = 0.0
best_solver = 'adam'
best_model = None

try:
    for config in configs:
        for activ in activations:
            network = MLPRegressor(hidden_layer_sizes=config, activation=activ, max_iter=10000, early_stopping=False, random_state=RS, solver='adam', momentum=0.9)
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
                best_model = network
except Exception as e:
    with open("result.txt", "w") as file:
        file.write(f"{best_accurracy}\n")
        file.write(f"{best_activation}\n")
        file.write(f"{best_config}\n")
        file.write(f"{best_solver}\n")

print(f"ACCURRACY: {best_accurracy:10.2f}")
print(f"ACTIVATION: {best_activation:>11}")
print(f"CONFIG: {best_config}")
print(f"SOLVER: {best_solver}")

with open("result.txt", "w") as file:
    file.write(f"{best_accurracy}\n")
    file.write(f"{best_activation}\n")
    file.write(f"{best_config}\n")
    file.write(f"{best_solver}\n")

# Salva o objeto inteiro em um único arquivo
joblib.dump(best_model, 'modelo_completo_treinado.joblib')

print("Modelo treinado salvo em 'modelo_completo_treinado.joblib'")
"""

"""
# resultado
best_activation: str = 'tanh'
best_config = (4,5,3)
best_accurracy = 0.0
best_solver = 'adam'
best_alpha = None
best_learning_rate_init = None
best_model = None
# Fixei o momentum = 0.9, RS = 42 e MAX_ITER = 10000 
activation_function = 'tanh'
layers = (4,5,3)
solver = 'adam'
learning_rate_init_v = (0.1, 0.01, 0.001)
alphas = (1e-5, 1e-4, 0.0001, 0.01, 0.1, 1)

# Descobrir a nova configuração
for learning in learning_rate_init_v:
    for a in alphas:
        network = MLPRegressor(hidden_layer_sizes=layers, activation=activation_function, solver=solver, random_state=RS, momentum=0.9, max_iter=10000, alpha=a, learning_rate_init=learning)
        network.fit(x_train, y_train)
        y_pred = network.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        if(r2 > best_accurracy):
            print(f"Find model with accurracy: {r2}")
            best_accurracy = r2
            best_alpha = a
            best_learning_rate_init = learning
            best_model = network

with open("result2.txt", "w") as file:
    file.write(f"BEST_ACCURRACY: {best_accurracy}\n")
    file.write(f"BEST_LAYERS_CONFIG: {best_config}\n")
    file.write(f"BEST_ACTIVATION_FUNCTION: {best_activation}\n")
    file.write(f"BEST_SOLVER: {best_solver}\n")
    file.write(f"BEST_ALPHA: {best_alpha}\n")
    file.write(f"BEST_LEARNING_RATE_INIT: {best_learning_rate_init}\n")

# Salva o objeto inteiro em um único arquivo
joblib.dump(best_model, 'modelo_completo_treinado2.joblib')
"""

network: MLPRegressor = joblib.load("./modelo_completo_treinado.joblib")
print("Modelo carregado . . .")

x_800v = []
y_800v_grav = []

# Irei testar com dados do 800v na predição do modelo
with open(path_to_casualty_severity_prediction_file, "r") as file:
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

        x_800v.append([qPA, pulso, freq_resp])
        y_800v_grav.append(grav)
"""
print("Dados das 800 vítimas lidos . . .")

y_pred = network.predict(x_800v)
print(f"Accurracy: {r2_score(y_800v_grav, y_pred)}")

network.fit(x_800v, y_800v_grav)
y_pred = network.predict(x_test)
print(f"Accurracy: {r2_score(y_test, y_pred)}")
"""

# Irei criar duas redes neurais regressor para colocar na tabela
# Rede Neural 2:
"""
solver = 'sgd'
momentum = 0.9
learn_init = 0.0005
max_iter = 10000
activations = ['relu', 'tanh']
layers = (30,20)
alpha = 0.001

best_actv = None
best_r2 = 0.0
for actv in activations:
    network2 = MLPRegressor(hidden_layer_sizes=layers, solver=solver, momentum=momentum, learning_rate_init=learn_init, max_iter=10000, random_state=RS, activation=actv, alpha=alpha)
    network2.fit(x_train, y_train)
    y_pred = network2.predict(x_test)
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    if r2 > best_r2:
        best_r2 = r2
        print(f"R²: {r2}")
        best_actv = actv

with open("config2.txt", "w") as file:
    file.write(f"BEST_ACCURRACY: {best_r2}\n")
    file.write(f"BEST_LAYERS_CONFIG: {layers}\n")
    file.write(f"BEST_ACTIVATION_FUNCTION: {best_actv}\n")
    file.write(f"BEST_SOLVER: {solver}\n")
    file.write(f"BEST_ALPHA: {alpha}\n")
    file.write(f"BEST_LEARNING_RATE_INIT: {learn_init}\n")
    file.write(f"RANDON STATE: 42\n")


# Rede Neural 3:

solver = 'lbfgs'
momentum = 0.9
learn_init = 0.001
max_iter = 10000
activations = ['relu', 'identity', 'logistic', 'tanh']
layers = (10,5,10)
alpha = 0.001

best_actv = None
best_r2 = 0.0
for actv in activations:
    network2 = MLPRegressor(hidden_layer_sizes=layers, solver=solver, momentum=momentum, learning_rate_init=learn_init, max_iter=20000, random_state=RS, activation=actv, alpha=alpha)
    network2.fit(x_train, y_train)
    y_pred = network2.predict(x_test)
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    if r2 > best_r2:
        best_r2 = r2
        print(f"R²: {r2}")
        best_actv = actv

with open("config3.txt", "w") as file:
    file.write(f"BEST_ACCURRACY: {best_r2}\n")
    file.write(f"BEST_LAYERS_CONFIG: {layers}\n")
    file.write(f"BEST_ACTIVATION_FUNCTION: {best_actv}\n")
    file.write(f"BEST_SOLVER: {solver}\n")
    file.write(f"BEST_ALPHA: {alpha}\n")
    file.write(f"BEST_LEARNING_RATE_INIT: {learn_init}\n")
    file.write(f"RANDON STATE: 42\n")


"""
# Comentarios sobre a rede neural
#####################################################################################
# Código inutil, porque mesmo utilizando a mesma configuração dos hiperparâmetros,  #
# a natureza estocástica(aleatória) do treinamento da rede neural.                  #
# Seus "pesos" e "biases" (os parâmetros internos que ela aprende) são iniciados    #
# com valores numéricos pequenos e aleatórios.                                      #
#####################################################################################
# Parada Antecipada                                                                 #
#####################################################################################
# Fixa a "semente aleatória"(random seed) não apenas na divisão dos dados, mas      #
# também no próprio modelo.                                                         #
#####################################################################################
# Alpha é o parametro de penalidade do overfitting(sobreajuste), se for muito baixo #
# é ruim                                                                            #
#####################################################################################

## CART ##
# Cuidado com overfitting se árvore for muito profunda
# Pruning: poda da árvore para evitar overfitting

# Parameters' definition
parameters = {
    'criterion': ['squared_error'],
    'max_depth': [2, 3,4, 5, 6],
    'min_samples_leaf': [2, 10]
}

regressor = DecisionTreeRegressor(random_state=42)

# grid search using cross-validation
# cv = 3 is the number of folds
# scoring = 'f' the metric for chosing the best model
clf = GridSearchCV(regressor, parameters, cv=3, scoring='r2', verbose=4)
clf.fit(x_train, y_train)
best_tree = clf.best_estimator_
best_param = clf.best_params_
joblib.dump(best_tree, 'modelo_arvore_decisao.joblib')
with open("config_tree.txt", "w") as file:
    for key, value in best_param.items():
        file.write(f"{key}: {value}\n")

y_pred = best_tree.predict(x_test)
print(f"R²: {r2_score(y_pred, y_test): 10.2f}")

plt.figure(figsize=(16, 10))
tree.plot_tree(best_tree, filled=True)
plt.title("Árvore de Regressão - CART")
plt.show()