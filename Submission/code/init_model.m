function model = init_model()

load('model_em.mat');
load('model_logistic.mat');
load('index.mat');
load('model_svm.mat');
load('X_train.mat');
model.model1 = model_em;
model.model2 = model_svm;
model.model3 = model_logistic;
model.X_train = X_train;
model.index = index;
