function model = init_model()

load('model_em.mat');
load('model_logistic.mat');
load('index.mat');
load('model_svm.mat');
load('X_train_svm.mat');
load('index_svm.mat');
model.model1 = model_em;
model.model2 = model_svm;
model.model3 = model_logistic;
model.X_train_svm = X_train_svm;
model.index = index;
model.index_svm = index_svm;
