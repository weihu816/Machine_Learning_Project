function model = init_model()

load('model_em.mat');
load('model_logistic.mat');
load('index.mat');
model.model1 = model_em;
model.model2 = model_logistic;
model.index = index;
