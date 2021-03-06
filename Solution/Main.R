#Yeast classfier using constant learning rate of 0.1,activation function = softmax, hidden_layer = RELU. Hidden layer and units are changed according to the question
source("own_functions.R")
library(pracma)
library(tensorflow)
library(keras)

yeast <- read.table("~/Desktop/ML/ECS 171/Assignment 2/yeast.data", quote="\"", comment.char="")
train_size = 964

#Compute training and testing values

values = Get_values(yeast,train_size)
X_train = values[[1]]
Y_train = as.matrix(values[[2]])
X_test = values[[3]]
Y_test = values[[4]]

Keras_x = rbind(X_train,X_test)
Keras_x = Keras_x[,-1]
Keras_y = rbind(Y_train,Y_test)

#Assign Theta values 
Theta1 = rand(3,9)
Theta2 = rand(10,4)

#using gradient descent 

#Gradient_descent_val = gradient_descent_2layer(X_train,Y_train,Theta1,Theta2,6000,X_test,Y_test,0.01)

plot(Gradient_descent_val[[3]], type = "o")
plot(Gradient_descent_val[[4]], type = "o")

#using keras

model <- keras_model_sequential() 
model %>% 
  # hidden layer1
  layer_dense(units = 12, use_bias = TRUE, activation = 'relu', input_shape = c(8) )%>% 
  #layer_dropout(rate = 0 .01) %>% 
  # hidden layer2
  layer_dense(units = 12, use_bias = TRUE, activation = 'relu') %>%
  #layer_dropout(rate = 0.01) %>%
  #hidden layer3
  layer_dense(units = 12, use_bias = TRUE, activation = 'relu') %>%
  #outputlayer
  layer_dense(units = 10, activation = 'softmax')

sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False,clipnorm=1, clipvalue=0.5 )


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = 'accuracy'
)
 summary(model)

 

y_binary = to_categorical(Keras_y)
y_binary = y_binary[,-1]
history <- model %>% fit(
  Keras_x, 
  y_binary,
  callback = list(callback_model_checkpoint("5layer12unitssofrmax",monitor='val_acc', verbose=0, save_best_only= TRUE, save_weights_only=TRUE, mode='auto', period=1)),
  batch_size = 1,
  epochs = 300, 
  
  validation_split = 0.35
)

 