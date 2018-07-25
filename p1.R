
source("own_functions.R")


yeast <- read.table("~/Desktop/ML/ECS 171/Assignment 2/yeast.data", quote="\"", comment.char="")
train_size = 964

#Compute training and testing values

values = Get_values(yeast,train_size)
X_train = values[[1]]
Y_train = as.matrix(values[[2]])
X_test = values[[3]]
Y_test = values[[4]]

library(pracma)

#Assign Theta values 
Theta1 = rand(3,9)
Theta2 = rand(10,4)

#using gradient descent 

Gradient_descent_val = gradient_descent_2layer(X_train,Y_train,Theta1,Theta2,7000,X_test,Y_test)

plot(Gradient_descent_val[[3]], type = "o")
plot(Gradient_descent_val[[4]], type = "o")
Theta1 = 