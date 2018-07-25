#back propogation algorithm 

back_propogate_2layer<-function(x,l1,l2,y,Theta1,Theta2){
  y_matrix = matrix(0,nrow(l2),1)
  y_matrix[y,1] = 1
  
  delta3 = l2 - y_matrix 
  delta2 = t(Theta2) %*% delta3 *(l1)* (1-l1)
  Theta2_grad = delta3 %*% t(l1)
  x = as.matrix(x)
  Theta1_grad = delta2[-1,] %*% x
  
  return(list(Theta1_grad,Theta2_grad))
}


#compute accuracy given a test and training set 

compute_accuracy<-function(X,Y,Theta1,Theta2){
  correct = 0 
  wrong = 0
  for (i in 1:nrow(X)){
    xval = t(as.matrix(X[i,]))
    yval = as.matrix(Y[i,1])
    layers = forward_propogate_2layer(xval,Theta1,Theta2)
    hypothesis = as.matrix(layers[[2]])
    #find returned value
    val = which(hypothesis == max(hypothesis))
    
    if (val == yval){
      correct = correct +1
    }
    if (val != yval){
      wrong = wrong +1
    }
  }
  accuracy = matrix(0,1,1)
  percent = (correct)/(correct+wrong)
  accuracy = rbind(accuracy,percent)
  accuracy = accuracy[-1,]
  return(accuracy)
}

# convert labels from strings to integers for classification 

convert_label<-function(Y){
  m = nrow(Y)
  #declare matrix to return
  n = matrix(0,m,1)
  #iterate over all input
  for ( i in 1:m)
  {
    # Convert Cases
    if(Y[i,] == "CYT"){ n[i,] = 1}
    if(Y[i,] == "ERL"){ n[i,] = 2}
    if(Y[i,] == "EXC"){ n[i,] = 3}
    if(Y[i,] == "ME1"){ n[i,] = 4}
    if(Y[i,] == "ME2"){ n[i,] = 5}
    if(Y[i,] == "ME3"){ n[i,] = 6}
    if(Y[i,] == "MIT"){ n[i,] = 7}
    if(Y[i,] == "NUC"){ n[i,] = 8}
    if(Y[i,] == "POX"){ n[i,] = 9}
    if(Y[i,] == "VAC"){ n[i,] = 10}
    
    
  }
  return(n)
}


#forward propogate to get output and layer_values for 2 layer network

forward_propogate_2layer<-function(X,Theta1,Theta2){
  bias = 1
  #feed forward 
  l1 = sigmoid( Theta1 %*% t(X))
  l1 = rbind(1,l1)
  l2 = sigmoid(Theta2 %*% l1)
  
  return(list(l1,l2))
  
  
}





#stochastic gradient descent for 2 layer feed forward neural network with back propogation 

gradient_descent_2layer<- function(X,Y,Theta1,Theta2,Max_iter,X_test,y_test) {
  m = nrow(X)
  accuracy = matrix(0,1,1)
  test_accuracy = matrix(0,1,1)
  test_set = cbind(X,Y)
  X_test = as.matrix(X_test)
  y_test = as.matrix(y_test)
  
  for ( j in 1:Max_iter){
    #random shuffling
    sample_set = test_set[sample(nrow(test_set), nrow(X)), ]
    X = sample_set[,-(ncol(sample_set))]
    Y = as.matrix(sample_set[,(ncol(sample_set))])
    m1 = nrow(X)
    #to test normal gradient descent
    Acc_Theta1_gradient = Theta1 * 0
    Acc_Theta2_gradient = Theta2 * 0
    
    
    #accumulate gradient/ use stochastic gradient descent depending upon wheter you use Acc_theta1_gradient or Theta1_gradient 
    for( i in 1:m1){
      #initialize Xval and Yval
      Xval = t(as.matrix(X[i,]))
      Yval = as.matrix(Y[i,])
      
      
      #get layers value
      list_feed_forward = forward_propogate_2layer(Xval,Theta1,Theta2)
      l1 = as.matrix(list_feed_forward[[1]])
      l2 = as.matrix(list_feed_forward[[2]])
      
      #compute gradient by backpropogation 
      list_gradient = back_propogate_2layer(Xval,l1,l2,Yval,Theta1,Theta2)
      Theta1_gradient <- (1)* list_gradient[[1]]
      Theta2_gradient <- (1)* list_gradient[[2]]
      
      
      
      Theta1 <- Theta1 - (0.01)*Theta1_gradient 
      Theta2 <- Theta2 - (0.01)*Theta2_gradient 
      
      
      
    }
    
    
    if(j %% 50 == 0 ){
    A = compute_accuracy(X,Y,Theta1,Theta2)
    accuracy <- rbind(accuracy,A)
    B = compute_accuracy(X_test,y_test,Theta1,Theta2)
    test_accuracy <- rbind(test_accuracy,B)
    }
    print(j)
  }
  return(list(Theta1,Theta2,accuracy,test_accuracy))
  
}


#Compute test and train values 

Get_values <- function(yeast,train_size){
  X_val = yeast[, - ncol(yeast)]
  X_val[,1] = 1
  y_val = as.matrix(yeast[,ncol(yeast)])
  y_val = convert_label(y_val)
  X_train = as.matrix(X_val[1:train_size,])
  Y_train = as.matrix(y_val[1:train_size,])
  
  X_test = as.matrix(X_val[-(1:train_size),])
  Y_test = as.matrix(y_val[-(1:train_size),])
  
  return(list(X_train,Y_train,X_test,Y_test))
  
}


