# digit recognizer using keras package in R
# by: Mohsin Uddin

# setting working directory
setwd("E:\\Mohsin important\\data science\\Machine learning\\digit recognizer\\pics") # set the directory if needed, it is where my pictures of handwritten digits are

# installing keras for windows
library(reticulate) # this is must before you install keras on windows, otherwise won't work
library(keras)
install_keras(method = "conda", envname = "r-reticulate") # you have to write the command in this manner for windows, otherwise won't work

# libraries
library(tidyverse)
library(EBImage)

# getting mnist data
mnist <- dataset_mnist()
trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

# plotting first four digits as images
par(mfrow = c(2,2))
for (i in 1:4) plot(as.raster(trainx[i,,], max = 255))
par(mfrow= c(1,1))

# reshaping & rescaling of trainx and testx data
trainx <- array_reshape(trainx, c(nrow(trainx), 784))
testx <- array_reshape(testx, c(nrow(testx), 784))
trainx <- trainx / 255
testx <- testx /255

# one hot encoding for trainy and testy data
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

# building the model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units= 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax')

# compiling the model
model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = 'accuracy')

# fitting the model
history <- model %>% 
  fit(trainx,
      trainy,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)

# evaluation and prediction of test data using the model
model %>% evaluate(testx, testy)
pred <- model %>% predict_classes(testx)
table(Predicted = pred, Actual = mnist$test$y)

prob <- model %>% predict_proba(testx)
cbind(prob, Predicted_class = pred, Actual = mnist$test$y)[1:5,]

# getting new image (handwriting) data
temp = list.files(pattern = "*.jpg")
mypic <- list()
for (i in 1:length(temp)) {mypic[[i]] <- readImage(temp[[i]])}

# plotting the images of digits
par(mfrow = c(4,2))
for (i in 1:length(temp)) plot(mypic[[i]])
par(mfrow = c(1,1))

# resizing and reshaping image data
for (i in 1:length(temp)) {colorMode(mypic[[i]]) <- Grayscale}
for (i in 1:length(temp)) {mypic[[i]] <- 1-mypic[[i]]}
for (i in 1:length(temp)) {mypic[[i]] <- resize(mypic[[i]], 28, 28)}
for (i in 1:length(temp)) {mypic[[i]] <- array_reshape(mypic[[i]], c(28,28,3))}

# preparing dataset
new <- NULL
for (i in 1:length(temp)) {new <- rbind(new, mypic[[i]])}
newx <- new[,1:784]
newy <- c(0, 1, 2, 3, 4, 5, 6, 7)

# prediction
pred <- model %>% predict_classes(newx)
table(Predicted = pred, Actual = newy)

prob <- model %>% predict_proba(newx)
cbind(prob, Predicted = pred, Actual = newy)
