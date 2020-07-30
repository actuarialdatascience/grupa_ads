library(keras)
library(dplyr)
library(reshape2)

## setting working directories 
## wd - with other scripts
## data_folder - with Mx_1x1
wd <- "/home/as/Pulpit/ADS/Analizy/CNN LC/LC CNN/"
data_folder <- "/home/as/Pulpit/ADS/Analizy/CNN LC/LC CNN/death_rates/Mx_1x1"

setwd(wd)

ObsYear = 1999 # last year of training set
T0 <- 10 # number of years back used to forecast
model_type = "CNN"
source("0_dataReading.R")

# number of countries
N <- HMD_final %>% select(Country) %>% distinct() %>% nrow()

#model specifications
source("0_b_CNN_model_specification.R")
source("0_c_LSTM_model_specification.R")


# data scaling (MinMaxScaling on whole dataset) of logmx
val.min <- HMD_final %>% summarize(min(logmx)) %>% unlist()
val.max <- HMD_final %>% summarize(max(logmx)) %>% unlist()
HMD_final <- HMD_final %>% mutate(val = (logmx - val.min)/(val.max-val.min))

## transforming HMD data to NN input format
data.preprocessing.CNNs <- function(data.raw, gender, country, T0, ObsYear=1999){    
  mort_rates <- data.raw %>% filter(Gender == gender, Country == country) %>% select(Year, Age, val)
  mort_rates <- dcast(mort_rates, Year ~ Age, value.var="val")
  train.rates <- mort_rates %>% filter(Year <= ObsYear) %>% select(-Year) %>% as.matrix()
  n.train <- nrow(train.rates)-(T0-1)-1 # number of training samples
  xt.train <- array(NA, c(n.train, T0, 100))
  YT.train <- array(NA, c(n.train, 100))
  for (t0 in (1:n.train)){
    xt.train[t0,,] <- train.rates[t0:(t0+T0-1), ]
    YT.train[t0,] <-   train.rates[t0+T0,]
  }
  list(xt.train, YT.train)
}
    
## creating the training set (all observation up to 1999) based on mx values only (with data.preprocessing.CNNs function)
## Each yt observation is equal to a whole mortality curve and xt was equal to ten previous mortality curves (matrix 10:100). 
## It was done only when all ten previous curves where available for given country.

Genders <- c("Male","Female")
Countries <- HMD_final %>% select(Country) %>% distinct() %>% unlist()

VecGenders <- vector()
VecCountries <- vector()
ListData <- list()

obs <-0
for(c in 1:length(Countries)){
  for(g in 1:2){
    data <- data.preprocessing.CNNs(HMD_final,Genders[g],Countries[c],T0, ObsYear)
    n <- dim(data[[1]])[1]
    obs <- obs + n
    ListData[[(c-1)*2 + g]] <- data
    # VecGenders (with 0 or 1 for each observation)
    VecGenders<- c(VecGenders,rep(g-1,n))
    # VecCounties (with number from 0 to 37 corresponding to each country for each observation)
    VecCountries <- c(VecCountries,rep(c-1,n))
  }
}
    
## Binding observations form different countries into one dataset
## transformation of xtrain from data.preprocessing.CNNs to list of previous xtrain, veccountries and vecgender

x.train <- array(NA, dim=c(obs, dim(ListData[[1]][[1]])[c(2,3)]))
y.train <- array(NA, dim=c(obs,dim(ListData[[1]][[2]])[2]))

counter = 0
for (i in 1:(g*c)){
  n <- dim(ListData[[i]][[1]])[1]
  for(j in 1:n){
    x.train[counter+j,,] <- ListData[[i]][[1]][j,,]
    y.train[counter+j,] <- ListData[[i]][[2]][j,]
  }
  counter <- counter + n
}

x.train <- list(x.train, VecCountries, VecGenders)
    
# model

if(model_type == "CNN"){
  model <- CNN(N, T0)
} else if(model_type == "LSTM"){
  model <- LSTM(N,T0)  
} else
{
  stop("Wrong arcitecture specified within model_type variable")
}
      
modelName = paste(model_type ,T0, sep ="_")
fileName <- paste("./CallBack/best_model_", modelName, sep="")
summary(model)

# define callback
CBs <- callback_model_checkpoint(fileName, monitor = "val_loss", verbose = 1,  save_best_only = TRUE, save_weights_only = TRUE)

# gradient descent fitting: takes roughly 800 seconds on my laptop
  t1 <- proc.time()
      fit <- model %>% fit(x=x.train, y=y.train, validation_split=0.05,
                           epochs=500, verbose=1, callbacks=CBs) #in paper 2000 there is no difference at this moment                                       
      proc.time()-t1

# in-sample error              
load_model_weights_hdf5(model, fileName)
    
mort_y = exp(y.train*(val.max-val.min) + val.min)
pred_y = exp(((model %>% predict(x.train))*(val.max-val.min)+ val.min))
pred_y = pred_y[,1,]
round(10^4*mean((pred_y - mort_y)^2),4)
            
## recursive prediction
    
# validation data pre-processing
all_mort2 <- HMD_final[which(HMD_final$Year > (ObsYear-10)),]
all_mortV <- all_mort2
vali.Y <- all_mortV[which(all_mortV$Year > ObsYear),]

recursive.prediction <- function(ObsYear, all_mort2, gender, country_name, country_index, T0, val.min, val.max, model.p){       
  single.years <- array(NA, c(2016-ObsYear))
  
  for (ObsYear1 in ((ObsYear+1):2016)){
    data1 <- all_mort2 %>% filter(Year >= (ObsYear1-10))
    data2 <- data.preprocessing.CNNs(data1, gender, country_name,T0, ObsYear1)
    # MinMaxScaler (with minimum and maximum from above)
    x.vali <- data2[[1]]
    if (gender=="Female"){yy <- 1}else{yy <- 0}
    x.vali <- list(x.vali, rep(country_index, dim(x.vali)[1]), rep(yy, dim(x.vali)[1]))
    y.vali <- as.vector((data2[[2]])*(val.max-val.min)+ val.min)
    predicted_logmx <- ((model %>% predict(x.vali))*(val.max-val.min)+ val.min) %>% as.vector()
    Yhat.vali2 <- exp(predicted_logmx)
    single.years[ObsYear1-ObsYear] <- round(10^4*mean((Yhat.vali2-exp(y.vali))^2),4)
    predicted <- all_mort2 %>% filter(Year==ObsYear1, Gender == gender, Country == country) # [which(all_mort2$Year==ObsYear1),]
    keep <- all_mort2 %>% filter(Year!=ObsYear1, Gender == gender, Country == country)
    predicted$logmx <- predicted_logmx
    predicted$mx <- exp(predicted$logmx)
    all_mort2 <- rbind(keep,predicted)
    #all_mort2 <- all_mort2 %>% order_by(Year, Age)
  
  list(all_mort2, single.years)
  }                  
}

    
#example prediction      
pred.CHE.F <- recursive.prediction(1999, all_mort2, "Female", "CHE",5,T0, val.min, val.max, model)
pred.CHE.F <- pred.CHE.F[[1]] %>% filter(Year > ObsYear)
vali.Y.CHE.F <- vali.Y %>% filter(Country == "CHE", Gender == "Female")
round(10^4*mean((pred.CHE.F$mx-vali.Y.CHE.F$mx)^2),4)

  
  