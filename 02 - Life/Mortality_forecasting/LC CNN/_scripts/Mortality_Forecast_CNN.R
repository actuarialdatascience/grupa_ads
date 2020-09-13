library(keras)
library(dplyr)
library(reshape2)

## setting working directories 
## wd - with other scripts
## data_folder - with Mx_1x1
wd <- "/home/as/Pulpit/ADS/Analizy/Own/CNN LC/LC CNN/"
data_folder <- "/home/as/Pulpit/ADS/Analizy/Own/CNN LC/LC CNN/death_rates/Mx_1x1"

setwd(wd)

ObsYear = 1999 # last year of training set
T0 <- 10 # number of years back used to forecast
model_type = "CNN"
source("0_dataReading.R")

#source("0_dataReading_Sal.R")

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
  YT_year <- mort_rates %>% filter(Year <= ObsYear)%>% select(Year) %>% unlist() %>% unname()
  YT_year <- tail(YT_year, -10) # omit first 10 years (used only as x)
  n.train <- nrow(train.rates)-(T0-1)-1 # number of training samples
  xt <- array(NA, c(n.train, T0, 100))
  YT <- array(NA, c(n.train, 100))
  for (t0 in (1:n.train)){
    xt[t0,,] <- train.rates[t0:(t0+T0-1), ]
    YT[t0,] <-   train.rates[t0+T0,]
  }
  list(xt, YT, YT_year)
}
    
## creating the training set (all observation up to 1999) based on mx values only (with data.preprocessing.CNNs function)
## Each yt observation is equal to a whole mortality curve and xt was equal to ten previous mortality curves (matrix 10:100). 
## It was done only when all ten previous curves where available for given country.

Genders <- c("Male","Female")
Countries <- HMD_final %>% select(Country) %>% distinct() %>% unlist()

VecGenders <- vector()
VecCountries <- vector()
#VecYears = vector()

ListData <- list()

#HMD_final = HMD_final[with(HMD_final, order(Gender, Country)), ]


obs <-0
for(c in 1:length(Countries)){
  for(g in 1:2){
    data <- data.preprocessing.CNNs(HMD_final,Genders[g],Countries[c],T0, ObsYear)
    n <- dim(data[[1]])[1]
    obs <- obs + n
    ListData[[(c-1)*2 + g]] <- data
    # VecGenders (with 0 or 1 for each observation)
    VecGenders<- c(VecGenders,rep(g-1,n))
    
    #years_fore = HMD_final[Gender == Genders[g] & Country ==Countries[c] & Year<2000]$Year%>%unique()%>%as.numeric()
    #VecYears = c(VecYears,years_fore[11:length(years_fore)])
    # VecCounties (with number from 0 to 37 corresponding to each country for each observation)
    VecCountries <- c(VecCountries,rep(c-1,n))
  }
}
    
## Binding observations form different countries into one dataset
## transformation of xtrain from data.preprocessing.CNNs to list of previous xtrain, veccountries and vecgender

x.train <- array(NA, dim=c(obs, dim(ListData[[1]][[1]])[c(2,3)]))
y.train <- array(NA, dim=c(obs,1,dim(ListData[[1]][[2]])[2]))
obsYearVec <- vector()

counter = 0
for (i in 1:(g*c)){
  n <- dim(ListData[[i]][[1]])[1]
  obsYearVec <- c(obsYearVec,ListData[[i]][[3]] )
  for(j in 1:n){
    x.train[counter+j,,] <- ListData[[i]][[1]][j,,]
    y.train[counter+j,1,] <- ListData[[i]][[2]][j,]
  }
  counter <- counter + n
}

# sort to be in a temporal order
OrderByYear <- order(obsYearVec)

x.train.sorted <- x.train[OrderByYear,,]
y.train.sorted <- y.train[OrderByYear,,]
dim(y.train.sorted) <- c(2662,1,100)
VecGenders.sorted <- VecGenders[OrderByYear]
VecCountries.sorted <- VecCountries[OrderByYear]

x.train <- list(x.train.sorted, VecCountries.sorted, VecGenders.sorted)
y.train <- y.train.sorted    
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
  
# define callbacks

  
model_callback <- callback_model_checkpoint(fileName, monitor = "val_loss", verbose = 1,  save_best_only = TRUE, save_weights_only = TRUE)

lr_callback <- callback_reduce_lr_on_plateau(factor=.90, patience = 
                                              50, verbose=1, cooldown = 5, min_lr = 0.00005)

CBs <- list(model_callback, lr_callback)

# gradient descent fitting
t1 <- proc.time()
    fit <- model %>% fit(x=x.train, y=y.train, epochs = 2000, batch_size =16, 
            verbose = 2, validation_split = 0.05, shuffle = T,callbacks=CBs) #in paper 2000 there is no difference at this moment                                       
proc.time()-t1

# in-sample error (validation)       
fit$metrics$val_loss%>%min()

load_model_weights_hdf5(model, fileName)

## recursive prediction
    
# testing data pre-processing
testData <- HMD_final %>% filter(Year > (ObsYear - 10)) 
#vali.Y <- testData %>% fi[which(all_mortV$Year > ObsYear),]

recursive.prediction <- function(ObsYear, all_mort2, gender, country_name, country_index, T0, val.min, val.max, model.p){       
  single.years <- array(NA, c(2016-ObsYear))
  
  for (ObsYear1 in ((ObsYear+1):2016)){
    data1 <- all_mort2 %>% filter(Year >= (ObsYear1-10))
    data2 <- data.preprocessing.CNNs(data1, gender, country_name,T0, ObsYear1)
    # MinMaxScaler (with minimum and maximum from above)
    x.vali <- data2[[1]]
    if (gender=="Female"){yy <- 1}else{yy <- 0}
    x.vali <- list(x.vali, rep(country_index, dim(x.vali)[1]), rep(yy, dim(x.vali)[1]))
    y.vali <- data2[[2]]
    predicted_val <- model %>% predict(x.vali) %>% as.vector()
    predicted_logmx <- (predicted_val*(val.max-val.min)+ val.min)
    Yhat.vali2 <- exp(predicted_logmx)
    ## error calculated on transformed data, in line with Salvatore comment
    single.years[ObsYear1-ObsYear] <- round(10^4*mean((predicted_val-y.vali)^2),4)
    
    predicted <- all_mort2 %>% filter(Year==ObsYear1, Gender == gender, Country == country_name) # [which(all_mort2$Year==ObsYear1),]
    keep <- all_mort2 %>% filter(Year!=ObsYear1, Gender == gender, Country == country_name)
    predicted$logmx <- predicted_logmx
    predicted$mx <- exp(predicted$logmx)
    predicted$val <- predicted_val
    all_mort2 <- rbind(keep,predicted)
    all_mort2 <- all_mort2 %>% arrange(Year, Age)
  }
  list(all_mort2, single.years)
}

    
#example prediction      
pred.CHE.F <- recursive.prediction(1999, testData, "Female", "CHE",(match("CHE", Countries)-1),T0, val.min, val.max, model)
pred.POL.M <- recursive.prediction(1999, testData, "Male", "POL",(match("POL", Countries)-1),T0, val.min, val.max, model)

  
  