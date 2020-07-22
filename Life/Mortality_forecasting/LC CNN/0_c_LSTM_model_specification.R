
LSTM <- function(N,T0, tau0, optim="adam"){
rates <- layer_input(shape = c(T0 ,tau0) , dtype = 'float32' , name = 'rates')
#
Country <- layer_input( shape = c(1) , dtype = 'int32' , name = 'Country')
Country_embed = Country %>%
  layer_embedding(input_dim = N , output_dim = 5) %>%
  layer_flatten(name = 'Country_embed')
#
Gender <- layer_input(shape = c(1) , dtype = 'int32' , name = 'Gender')
Gender_embed = Gender %>%
  layer_embedding(input_dim = 2 , output_dim = 5) %>%
  layer_flatten(name = 'Gender_embed')
#
LSTM1 = rates %>% layer_lstm(units = 128 , activation = "linear" ,
                                recurrent_activation = "tanh", return_sequences = F ) %>%
  layer_batch_normalization() %>%
  layer_dropout (rate = 0.35)
#
decoded = LSTM1 %>% list ( Country_embed , Gender_embed )%>%
  layer_concatenate() %>%
  layer_dropout ( rate = 0.4) %>%
  layer_dense ( units = tau0 , activation = 'sigmoid') %>%
  layer_reshape ( c(1 ,tau0) , name = 'forecast_rates')
#
model <- keras_model( inputs = list ( rates , Country , Gender ) , outputs = c ( decoded ))
model %>% compile(loss = 'mean_squared_error', optimizer = optim)
return(model)
}