#Listing 1: Deep neural network model LCCONV with a 1d-CNN layer.
# LCCONV base model
CNN <- function(N,T0, optim="adam"){
rates <- layer_input(shape = c(T0 ,100) , dtype = 'float32', name = 'rates')
#

# Country Embedding

Country <- layer_input(shape = c(1) , dtype = 'int32' , name = 'Country')
Country_embed = Country %>%
  layer_embedding(input_dim = N , output_dim = 5) %>%
  layer_flatten(name = 'Country_embed')
#

# Gender Embedding
# We have seen that using a larger dimension for gender embedding improves
# performance in mortality forecasting, because this allows us for more flexible
# interactions in gradient descent calibrations between gender, age and region.

Gender <- layer_input(shape = c(1), dtype = 'int32' , name = 'Gender')
Gender_embed = Gender %>%
  layer_embedding(input_dim = 2 , output_dim = 5) %>%
  layer_flatten(name = 'Gender_embed')
#


conv = rates %>% layer_conv_1d(filter = 32 , kernel_size = 3 , # kernel_size = m, filter = q
                               activation = 'linear' , padding = "causal") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.35) %>%
  layer_flatten()
#

# decoding z(r, g, U(t0)) results to mortality rates by multivariate GLM (using FCN Layer)
  
decoded = conv %>% list(Country_embed, Gender_embed)%>%
  layer_concatenate() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 100 , activation = 'sigmoid') %>%
  layer_reshape(c(1 ,100) , name = 'forecast_rates')
#
model <- keras_model(inputs = list(rates, Country, Gender) , outputs = c(decoded))
model %>% compile(loss = 'mean_squared_error', optimizer = optim)
return(model)
}