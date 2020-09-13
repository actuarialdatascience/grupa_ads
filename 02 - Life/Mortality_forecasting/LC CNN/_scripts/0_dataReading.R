options(scipen=999)

library(HMDHFDplus)
library(dplyr)
library(tidyr)

## reading the data from Mx_1x1 tables from HMD for Countries listed in appendix A of paper:
## A Neural Network Extension of the Lee-Carter Modelto Multiple Populations Ronald Richman∗Mario V. W ̈uthrich (Version of October 22, 2018)


stat <- "Mx_1x1"
filetype <- "txt"

countries <- c("AUS", "AUT", "BEL", "BGR", "BLR", "CAN","CHE","CHL","CZE","DEUTNP","DNK",
                 "ESP", "EST", "FIN", "FRATNP", "GBRTENW", "GBR_NIR", "GBR_SCO",
                 "GRC", "HRV", "HUN", "IRL", "ISL", "ISR", "ITA", "JPN", "LTU", "LUX", "LVA",
                 "NLD", "NOR", "NZL_NM", "POL", "PRT", "RUS", "SVK", "SVN","SWE", "TWN", "UKR", "USA")

files <- paste(countries, stat, filetype, sep = ".")
long_dir_files <- paste(data_folder,files, sep="/")

HMD_list <- list()

## filtering the data by Age (<100) and Year(1950 <= Year <= 2016)
## getting rid of columns "Total" and "OpenInterval"
## pivoting the data to create column Gender and column mx from columns "Female" and "Male".

for (i in 1:length(long_dir_files)){
HMD_list[[countries[i]]] <- readHMD(long_dir_files[i]) %>% filter(Year >= 1950 & Year <= 2016 & Age < 100) %>% 
                            select(-c(Total,OpenInterval)) %>% 
                            pivot_longer(c(Female,Male), names_to = "Gender", values_to = "mx" )
}

rm(list = c("files", "countries", "filetype","stat", "i"))

HMD_data <- bind_rows(HMD_list, .id = "Country") %>% data.table()


## Choosing the cuntries with more than 10 years of observation before year 2000
Analyzed_Countries <- HMD_data %>% select(Country, Year) %>% distinct() %>% filter(Year < 2000) %>% 
                      group_by(Country) %>% summarise(n= n()) %>% filter(n>10) %>% select(Country) %>% unlist()

HMD_data_chosen <- HMD_data %>% filter(Country %in% Analyzed_Countries)

## settin mx to NA if mx =< 0 or mx >= 1
HMD_data_chosen <- HMD_data_chosen %>% 
  mutate(mx = if_else(mx > 0, mx, NA_real_))

rm(list = c("Analyzed_Countries","HMD_data", "HMD_list"))

## imputting values with NA in mx column by average by Age, Gednder and Year from all countries read in point 1
toImpute <- HMD_data_chosen %>% filter(mx >0) %>% group_by(Age, Gender, Year) %>% summarize(mx_avg = mean(mx))

HMD_final <- HMD_data_chosen %>% inner_join(toImpute, by = c("Age", "Gender", "Year")) %>%
                  mutate(imputed_flag = is.na(mx), mx = if_else(is.na(mx),mx_avg,mx), logmx = log(mx)) %>% 
                  select(-mx_avg)

rm(list = c("HMD_data_chosen","toImpute"))
