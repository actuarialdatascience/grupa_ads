    library(HMDHFDplus)
library(dplyr)
library(tidyr)

stat <- "Mx_1x1"
filetype <- "txt"
countries <- c("AUS", "AUT", "BEL", "BGR", "BLR", "CAN","CHE","CHL","CZE","DEUTNP","DNK",
               "ESP", "EST", "FIN", "FRATNP", "GBRTENW", "GBR_NIR", "GBR_SCO",
               "GRC", "HRV", "HUN", "IRL", "ISL", "ISR", "ITA", "JPN", "LTU", "LUX", "LVA",
               "NLD", "NOR", "NZL_NM", "POL", "PRT", "RUS", "SVK", "SVN","SWE", "TWN", "UKR", "USA")

files <- paste(countries, stat, filetype, sep = ".")

HMD_list <- list()

for (i in 1:length(files)){
HMD_list[[countries[i]]] <- readHMD(files[i]) %>% filter(Year >= 1950 & Year <= 2016 & Age < 100) %>% 
                            select(-c(Total,OpenInterval)) %>% 
                            pivot_longer(c(Female,Male), names_to = "Gender", values_to = "mx" )
}

rm(list = c("files", "countries", "filetype","stat", "i"))

HMD_data <- bind_rows(HMD_list, .id = "Country")

Analyzed_Countries <- HMD_data %>% select(Country, Year) %>% distinct() %>% filter(Year < 2000) %>% 
                      group_by(Country) %>% summarise(n= n()) %>% filter(n>10) %>% select(Country) %>% unlist()

HMD_data_chosen <- HMD_data %>% filter(Country %in% Analyzed_Countries) %>% mutate(mx = if_else(mx < 1, mx, NA_real_))

first_observed <- HMD_data_chosen %>% group_by(Country) %>% summarize(MinYear = min(Year))

#HMD_train <- HMD_data_chosen %>% filter(Year < 2000)
#HMD_test <- HMD_data_chosen %>% filter(Year >= 2000) %>% mutate(logmx = log(mx))



rm(list = c("Analyzed_Countries","HMD_data", "HMD_list"))

## missing imputation

Ages <- tibble(Age = seq(0,99))
Years <- tibble(Year = seq(1950, 2016))
Genders <- tibble(Gender = c("Female","Male"))
Countries <- HMD_data_chosen %>% select(Country) %>% distinct()

template <- full_join(Ages, Years, by = character()) %>% 
            full_join(Countries, by = character()) %>%
            full_join(Genders, by = character())

rm(list = c( "Countries", "Genders","Ages","Years"))


#HMD_train %>% filter(mx == 0) %>% nrow()
#(template %>% nrow()) - (HMD_train %>% nrow())

HMD_full <- template %>% left_join(HMD_data_chosen, by = c("Age","Year","Gender","Country")) %>%
                  mutate(mx = na_if(mx,0))

toImpute <- HMD_data_chosen %>% filter(mx >0) %>% group_by(Age, Gender, Year) %>% summarize(mx_avg = mean(mx))

HMD_imputed <- HMD_full %>% inner_join(toImpute, by = c("Age", "Gender", "Year")) %>%
                  mutate(imputed_flag = is.na(mx), mx = if_else(is.na(mx),mx_avg,mx), logmx = log(mx)) %>% 
                  select(-mx_avg)
HMD_final <- HMD_imputed %>% inner_join(first_observed, by = "Country") %>% filter(Year >= MinYear) %>% select(-MinYear)

#HMD_train <- HMD_train_final

#rm(list = c("HMD_train_final","HMD_train_full","toImpute","template"))
rm(list = c("HMD_data_chosen","HMD_full","toImpute","template", "HMD_imputed", "first_observed"))
