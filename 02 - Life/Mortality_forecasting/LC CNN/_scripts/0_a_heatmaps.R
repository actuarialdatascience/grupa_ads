
##################################################################
#### Heatmap of selected data
##################################################################

## run after 0_dataReading as HMD_final is necessary

mortality_heatmap <- function(mat, country, gender, year_min, year_max){
  m0 <- c(min(mat), max(mat))
  # rows are calendar year t, columns are ages x
  image(z=mat, useRaster=TRUE,  zlim=m0, col=rev(rainbow(n=60, start=0, end=.72)), xaxt='n', yaxt='n', main=list(paste(country," ",gender, " raw log-mortality rates", sep=""), cex=1.5), cex.lab=1.5, ylab="age x", xlab="calendar year t")
  axis(1, at=c(0:(year_max-year_min))/(year_max-year_min), c(year_min:year_max))                   
  axis(2, at=c(0:49)/50, labels=c(0:49)*2)                   
  lines(x=rep((1999-year_min+0.5)/(year_max-year_min), 2), y=c(0:1), lwd=2)
}

Genders <- tibble(Gender = c("Female","Male"))
Countries <- HMD_final %>% select(Country) %>% distinct()


for(g in unlist(Genders)){
  for(c in unlist(Countries)){
    tmp <- HMD_final %>% filter(Gender == g, Country == c) %>% select(Age,Year, logmx) %>%
      pivot_wider(names_from = Age, values_from = logmx)
    year_min <- tmp %>% select(Year) %>% min()
    year_max <- tmp %>% select(Year) %>% max()
    tmp <- tmp %>% select(-Year) %>% as.matrix()
    mortality_heatmap(tmp, c, g, year_min, year_max)    
  }
}
