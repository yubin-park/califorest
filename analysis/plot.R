library(ggplot2)

df <- read.csv("res_simul.csv")

ggplot(df, aes(x=as.factor(test_size), y=brier, colour=model_name)) + geom_boxplot() 
ggsave("fig_simul.png", width=8, height=6)