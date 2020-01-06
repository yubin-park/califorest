library(ggplot2)

fn <- "results/breast_cancer.csv"
#fn <- "results/hastie.csv"
df <- read.csv(fn)

ggplot(df, aes(x=model, y=reliability_large, colour=model)) + geom_boxplot() + facet_wrap(n_estimators~max_depth) 


ggplot(df, aes(x=model, y=hosmer_lemshow, colour=model)) + geom_boxplot() + facet_wrap(n_estimators~max_depth) 

ggplot(df, aes(x=model, y=brier, colour=model)) + geom_boxplot() + facet_wrap(n_estimators~max_depth) 

