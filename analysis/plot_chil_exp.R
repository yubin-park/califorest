library(ggplot2)

dataset_name <- "hastie"

palette5 <- c("#2b83ba", "#abdda4", "#d7191c", "#fdae61", "#b0b0b0")

fn <- paste0("results/", dataset_name, ".csv")

df <- read.csv(fn)
df.long <- rbind(data.frame(model=df$model, var="Scaled Brier Score", val=df$brier_scaled),
				data.frame(model=df$model, var="Hosmer-Lemeshow p-value", val=df$hosmer_lemshow),
				data.frame(model=df$model, var="Spiegelhalter p-value", val=df$speigelhalter),
				data.frame(model=df$model, var="Brier Score", val=df$brier),
				data.frame(model=df$model, var="Reliability-in-the-small", val=df$reliability_small),
				data.frame(model=df$model, var="Reliability-in-the-large", val=df$reliability_large))

ggplot(df.long, aes(x=model, y=val, colour=model)) + 
geom_boxplot() + 
facet_wrap(var~., scale="free", nrow=2) + 
theme_minimal() + 
theme(axis.text.x = element_text(angle=90, hjust=1), 
	axis.title.y = element_blank(),
	legend.position="none") + 
scale_color_manual(values=palette5)

ggsave("hastie-results.png", width=8, height=5)
