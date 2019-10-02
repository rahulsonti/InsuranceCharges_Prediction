## Insurance Dataset
## Rahul Sonti


#################################
#################################
## The analysis about predicting the medical charges depending on several factors in the 
##  dataset like age, sex, bmi, number of children, and region.
#################################
#################################


#################################
# Preliminary Analysis and Regression Trees
##################################

##load libraries
library(tidyverse)
library(caret)
library(Amelia)
library(GGally)
library(Rcpp)
library(MASS)
library(tree)
library(rpart)

##import dataset
insurance <- read.csv("insurance.csv")

# plots by total charges for numeric data
par(mfrow = c(1,3))
plot(insurance$charges, insurance$age,
     main = "Plot of Charges by Age",
     xlab = "Total Charges", ylab = "Age of Beneficary")
plot(insurance$charges, insurance$bmi,
     main = "Plot of Charges by BMI",
     xlab = "Total Charges", ylab = "BMI of Beneficiary")
plot(insurance$charges, insurance$children,
     main = "Plot of Charges by Children",
     xlab = "Total Charges", ylab = "Number of Children/Dependents")

##histograms for numeric data
par(mfrow=c(2,2))
hist(insurance$charges, main = "Histogram of Total Charges", 
     xlab = "Total Charges", ylab = "Frequency")
hist(insurance$age, main = "Histogram of Age of Beneficiary",
     xlab = "Age", ylab = "Frequency")
hist(insurance$bmi, main = "Histogram of BMI of Beneficiary",
     xlab = "BMI", ylab = "Frequency")
hist(insurance$children, main = "Histogram of Number of Children",
     xlab = "Children", ylab = "Frequency")

## sex of beneficiary and bar plot
sex <- aggregate(data.frame(sex.count = insurance$sex), 
                 list(state = insurance$sex), length)

sex <- sex[order(-sex$sex.count),]

ggplot(sex, aes(x = reorder(state, -sex.count), y = sex.count))+geom_bar(stat ="identity")+
  labs(x = "Sex",
       y = "Count",
       title = "Sex of Beneficiary")

##region and bar plot
region <- aggregate(data.frame(region.count = insurance$region), 
                    list(state = insurance$region), length)
region <- region[order(-region$region.count),]

ggplot(region, aes(x = reorder(state, -region.count), y = region.count))+geom_bar(stat ="identity")+
  labs(x = "Region",
       y = "Count",
       title = "Region of Residency of Beneficiary")

##smoker and bar plot
smoker <- aggregate(data.frame(smoker.count = insurance$smoker), 
                    list(state = insurance$smoker), length)
smoker <- smoker[order(-smoker$smoker.count),]

ggplot(smoker, aes(x = reorder(state, -smoker.count), y = smoker.count))+geom_bar(stat ="identity")+
  labs(x = "Smoker",
       y = "Count",
       title = "Count of Beneficiary Smoking Habit")

##view structure of data set
str(insurance)
levels(insurance$sex)<-c("F","M")

##create age group column
insurance %>%
  select_if(is.numeric) %>%
  map_dbl(~max(.x))

##breaking age into 3 groups, breaking region into the 4 regions 
##and smokers into yes and no
##minimum age is 18 and max age is 64
insurance<-insurance %>% 
  mutate(age=as.factor(findInterval(age,c(18,33,48,65))))
levels(insurance$age)<-c("Youth","Mid Aged","Old")
levels(insurance$smoker)<-c("N","Y")
levels(insurance$region)<-c("NE","NW","SE","SW")

##visualize distribution of charges based on age group, sex and region
insurance %>%
  ggplot(aes(region, charges, fill=sex)) + geom_boxplot() + facet_grid(~age)+
  ggtitle("Insurance charges based on age group, sex and region") + 
  scale_fill_manual(values = c("pink", "steelblue"))

##visualize distribution of charges based on smoker status
ggplot(insurance, aes(smoker, charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(breaks = seq(0, 65000, by = 5000)) +
  stat_summary(fun.y = mean, geom = "point", color = "red", size = 2) +
  xlab("Smoker") +
  ylab("Medical Cost Charges") +
  theme(legend.position = "none")



# Here we fit a regression tree to the data set. First, we create a
# training set, and fit the tree to the training data.
set.seed(1)
train = sample(1:nrow(insurance), nrow(insurance)/2)
tree.insurance=tree(charges~.,insurance,subset=train)
summary(tree.insurance)

# Notice that the output of summary() indicates that only three of the variables have been used in constructing
# the tree. In the context of a regression tree, the deviance is simply the sum of squared errors for the tree. 
# We now plot the tree.
#
#[Uncomment below for Mac]
#quartz()
plot(tree.insurance)
text(tree.insurance,pretty=0)


# Now we use the cv.tree() function to see whether pruning the tree will
# improve performance.
cv.insurance=cv.tree(tree.insurance)
plot(cv.insurance$size,cv.insurance$dev,type='b')


# do not need to purne tree so using unpruned tree to 
# make predictions on the test set.
yhat=predict(tree.insurance,newdata=insurance[-train,])
insurance.test=insurance[-train,"charges"]

#[Uncomment below for Mac]
#quartz()
plot(yhat,insurance.test)
abline(0,1)

mean((yhat-insurance.test)^2)
sqrt(mean((yhat-insurance.test)^2))


#################################
# Fitting Random Forests
##################################


library(randomForest)

# Growing a random forest
set.seed(1)
rf.insurance=randomForest(charges~.,data=insurance,subset=train,mtry=5,importance=TRUE)
yhat.rf = predict(rf.insurance,newdata=insurance[-train,])

mean((yhat.rf-insurance.test)^2) #test set MSE
sqrt(mean((yhat.rf-insurance.test)^2))

#variable importance
importance(rf.insurance)
varImpPlot(rf.insurance)


#################################
# Logistic Regression
##################################

library(tidyverse)
library(ggplot2)
library(Sleuth3)
library(randomForest)

####################
###################
# Reading the insurance dataset

insurance <- read.csv("insurance.csv", header=TRUE)

Insurance_Charges <- na.omit(insurance)

Insurance_Charges

hist <- ggplot(insurance, aes(charges)) + geom_histogram(bins=200)

hist

########################
########################

# Setting a variable to know the credit worthiness of the person 
# Charges more than 40000 indicates that the person has more credit score

Charges_sorted <- Insurance_Charges[order(x= Insurance_Charges$charges, decreasing = TRUE),]

Charges_sorted

topQuarter <- floor(nrow(Charges_sorted)/4)

Charges_sorted_topquarter <- Charges_sorted[1:topQuarter,]

Highest_Charges_threshold <- Charges_sorted_topquarter[topQuarter,]$charges

Insurance_Charges$charges <- ifelse(insurance$charges>=20000,1,0)

View(Insurance_Charges)

########################
########################
# Splitting the data into test and training data sets

library(caTools)
set.seed(94)
split = sample.split(Insurance_Charges, SplitRatio = 0.7)
Insurance_train = subset(Insurance_Charges, split == TRUE)
Insurance_test = subset(Insurance_Charges, split==FALSE)

########################
########################
# Creating a logistic regression model

log_mod <- glm(charges~., data= Insurance_train, family= "binomial", control = list(maxit = 50))

########################

# Summary stats for the model 

summary(log_mod)

#######################

# One unit increase in other attributes affecting the Insurance charges to be more than 20,000

exp(coef(log_mod))

predict.charges1 <- predict(log_mod, Insurance_test, type= "response", interval= "confidence")

predict.charges1

predict.charges1 <- ifelse(predict.charges1>0.5, 1,0)

table(predict.charges1, Insurance_test$charges)

##########################################

log_mod1 <- glm(charges~age+bmi+smoker, data= Insurance_train, family= "binomial")

########################

# Summary stats for the model 

summary(log_mod1)

#######################

# One unit increase in other attributes affecting the Insurance charges to be more than 20,000

exp(coef(log_mod1))

predict.charges2 <- predict(log_mod1, Insurance_test, type= "response", interval="confidence")

predict.charges2

predict.charges2 <- ifelse(predict.charges1>0.5, 1,0)

table(predict.charges2, Insurance_test$charges)

############
# Mean Square Error
############

mean(predict.charges2 != Insurance_test$charges) 

############
# Accuracy
############

mean(predict.charges2 == Insurance_test$charges)

#######################
#######################

# Linear Plot of charges with respect to smoker or not

insurance%>%
  mutate(prob = ifelse(smoker == "yes", 1, 0)) %>%
  ggplot(aes(charges, prob)) +
  geom_point(alpha = .15) +
  geom_smooth(method = "lm") +
  ggtitle("Linear regression model fit") +
  xlab("Charges") +
  ylab("Probability of Smokers")

#######################
#######################

# Logistic regression plot with respect to smoker or not

insurance %>%
  mutate(prob = ifelse(smoker == "yes", 1, 0)) %>%
  ggplot(aes(charges, prob)) +
  geom_point(alpha = .15) +
  geom_smooth(method = "glm", method.args = list(family = "binomial",control = list(maxit = 50))) +
  ggtitle("Logistic regression model fit") +
  xlab("Charges") +
  ylab("Probability of Smokers")


#################################
# PCA
##################################

insurance <-read.csv("C:/Users/home/Desktop/STAT 515/Assignment/project/insurance.csv",header=TRUE,sep=',')
region=row.names(insurance$region)
drop<-c("sex","smoker","region","children")
insurance<-insurance[,!(names(insurance)%in% drop)]
lapply(insurance, mean)
lapply(insurance, var)
pca_insurance=prcomp(insurance,center=TRUE,scale=TRUE)
names(pca_insurance)
pca_insurance$center
pca_insurance$scale
pca_insurance$rotation
dim(pca_insurance$x)
biplot(pca_insurance, scale=0)



