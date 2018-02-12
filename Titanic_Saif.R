#Setting Directory
setwd("E:/Google Drive/Kaggle/Titanic")


#Loading Libraries
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

#Loading Datasets
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

#Checking encoding of dataset
str(train)
str(test)

#Basic summary of datasets
summary(train)
summary(test)

#########Data Exploration

#Checking the proportion survived of genders
prop.table(table(train$Sex, train$Survived),1)
#Can see that more proportion of females have survived as compared with males who mainly did not survive


########Feature Engineering

#Creating a child variable for children under the age of 18
train$Child <- 0
train$Child[train$Age < 18] <- 1

#Checking to see if Children with specific genders have a better chance of survival
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#Didn't seem to have any impact

#Binning the Fare variable
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'

#Checking to see if Fare class makes a difference in terms of survival rate
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#Seems that Females which have a Fare Class of 3 and paid more than $20 in their ticket have a lower chance of survival



########Model Building

#Basic Decision Tree
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train,
             method="class")
#Visualizing the Decision Tree
fancyRpartPlot(fit)


########Retraining the model with more feature engineering 

#Reading in the dataset again
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

#Combining the train and test datasets
test$Survived <- NA
combined <- rbind(train,test)

###Feature Engineering and Exploration

str(combined$Name)

#Splitting the Name variable to create new variables
strsplit(combined$Name[1], split='[,.]')
#Extracting the title
strsplit(combined$Name[1], split='[,.]')[[1]][2]
#Creating new column with the titles
combined$Title <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

combined$Title <- sub(' ', '', combined$Title)
table(combined$Title)
#Combining the rare titles to more acceptable ones
combined$Title[combined$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combined$Title[combined$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combined$Title[combined$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

#Converting Titles to a factor level
combined$Title <- factor(combined$Title)


#Family Size

combined$FamilySize <- combined$SibSp + combined$Parch + 1
combined$Surname <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combined$FamilyID <- paste(as.character(combined$FamilySize), combined$Surname, sep="")

combined$FamilyID[combined$FamilySize <= 2] <- 'Small'

table(combined$FamilyID)
famIDs <- data.frame(table(combined$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combined$FamilyID[combined$FamilyID %in% famIDs$Var1] <- 'Small'
combined$FamilyID <- factor(combined$FamilyID)


#Breaking train and test for model building

train <- combined[1:891,]
test <- combined[892:1309,]

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, 
             method="class")

#########Ensemble Modeling

train <- read.csv("train.csv")
test <- read.csv("test.csv")
test$Survived <- NA
combined <- rbind(train, test)


# Convert to a string
combined$Name <- as.character(combined$Name)

# Engineered variable: Title
combined$Title <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combined$Title <- sub(' ', '', combined$Title)
# Combine small title groups
combined$Title[combined$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combined$Title[combined$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combined$Title[combined$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
combined$Title <- factor(combined$Title)

# Engineered variable: Family size
combined$FamilySize <- combined$SibSp + combined$Parch + 1

# Engineered variable: Family
combined$Surname <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combined$FamilyID <- paste(as.character(combined$FamilySize), combined$Surname, sep="")
combined$FamilyID[combined$FamilySize <= 2] <- 'Small'
# Delete erroneous family IDs
famIDs <- data.frame(table(combined$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combined$FamilyID[combined$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
combined$FamilyID <- factor(combined$FamilyID)




#Checking the age variable
summary(combined$Age)
#263 NA's
#Predicting Age
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combined[!is.na(combined$Age),], 
                method="anova")
#Predicting age for all of the NA values
combined$Age[is.na(combined$Age)] <- predict(Agefit, combined[is.na(combined$Age),])

#Checking to see if there are other variables that need to be tackled
summary(combined)

#Embarked
summary(combined$Embarked)
#Has two rows which have blank values
which(combined$Embarked == '')
combined$Embarked[c(62,830)] = "S"
combined$Embarked <- factor(combined$Embarked)
#Replacing the blank entries with S for Southhampton since most passengers boarded from Southhampton

#Fare
summary(combined$Fare)
#1 NA
which(is.na(combined$Fare))

combined$Fare[1044] <- median(combined$Fare, na.rm=TRUE)
#Replacing the missing fare with the median value

combined$FamilyID2 <- combined$FamilyID
combined$FamilyID2 <- as.character(combined$FamilyID2)
combined$FamilyID2[combined$FamilySize <= 3] <- 'Small'
combined$FamilyID2 <- factor(combined$FamilyID2)


train <- combined[1:891,]
test <- combined[892:1309,]



#For reproducibility
set.seed(100)

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FamilySize + FamilyID2,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)


varImpPlot(fit)




set.seed(100)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                   Embarked + Title + FamilySize + FamilyID,
                 data = train, 
                 controls=cforest_unbiased(ntree=2000, mtry=3))


Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "ciforest.csv", row.names = FALSE)


