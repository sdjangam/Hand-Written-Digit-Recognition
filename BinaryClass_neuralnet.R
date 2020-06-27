# Build Neural Network for classification using neuralnet library.
rm(list=ls(all=TRUE))

# Set the working directory
setwd("C:/Users/gmanish/Dropbox/latest/openminds/slides/MachineLearning/7.ANNs/")
# Importing "data.csv" files's data into R dataframe using read.csv function.
data = read.csv(file="data.csv", header=TRUE, sep=",")

# Understand the structure the summary of the data using str and summary R commands
str(data)
summary(data)
# Using subset remove 'ID' and 'ZIP.Code' columns from  the data
data = subset(data, select = -c(ID,ZIP.Code)) 
# Convert all the variables to appropriate type
#   To numeric using as.numeric()
#   To categoical using as.factor()
data$Education = as.factor(data$Education)
# R NN library takes only numeric attribues as input 
# Convert all categorical  attributes to numeric using appropriate technique. Hint: dummies
# Convert "Education" categorical attribute to numeric using dummy function in dummies R library
# Drop actual Education attribute from orginal data set 
# Add created dummy Education variables to orginal data set
library(dummies)
education = dummy(data$Education)
data = subset(data, select=-c(Education)) 
data = cbind(data, education)
rm(education)
# Separate Target Variable and Independent Variables.
# In this case "Personal.Loan" is a target variable and all others are independent variable. 
target_Variable = data$Personal.Loan
independent_Variables = subset(data, select = -c(Personal.Loan))
# Standardization the independent variables using decostand funcion in vegan R library
library(vegan)
# Note: To standardize the data using 'Range' method
independent_Variables = decostand(independent_Variables,"range")
data = data.frame(independent_Variables, Personal.Loan = target_Variable)
rm(independent_Variables, target_Variable)
# Use set.seed to get same test and train data 
set.seed(123) 
# Prepare train and test data in 70:30 ratio
num_Records = nrow(data)

# to take a random sample of  70% of the records for train data 
train_Index = sample(1:num_Records, round(num_Records * 0.7, digits = 0))
train_Data = data[train_Index,] 
test_Data = data[-train_Index,] 
rm(train_Index, num_Records, data)

# See data distribution in response variable in both Train and Test data:

table(train_Data$Personal.Loan)

table(test_Data$Personal.Loan)

# Load neuralnet R library
library(neuralnet)


# Build a Neural Network having 1 hidden layer with 2 nodes 
set.seed(1234)
nn = neuralnet(Personal.Loan ~ Age+Experience+Income+Family+CCAvg+Mortgage+
                               Securities.Account+CD.Account+Online+CreditCard+
                               Education1+Education2+Education3, 
               data=train_Data, hidden=2,linear.output = F)

# See covariate and result varaibls of neuralnet model - covariate mens the variables extracted from the data argument
out <- cbind(nn$covariate, nn$net.result[[1]])
head(out)
# Remove rownames and set column names
dimnames(out) = list(NULL,c 
                     ("Age","Experience","Income","Family","CCAvg","Mortgage",
                      "Securities.Account","CD.Account","Online","CreditCard",
                      "Education1","Education2", "Education3","nn_Output"))

# To view top records in the data set
head(out) 
rm(out)

# Plot the neural network
plot(nn)

# Compute confusion matrix for train data.
#predicted = factor(ifelse(nn$net.result[[1]] > 0.5, 1, 0))
#conf_Matrix = table(train_Data$Personal.Loan, predicted)


# Remove target attribute from Test Data
test_Data_No_Target = subset(test_Data, select=-c(Personal.Loan))

# Predict 
nn_predict <- compute(nn, covariate= test_Data_No_Target)
rm(test_Data_No_Target)

# View the predicted values
nn_predict$net.result

# Compute confusion matrix and accuracy
predicted = factor(ifelse(nn_predict$net.result > 0.5, 1, 0))
conf_Matrix<-table(test_Data$Personal.Loan, predicted)
sum(diag(conf_Matrix))/sum(conf_Matrix)*100
