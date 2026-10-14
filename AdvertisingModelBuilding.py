import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

def Advertize(dataPath):
    
    Border = "-"*40
    #--------------------------------------------------------
    #   Step 1 : Load Dataset
    #---------------------------------------------------------
    df = pd.read_csv(dataPath)

    print("Few recorsds from dataset ")
    print(df.head())

    #--------------------------------------------------------
    #   Step 2 : remove unwanted columns
    #---------------------------------------------------------

    
    print(Border)
    print("Step 2 : remove unwanted columns")
    print(Border)

    print("Shape of dataset before removal :")
    print(df.shape)
    
    if 'Unnamed : 0' in df.columns:
            df.drop(columns=["unnamed : 0"],inplace=True)

    print("Shape of dataset After removal :")
    print(df.shape)

    print("Clean Dataset is :")
    print(df.head())

    #--------------------------------------------------------
    #   Step 3 : cheack missing value 
    #---------------------------------------------------------
    
    print(Border)
    print("Step 3 : cheack missing value ")
    print(Border)

    print(df.isnull().sum())

     #--------------------------------------------------------
    #   Step 4 : Display Statistical Summary 
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 4 : Display Statistical Summary ")
    print(Border)

    print(df.describe())


      #--------------------------------------------------------
    #   Step 5: Corelation between columns
    #---------------------------------------------------------
    
    print(Border)
    print("Step 5: Corelation between columns")
    print(Border)

    print(df.corr())

      #--------------------------------------------------------
    #   Step 6: Split dataset into dependent and independent variable
    #---------------------------------------------------------
    
    print(Border)
    print("Step 6: Split dataset into dependent and independent variable")
    print(Border)

    X= df[['TV','radio','newspaper']]
    Y=df['sales']

    print("Shape of Independent Variable :",X.shape)
    print("Shape of dependent Variable :",Y.shape)

       #--------------------------------------------------------
    #   Step 7: Split dataset for trainning and testing 
    #---------------------------------------------------------
    
    print(Border)
    print("Step 7: Split dataset for trainning and testing")
    print(Border) 

    X_train,X_test,Y_train,Y_test =train_test_split(X,Y,random_state=42,test_size=0.2)

    print("X_train:",X_train.shape)
    print("Y_train:",Y_train.shape)
    print("X_test:",X_test.shape)
    print("Y_test:",Y_test.shape)


       #--------------------------------------------------------
    #   Step 8: Create and train the model 
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 8: Create and train the model ")
    print(Border) 

    model = LinearRegression()

    model.fit(X_train,Y_train)

    print("Model trainning gets Sucessfull")

      #--------------------------------------------------------
    #   Step 9: test the model 
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 9: test the model ")
    print(Border)

    Y_pred = model.predict(X_test)

    print("Model Testing gets Sucessfull")

    #--------------------------------------------------------
    #   Step 10: Evaluate the model 
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 10: Evaluate model ")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)
    R2 =r2_score(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)

    print("Mean Square Error ",MSE)
    print("Root mean square ",RMSE)
    print("R square value ",R2)

    #--------------------------------------------------------
    #   Step 11:  calculate model coefficient  
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 11:  calculate model coefficient ")
    print(Border)

    for column,value in zip(X.columns,model.coef_):
        print(f"{column}:{value}")
        print("Intercept :",model.intercept_)

    #--------------------------------------------------------
    #   Step 12:  compare the actual & predicted values   
    #---------------------------------------------------------
    
    print(Border)
    print(" Step 12:  compare the actual & predicted values ")
    print(Border)

    Result = pd.DataFrame({'Actual sales:':Y_test.values,'predicted sales:':Y_pred})

    print(Result.head())


def main():
    
    Advertize("Advertising.csv")
    

    
if __name__ == "__main__":
    main()
