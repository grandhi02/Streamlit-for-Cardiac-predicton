 import os
 import warnings   # it is important to remove warnings
 warnings.filterwarnings('ignore')
 import streamlit as st
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.preprocessing import MinMaxScaler
 from sklearn.impute import SimpleImputer
 from collections import Counter
 from sklearn.ensemble import RandomForestClassifier
 import copy
 from heapq import heapify,heappop,heappush
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
 import numpy as np
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 
 
 diseases={1:"Normal",2:"Ischemic changes (Coronary Artery Disease)",3:"Old Anterior Myocardial Infarction",4:"Old Inferior Myocardial Infarction ",
    5:"Sinus tachycardy ",6:" Sinus bradycardy ",7:" Ventricular Premature Contraction (PVC) ",8:" Supraventricular Premature Contraction ",
    9:"  Left bundle branch block",10:" Right bundle branch block ",11:"degree AtrioVentricular block",12:"degree AV block",13:"degree AV block ",14:"Left ventricule hypertrophy",15:" Atrial Fibrillation or Flutter",16:"others"
   }
 
 #st.title(" Cardiac Arrhythmia Prediction using Streamlit")
 # st.subheader("Simple  with Streamlit")
 
 html_temp="""
 <div style="background-color:green;"><p style="color:white">Cardiac Arrhythmia Prediction using Streamlit</p></div>
 """
 # Below One is required to have html code
 st.markdown(html_temp,unsafe_allow_html=True)
 input="/path_to_dataset "
 p=pd.read_csv(os.path.join(input,"cardiac_ris.csv"),delimiter=";")
 st.write(p.shape)
 st.info(" you selected {}".format("cardiac_risk"))
 st.write("Data Types",p.dtypes)
 
 # Type Conversion
 for i,j in enumerate(p.dtypes):
     if j=="object":
         p.iloc[:,0].astype(int)
 # show sample dataset
 if st.checkbox("Show Dataset"):
     number=st.number_input("Number of Rows to view",5,10)
     st.dataframe(p.head(number))
 
 # show columns
 if st.button("Column Names"):
     st.write(p.columns)
 if st.checkbox("Show Shape"):
     #st.write(p.shape)
     # it makes two checkbox which only one can be selected
     k=st.radio("Select",("rows","columns")) 
     if k=="rows":
         st.write("Number of rows",p.shape[0])
     elif k=="columns":
         st.write("Number of columns",p.shape[1])
     else:
         st.write(p.shape)
 
 # Now to select only some columns
 if st.checkbox("Select Some Columns"):
     column=list(p.columns)  # it shoulb be iterable
     selected_columns=st.multiselect("select",column) # column arg should be list
     new_p=p[selected_columns]
     st.dataframe(new_p)
 
 # Button only used for once of its use
 
 # Finding target distint value counts
 
 if st.button("value counts"):
     st.write("Value counts of target",p.iloc[:,-1].value_counts())
 
 # Finding datatypes
 
 if st.button("DataTypes"):
     st.write("Data Types of dataset",p.dtypes)
 
 # Now go for the summary
 if st.checkbox("Summary"):
     st.write("The summary",p.describe())
 
 # Splitting the data
 target1=p["diagnosis"]
 p.drop("diagnosis",inplace=True,axis=1)
 splitted=p
 p,x_test,target,y_test=train_test_split(p,target1,test_size=0.1,random_state=101)   
 x_train,test,tar,yt=train_test_split(splitted,target1,test_size=0.1,random_state=101)  
 
  
 # Rescaling the Data
 
 st.subheader("MinMaxScaler")
 
 st.dataframe(target.head())
 x=p.select_dtypes(include=["number"])
 y=x.columns
 scaler=MinMaxScaler(feature_range=(0,1))
 rescaledx=scaler.fit_transform(x)
 rescaledx=pd.DataFrame(rescaledx,columns=y)
 
 if st.checkbox("Rescaled Data"):
     number=st.number_input("Number of Rows to views",5,10)
     st.dataframe(rescaledx.head(number))
     st.write(len(rescaledx.columns))
 
 st.subheader("Checking Missing Values")
 
 if st.checkbox("Check missing values"):
     k=rescaledx.isnull().sum()
     st.write(k)
     my_imputer=SimpleImputer()
     imputed_data=pd.DataFrame(my_imputer.fit_transform(rescaledx))
     imputed_data.columns=rescaledx.columns
     obtained=imputed_data
 
 
 # Finding Anamolies
 def plot(l,o):
     fig=plt.figure(11,(15,10))
     for i,col in enumerate(o.columns[:10]):
         ax=plt.subplot(4,3,i+1)
         st.write(sns.distplot(o[col], kde = True, color ='red', bins = 30))
         plt.tight_layout()
     st.pyplot()
 
 def detect_outlier(col,dataframe,col1):
     outlier=[]
     threshold=3         #   if any value greater than the 3,it is outlier after the finding the z_score
     mean=np.mean(col)           #  finding the mean using numpy
     stdeviation=np.std(col)      #  finding the standard deviation
     
     for j,i in enumerate(col):
         z_score=(i-mean)//stdeviation
         if z_score > threshold:
             outlier.append(i)   # adds to list when it is outlier
             dataframe[col1][j]=mean
             
     return outlier
 
 
 
 def outliers(dataframe):
   out=[]
   length1=len(dataframe)          # preserving the length of dataframe before finding ouliers
   l=list(dataframe.describe())   # identifying the continuous columns
  # print("Continous columns")
  # print(*l)
   d=Counter(l)
   plot(l[:10],dataframe)
   for i in l:
     d[i]=detect_outlier(dataframe[i],dataframe,i)   # we are detecting the outliers in each and 
                                  every columns
                                         
   # now outliers are stored in a dictionary,without empty list
   # we have to identify non empty lists in dictioanry
 
   for i in d:
     if len(d[i])>0:
       out.append(i)                      # this out list has oulier column variables
 
   print("outliers")
   st.write(*out,len(out))  
 
 
 st.write(obtained.shape)
 outliers(obtained)
 
 
 model = LogisticRegression(solver='lbfgs')
 rfe = RFE(model,100)
 fit = rfe.fit(obtained, target)
 
 new=[]
 from collections import Counter
 d=Counter(fit.ranking_)
 for i,j in enumerate(fit.support_):
     if j==True:    new.append(obtained.columns[i])
 
 new_data=obtained[new]
 o=new_data.select_dtypes(exclude=["object"])
 
 
 if st.checkbox("Feature to be selected"):
     number=st.number_input("Number of Features to selected",20,150)
     st.dataframe(new[0:number])
     new_obtained=obtained[new[0:number]]
 st.subheader("Data Visualisation")
 
 # we are going to use correlation plot
 #                    Seaborn plot
 #                    Count plot
 #                    pie chart
 #                    Customizable Plot
 
 if st.checkbox("Correlation Plot"):
     st.write(sns.heatmap(new_obtained.corr(),annot=True))
     st.pyplot()
 
 if st.checkbox("Pie Plot"):
     all_columns=list(p.columns)
     st.write("Pie Plot for target")
     st.write(target.iloc[:].value_counts().plot.pie())
     st.pyplot()
 
 st.write("""
 #### Customizable plot
 """ )
 all_columns=list(p.columns)
 type_of_plot=st.selectbox("select type of plot",["area","bar","line","kde"])
 selected_columns_names=st.multiselect("select columns",all_columns)
 
 
 if st.checkbox("Generate Plot"):
     st.success("Generate Customizable Plot of {} for {} ".format(type_of_plot,selected_columns_names))
     
 
     if type_of_plot=="area":
         cust_data=p[selected_columns_names]
         st.area_chart(cust_data)
     elif type_of_plot=="line":
         cust_data=p[selected_columns_names]
         st.line_chart(cust_data)
     elif type_of_plot=="bar":
         cust_data=p[selected_columns_names]
         st.bar_chart(cust_data)
     elif type_of_plot=="kde":
         cust_data=p[selected_columns_names].plot(kind="kde")
         st.write(cust_data)
         st.pyplot()
 x_test=x_test.select_dtypes("number")
 
 st.subheader("Model Creation")
 
 
 if st.button("Model"):
 
     from sklearn.model_selection import RandomizedSearchCV
     # Number of trees in random forest
     n_estimators = [int(x) for x in np.linspace(start = 200, stop = 700, num = 10)]
     # Number of features to consider at every split
     max_features = ['auto', 'sqrt']
     # Maximum number of levels in tree
     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
     max_depth.append(None)
     # Minimum number of samples required to split a node
     min_samples_split = [2, 5, 10]
     # Minimum number of samples required at each leaf node
     min_samples_leaf = [1, 2, 4]
     # Method of selecting samples for training each tree
     bootstrap = [True, False]
     # Create the random grid
     random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}
 
     rf = RandomForestClassifier()
     # Random search of parameters, using 3 fold cross validation, 
     # search across 100 different combinations, and use all available cores
     rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,       
     n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
 
 
     dataset=pd.read_csv(os.path.join(input,"cardiac_ris.csv"),delimiter=";")
     target2=dataset["diagnosis"]
     dataset.drop("diagnosis",inplace=True,axis=1)
     dataset=dataset.select_dtypes(exclude=["object"])
     rf_random.fit(dataset,target2)
     st.write(rf_random.best_params_)
     dict=rf_random.best_params_
 
     x_test=x_test.select_dtypes(exclude=["object"])
 
     def evaluate(model, test_features, test_labels):
         predictions = model.predict(test_features)
         errors = abs(predictions - test_labels)
         mape = 100 * np.mean(errors / test_labels)
         accuracy = 100 - mape
         print('Model Performance')
         print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
         print('Accuracy = {:0.2f}%.'.format(accuracy))
         return accuracy
 
     o=pd.read_csv("cardiac_ris.csv",sep=";")
    # o.drop("diagnosis",inplace=True)
     dataset=o.select_dtypes(exclude=["object"])
     xtrain,xtest,ytrain,ytest=train_test_split(dataset,target2,test_size=0.1,random_state=101)
     base_model = RandomForestClassifier(n_estimators =dict["n_estimators"], 
                            random_state =42, max_features=dict["max_features"],  
            min_samples_leaf=dict["min_samples_leaf"],
            max_features=dict["max_features"] ,
            max_depth=dict["max_depth"] ,
    bootstrap=dict["bootstrap"])     
     base_model.fit(xtrain,ytrain)
     base_accuracy = evaluate(base_model, xtest,ytest)
     st.write("accuracy is ",base_accuracy)
     st.write("Prediction for a single 
     item",base_model.predict(np.array(dataset.loc[1]).reshape(1,-1)))  
     st.write("He has ",diseases[base_model.predict(np.array(dataset.loc[1]).reshape(1,-1))[0]])
     st.write(np.array(dataset.loc[1]).reshape(1,-1)[0])
 
 
 st.subheader("Our Prediction")
 age1=st.number_input("Give the age",10,90)
 gender=st.number_input("Enter the Gender",0.0,1.0)
 height=st.number_input("Enter the Height",100,800)
 weight=st.number_input("Enter the weight",6,180)
 qrs_duration=st.number_input("Enter the BP",55,190)
 heart=st.number_input("Enter the printerval",40,200)
 qtinterval=st.number_input("Enter the sugar value",100,480)
 tinterval=st.number_input("Enter the tinterval",100,400)
 pinterval=st.number_input("Enter the printerval",40,300)
 qrs=st.number_input("Enter the qrs value",10,190)
 
 if gender==1.0:
     gender=1
 else:
     gender=0
 l=[age1,gender,height,weight,qrs_duration,heart,qtinterval,tinterval,pinterval,qrs]
 
 
 
 
 data=np.array(l)
 data_to_append = {}
 for i in range(len(dataset.columns)):
     data_to_append[dataset.columns[i]] = data[i]
 df = pd.DataFrame(data_to_append,index=[0])
 prediction=base_model.predict(np.array(df.loc[0]).reshape(1,-1))
 st.write("He has ",diseases[prediction[0]]) 
