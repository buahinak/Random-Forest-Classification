import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
from sklearn import linear_model as lm
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import seaborn as sns
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve


class buildingmodel:
    def __init__(self,pathfulldata):
        """
        Pathfulldata: path to folder with all relevant data
        :param pathfulldata:
        """
        self.path = pathfulldata

    def processdata(self):
        """
        Process,join, clean and compile data
        :return:
        player_data:dataframe with cleaned and processed data
        """
        filenames = os.listdir(os.path.expanduser(self.path))
        advanced=pd.DataFrame()
        per100 = pd.DataFrame()
        countingstats = pd.DataFrame()
        allstar=pd.DataFrame()
        adjustedshot=pd.DataFrame()



        for file in filenames:
            year = file[0:7]
            if "ALL STAR" in file:
                df = pd.read_csv(str("~/Documents/GWU DATA ANALYTICS /EMSE 6574/NBA DATA/" + file),header=None)
                df = pd.DataFrame(df)
                df = df.set_axis(["Player"],axis=1)
                df["Year"] = file[0:7]
                df["All Star"] = 1
                allstar=allstar.append(df)

        for file in filenames:
            file = str(file)
            pathstring = str(self.path+file)
            df = pd.read_csv(pathstring,encoding="ISO-8859-1")
            df = pd.DataFrame(df)
            if "adv" in file:
                df['Year'] = file[0:7]
                advanced = advanced.append(df)
            elif "per 100" in file:
                df['Year'] = file[0:7]
                per100 = per100.append(df)
            elif "nba" in file:
                df['Year'] = file[0:7]
                countingstats = countingstats.append(df)
            elif "adj" in file:
                df.columns = df.iloc[0]
                df.drop(index=df.index[0],
                        axis=0,
                        inplace=True)
                df['Year'] = file[0:7]
                df = df.dropna(how='all', axis=1)
                df.dropna(subset=["Player"], inplace=True)
                adjustedshot = adjustedshot.append(df)

        def removehofstarandsort(data):
            """
            remove "*" from player names
            :param data:
            :return:
            """
            newplayerlist = []
            for i in data.Player:
                if "*" in i:
                    i = i[0:-1]
                    newplayerlist.append(i)
                else:
                    newplayerlist.append(i)

            data.Player = newplayerlist
            data = data.sort_values(by=["Player", "Year"])

        removehofstarandsort(advanced)
        removehofstarandsort(countingstats)
        removehofstarandsort(per100)

        player_data = pd.merge(countingstats, allstar, on=["Player", "Year"], how="left")
        player_data["All Star"] = player_data["All Star"].fillna(0)


        player_data = player_data.groupby(["Player","Year"]).first()
        advanced = advanced.groupby(["Player","Year"]).first()
        per100 = per100.groupby(["Player","Year"]).first()

        player_data=player_data.reset_index()
        advanced=advanced.reset_index()
        per100=per100.reset_index()

        player_data = player_data.merge(per100, on=["Player","Year"],how='left')
        player_data.drop([col for col in player_data.columns if '_y' in col],axis=1,inplace=True)
        del player_data[player_data.columns[32]]

        def removesuffix(data):
            """
            remove suffix of _x after joins
            :param data:
            :return:
            """
            colnames = []
            for col in data.columns:
                if "_x" in col:
                    col = col[:-2]
                    colnames.append(col)
                else:
                    colnames.append(col)
            data.columns = colnames

        removesuffix(player_data)

        player_data = player_data.merge(advanced,on=["Player","Year"],how='left')
        player_data = player_data.dropna(how='all',axis=1)
        del player_data[player_data.columns[60]]
        player_data.drop([col for col in player_data.columns if '_y' in col], axis=1, inplace=True)
        removesuffix(player_data)

        player_data = player_data[player_data["G"]>=20]
        player_data = player_data.merge(adjustedshot,on=["Player","Year"],how="left")
        removesuffix(player_data)
        player_data.drop([col for col in player_data.columns if '_y' in col],axis=1,inplace=True)
        player_data = player_data.drop("Rk",axis=1)
        player_data = player_data.drop("Team",axis=1)

        return player_data

    def figures(self):
        """
        Create figures fore descriptive statistics
        :return:None
        """
        player_data = self.processdata()
        advancedvariables = player_data[["TS%",'3PAr','FTr','TRB%','AST%','STL%','BLK%','TOV%','USG%','WS','ORtg','BPM','VORP','TS%']]
        allstar = player_data[player_data["All Star"]==1]

        #Correlation heatmap for advanced statistics
        corradvanced = advancedvariables.corr()
        sns.heatmap(corradvanced,
                    xticklabels=corradvanced.columns,
                    yticklabels=corradvanced.columns)
        pyplot.show()

        #Histogram of distribution of All- Stars vs Non All-Stars
        allstar.value_counts().plot(kind='bar')
        pyplot.xlabel("All Star")
        pyplot.show()

        # Plot of "FTr","3PAr","TS%" metrics from 1979-80 to 2020-21 NBA Seasons to visualize change in playing style over years

        threep = pd.DataFrame(player_data.groupby(['Year'],as_index=False).mean().groupby('Year')["3PAr"].mean())
        threep=threep.reset_index()
        ftr = pd.DataFrame(player_data.groupby(['Year'],as_index=False).mean().groupby('Year')["FTr"].mean())
        ftr = ftr.reset_index()
        ws = pd.DataFrame(player_data.groupby(['Year'], as_index=False).mean().groupby('Year')["WS"].mean())
        ws = ws.reset_index()
        vorp = pd.DataFrame(player_data.groupby(['Year'], as_index=False).mean().groupby('Year')["VORP"].mean())
        vorp = vorp.reset_index()
        ts = pd.DataFrame(player_data.groupby(['Year'], as_index=False).mean().groupby('Year')["TS%"].mean())
        ts = ts.reset_index()
        bpm = pd.DataFrame(player_data.groupby(['Year'], as_index=False).mean().groupby('Year')["BPM"].mean())
        bpm = bpm.reset_index()
        ortg = pd.DataFrame(player_data.groupby(['Year'], as_index=False).mean().groupby('Year')["ORtg"].mean())
        ortg = ortg.reset_index()


        years = list(ftr["Year"])
        cleanyear=[]
        for year in years:
            year = int(year[0:4])
            cleanyear.append(year)
        ftr=list(ftr["FTr"])
        threep = list(threep["3PAr"])
        ws = list(ws["WS"])
        vorp = list(vorp["VORP"])
        ts = list(ts["TS%"])
        bpm = list(bpm["BPM"])
        ortg = list(ortg["ORtg"])
        cleanortg = []
        for i in ortg:
            ortg = i/100
            cleanortg.append(ortg)
        print(ortg)
        ortg=cleanortg
        pyplot.plot(cleanyear,ftr,cleanyear,threep,cleanyear,ts)
        pyplot.legend(["FTr","3PAr","TS%"])
        pyplot.show()

        #Plot of "WS","VORP","BPM","ORtg" metrics from 1979-80 to 2020-21 NBA Season
        pyplot.plot(cleanyear, ws,cleanyear,vorp,cleanyear,bpm,cleanyear,ortg)
        pyplot.legend(["WS","VORP","BPM","ORtg"])
        pyplot.show()
        pd.set_option("display.max_columns", 100)
        fouradvanced = player_data[['3PAr','FTr','TRB%','AST%','STL%','BLK%','TOV%','USG%','WS','ORtg','BPM','WS','All Star','PER','TS%','VORP']]
        corr = fouradvanced.corr()
        print(corr)

    def RFE(self):
        """
        Creating rankings of all advanced metrics being considered using Recursive Feature Elimination
        to choose best variables for my model
        :return: None
        """

        # Select advanced metrics
        cleaned_playerdata = self.processdata()
        baseadvml = RandomForestClassifier(random_state=13)
        advancedvariables = cleaned_playerdata[
            ["TS%", '3PAr', 'FTr', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'WS']]

        #Scale data
        scalar = StandardScaler()
        scalar.fit(advancedvariables)
        avscaled = scalar.transform(advancedvariables)
        advancednames = ["TS%", '3PAr', 'FTr', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'WS']
        y = cleaned_playerdata["All Star"]

        # Run RFE based on advanced metrics
        rfeadv = RFE(baseadvml)
        fitadv = rfeadv.fit(avscaled,y)

        # Print rankings
        print("Num Features: %d" % fitadv.n_features_)
        print("Selected Features: %s" % fitadv.support_)
        print("Feature Ranking: %s" % fitadv.ranking_)
        rankings={}
        rankings1={}
        for i in range(len(fitadv.ranking_)):
            rankings.update({advancednames[i]:fitadv.ranking_[i]})
        for i in range(len(fitadv.support_)):
            rankings1.update({advancednames[i]:fitadv.support_[i]})

        print(rankings1)
        print(rankings)

    def randomforestmodel(self):
        """
        Carry out GridSearch with a dictionary of parameters for  RandomForestClassifier() for hyperparameter tuning.
        Build and fit model based off of RFE and GridSearch
        Print cross-validation scores
        Create learning curves for training and validation data
        :return:
        ypred_train: prediction of All Star from training set
        ypred_test:prediction of All Star from test set
        ytrain: All star variable training data
        ytest: All star variable for test data
        xtrain: advanced metrics training data
        advancedvariablenames: names of advanced variables selected for random forest classifier
        xtest: advanced metrics test data
        """
        # Select advanced metrics chosen from RFE rankings
        cleaned_playerdata = self.processdata()
        advancedvariables = cleaned_playerdata[["WS","USG%","TOV%","BLK%","TS%"]]
        advancedvariablenames = ["WS","USG%","TOV%","BLK%","TS%"]
        y = cleaned_playerdata["All Star"]

        # Scale data
        scalar = StandardScaler()
        scalar.fit(advancedvariables)
        advparameters = scalar.transform(advancedvariables)

        #GridSearch(takes a while using class_weight, adjust based on parameters and CPU strength)
        weights = np.linspace(1, 50, 50)
        weight_list = [{0: 1, 1: x} for x in weights]
        model_params = {
            'n_estimators': [50, 100,150,200,250],
            'max_features': ['sqrt',0.25, 0.5, 0.75, 1.0],
            'min_samples_split': [2, 4, 6,8,10],
            'class_weight':weight_list
        }

        #GridSearch and print best hyperparameters
        # basemodel = RandomForestClassifier(random_state=13)
        # grid = GSCV(estimator=basemodel,param_grid=model_params,scoring='f1')
        # grid.fit(advparameters,y)
        # results = grid.cv_results_
        # print("Best Score:" + str(grid.best_score_))
        # print("Best Parameters: " + str(grid.best_params_))

        # Splitting data into test and training data
        xtrain, xtest, ytrain, ytest = train_test_split(advparameters, y, random_state=13, test_size=0.20)
        print(y.value_counts())
        print(len(xtest))
        print(len(xtrain))

        #Fitting model and predicting test data
        modelrefit = RandomForestClassifier(random_state=13,max_features=0.75,min_samples_split=10,n_estimators=150)
        modelrefit.fit(advparameters,y)
        ypred_train = modelrefit.predict(xtrain)
        ypred_test = modelrefit.predict(xtest)


        # Cross-validation scores,classification report and confusion matrix on test data
        print("Accuracy Score for training data:")
        print(met.accuracy_score(ytrain, ypred_train))
        print("Accuracy Score for test data")
        print(met.accuracy_score(ytest, ypred_test))

        #Confusion matrix
        # sns.heatmap((met.confusion_matrix(ytest, ypred_test)),annot=True,fmt='.1f',cmap="Blues")
        #pyplot.show()

        # Printing the classification report and other metrics for test/training data
        print("Classification Report for test data")
        print(met.classification_report(ytest, ypred_test))
        print("ROC/AUC Score Training Data: ")
        print(met.roc_auc_score(ytrain, ypred_train))
        print("ROC/AUC Score Test Data: ")
        print(met.roc_auc_score(ytest, ypred_test))

        # Fitting model and predicting full data
        ypred = modelrefit.predict(advparameters)

        # Cross-validation scores,classification report and confusion matrix on full data
        print("Accuracy Score for full data:")
        print(met.accuracy_score(y, ypred))
        print("Classification Report for full data")
        print(met.classification_report(y, ypred))

        # Confusion matrix for full data
        sns.heatmap((met.confusion_matrix(y, ypred)), annot=True, fmt='.1f', cmap="Blues")
        pyplot.show()
        print("ROC/AUC Score for full data: ")
        print(met.roc_auc_score(y, ypred))

        # Making learning curve with unweighted model
        train_sizes = [1,100,500,1000,2000,5000,8000,10000,12876]
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=RandomForestClassifier(random_state=13,max_features=0.75,min_samples_split=10,n_estimators=150),
            X=advancedvariables,
            y=y, train_sizes=train_sizes,
            scoring='neg_mean_squared_error')
        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        pyplot.style.use('seaborn')
        pyplot.plot(train_sizes, train_scores_mean, label='Training error')
        pyplot.plot(train_sizes, validation_scores_mean, label='Validation error')
        pyplot.ylabel('MSE', fontsize=14)
        pyplot.xlabel('Training set size', fontsize=14)
        pyplot.title('Learning curves for WS,USG%,TOV%,BLK%,TS% ', fontsize=18, y=1.03)
        pyplot.legend()
        pyplot.ylim(0, 0.2)
        pyplot.show()

        # Making learning curve with weighted model
        train_sizes, train_scoresw, validation_scoresw = learning_curve(
            estimator=RandomForestClassifier(random_state=13, max_features=0.75, min_samples_split=10,class_weight={0: 1, 1: 2},
                                             n_estimators=150),
            X=advancedvariables,
            y=y, train_sizes=train_sizes,
            scoring='neg_mean_squared_error')
        train_scores_meanw = -train_scoresw.mean(axis=1)
        validation_scores_meanw = -validation_scoresw.mean(axis=1)

        pyplot.style.use('seaborn')
        pyplot.plot(train_sizes, train_scores_meanw, label='Training error with weights')
        pyplot.plot(train_sizes, validation_scores_meanw, label='Validation error with weights')
        pyplot.ylabel('MSE', fontsize=14)
        pyplot.xlabel('Training set size', fontsize=14)
        pyplot.title('Learning curves for WS,USG%,TOV%,BLK%,TS%,STL% weighted ', fontsize=18, y=1.03)
        pyplot.legend()
        pyplot.ylim(0,0.2)
        pyplot.show()

        return ypred_train,ypred_test,ytrain,ytest,xtrain,advancedvariablenames,xtest


    def recallmodel(self):
        """
        create recall precision model highlighting optimal threshold
        :return:
        """
        playerdata = self.processdata()
        y = playerdata["All Star"]
        ytest = self.randomforestmodel()[3]
        probs = self.randomforestmodel()[5]
        prec,recall,thresholds = met.precision_recall_curve(ytest,probs)

        #F-Measure = (2 * Precision * Recall) / (Precision + Recall). Pyplot of Precision recall curve
        f=(2*prec*recall)/(prec+recall)
        optimalindex = np.argmax(f)
        pyplot.scatter(recall[optimalindex],prec[optimalindex],marker='o',color="red",label="Optimal Threshold",zorder=5)
        pyplot.plot(recall, prec, marker='.', color='black',label='Model Recall-Precision curve',zorder=0)
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.legend()
        pyplot.show()

        #adjusting model according to threshold
        optimalthreshold = f[optimalindex]
        adjpredictions = np.where(probs>optimalthreshold,1,0)
        classrep = pd.DataFrame(data=[met.accuracy_score(ytest, adjpredictions), met.recall_score(ytest, adjpredictions),
                          met.precision_score(ytest, adjpredictions), met.roc_auc_score(ytest, adjpredictions)],
                    index=["accuracy", "recall", "precision", "roc_auc_score"])
        print(classrep)
        return optimalthreshold

if __name__ =="__main__":
    nbadatapath = '~/Documents/GWU DATA ANALYTICS /EMSE 6574/NBA DATA/'
    regmodel = buildingmodel(nbadatapath)
    regmodel.figures()
    regmodel.randomforestmodel()
    regmodel.recallmodel()


