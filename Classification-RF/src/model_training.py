from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class RandomForestModel:

    def ml_model(self):
        # Random Forest Classifier
        model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt', random_state=21)
        return model

    def training(self, train_df, test_df, yTest, model):
        # There is two NaN values which we are dropping
        train_df.dropna(inplace= True)
        test_df.dropna(inplace= True)
        # Split the dataset into dependant and independent variables
        xTrain = train_df.iloc[:, 1:9]
        yTrain = train_df.iloc[:, 0]
        xTest = test_df
        yTest = yTest.iloc[:, 1]

        # Train the model
        model.fit(xTrain, yTrain)
        y_predict = model.predict(xTest)
        y_pred_prob = model.predict_proba(xTest)[:,1]

        return yTest, y_predict, y_pred_prob

    def evaluation_metrics(self, yTest, y_predict, y_pred_prob):
        # Perform all the evaluation metrics
        accuracy = metrics.accuracy_score(yTest, y_predict)*100
        balanced_accuracy = metrics.balanced_accuracy_score(yTest, y_predict)*100
        recall = metrics.recall_score(yTest, y_predict)
        auc_roc = metrics.roc_auc_score(yTest, y_pred_prob)
        f1_score = metrics.f1_score(yTest, y_predict)
        confusion_matrix = metrics.confusion_matrix(yTest, y_predict)
        tn, fp, fn, tp = metrics.confusion_matrix(yTest, y_predict).ravel()
        specificity = tn/(tn+fp)

        print('Accuracy: {0:0.2f}'.format(accuracy))
        print('Balanced Accuracy: {0:0.2f}'.format(balanced_accuracy))
        print('Sensitivity (TPR): {0:0.2f}'.format(recall))
        print('Specificity (TNR): {0:0.2f}'.format(specificity))
        print('AUC metric: {0:0.2f}'.format(auc_roc))
        print('F1-score: {0:0.2f}'.format(f1_score))
        print('Confusion Matrix: \n', confusion_matrix)
        return accuracy
            
