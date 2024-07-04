import sklearn
import pandas as pd
import numpy as np
import shap
import h2o
from h2o.automl import H2OAutoML
#h2o.init(nthreads=-1)
class H2OProbWrapper:
    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names

    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        self.dataframe= pd.DataFrame(X, columns=self.feature_names)
        self.predictions = self.h2o_model.predict(h2o.H2OFrame(self.dataframe)).as_data_frame().values
        return self.predictions.astype('float64')[:,-1] #probability of True class
class StartProcessAutoML:
    def startDataPreprocess(self):
        h2o.init(nthreads=-1) ### Start the h20 Server
        X, y = shap.datasets.adult()
        X_display, y_display = shap.datasets.adult(display=True)
        #X_display.to_csv("output.csv", index=False)
        #print(y_display.shape)
        print(X.head())
        #print(X_display.dtypes)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                                    test_size=0.2, random_state=7)

        train_indices = X_train.index
        test_indices = X_test.index

        X_train_display = X_display.iloc[train_indices]
        y_train_display = y_display[train_indices]
        X_test_display = X_display.iloc[test_indices]
        y_test_display = y_display[test_indices]

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        X_train_display.reset_index(drop=True, inplace=True)
        X_test_display.reset_index(drop=True, inplace=True)

        train_h2o_df = h2o.H2OFrame(X_train)
        train_h2o_df['labels'] = h2o.H2OFrame(y_train)
        train_h2o_df['labels'] = train_h2o_df['labels'].asfactor()
        train_h2o_df['Workclass'] = train_h2o_df['Workclass'].asfactor()
        train_h2o_df['Marital Status'] = train_h2o_df['Marital Status'].asfactor()
        train_h2o_df['Relationship'] = train_h2o_df['Relationship'].asfactor()
        train_h2o_df['Occupation'] = train_h2o_df['Occupation'].asfactor()
        train_h2o_df['Sex'] = train_h2o_df['Sex'].asfactor()
        train_h2o_df['Race'] = train_h2o_df['Race'].asfactor()
        train_h2o_df['Country'] = train_h2o_df['Country'].asfactor()

        test_h2o_df = h2o.H2OFrame(X_test)
        test_h2o_df['labels'] = h2o.H2OFrame(y_test)
        test_h2o_df['labels'] = test_h2o_df['labels'].asfactor()
        test_h2o_df['Workclass'] = test_h2o_df['Workclass'].asfactor()
        test_h2o_df['Marital Status'] = test_h2o_df['Marital Status'].asfactor()
        test_h2o_df['Relationship'] = test_h2o_df['Relationship'].asfactor()
        test_h2o_df['Occupation'] = test_h2o_df['Occupation'].asfactor()
        test_h2o_df['Sex'] = test_h2o_df['Sex'].asfactor()
        test_h2o_df['Race'] = test_h2o_df['Race'].asfactor()
        test_h2o_df['Country'] = test_h2o_df['Country'].asfactor()

        feature_names = list(X_train.columns)

        aml = H2OAutoML(max_runtime_secs=50, seed=2)
        #aml = H2OAutoML(max_runtime_secs=500, seed=42)
        aml.train(x=feature_names, y='labels', training_frame=train_h2o_df)

        lb = aml.leaderboard

        print(lb)
        bst_model = aml.leader

        h2o_wrapper = H2OProbWrapper(bst_model, feature_names)

        X_train.shape[0]

        explainer = shap.KernelExplainer(h2o_wrapper.predict_binary_prob, X_train.iloc[:100, :])

        person = 0  # first person in test dataset

        print('prediction (probability that this person earns more than $50k/year) =',
              h2o_wrapper.predict_binary_prob(X_test.iloc[person])[0])
        print('ground_truth (this person earns more than $50k/year) =', y_test_display[person])

        shap.initjs()
        shap_values = explainer.shap_values(X_test.iloc[person, :], nsamples=500)
        shap.force_plot(explainer.expected_value, shap_values, X_test_display.iloc[person, :])

        person = 1  # second person in test dataset

        print('prediction (probability that this person earns more than $50k/year) =',
              h2o_wrapper.predict_binary_prob(X_test.iloc[person])[0])
        print('ground_truth (this person earns more than $50k/year) =', y_test_display[person])

        
        shap.initjs()
        shap_values = explainer.shap_values(X_test.iloc[person, :], nsamples=500)
        shap.force_plot(explainer.expected_value, shap_values, X_test_display.iloc[person, :])
        '''
        h2o.save_model(bst_model)
        X_test.to_pickle('X_test.pkl')
        X_train.to_pickle('X_train.pkl')
        np.save('y_test.npy', y_test)
        np.save('y_train.npy', y_train)
        X_test_display.to_pickle('X_test_display.pkl')
        X_train_display.to_pickle('X_train_display.pkl')
        np.save('y_test_display.npy', y_test_display)
        np.save('y_train_display.npy', y_train_display)
        X.to_pickle('X.pkl')'''

        #h2o.cluster().shutdown()

        return lb



