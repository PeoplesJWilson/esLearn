from esLearn.additive_models_and_trees import RegressionTree
from random import sample, choices
from multiprocessing import cpu_count, Pool
from numpy import zeros
from math import ceil
from functools import reduce

def make_batches(data, num_batches):
        batch_size = ceil(len(data) / num_batches)
        return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    
def map_reduce(data, num_processes, mapper, reducer):
        batches = make_batches(data, num_processes)
        pool = Pool(num_processes)                       
        batch_results = pool.map(mapper, batches)          
        return reduce(reducer, batch_results)     

def mapper(forest_chunk):
    trained_chunk = []        
    for (X_bootstrap, y_bootstrap, feature_subset, tree) in forest_chunk:
        tree.fit(X_bootstrap, y_bootstrap)
        trained_chunk.append((feature_subset, tree))  
    return trained_chunk
        
def reducer(forest_1, forest_2):
    return forest_1 + forest_2



class RandomForestRegressor(RegressionTree):
    def __init__(self, num_estimators = 50, max_features = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_estimators = num_estimators
        self.max_features = max_features
        self._representation = []
    

    def fit(self, X,y):
        N = X.shape[0]
        features = range(X.shape[1])
        samples = range(X.shape[0])

        forest = [(sample(features, self.max_features), RegressionTree(self.min_samples_split, self.min_samples_leaf)) for _ in range(self.num_estimators)]      

        training_forest = []
        for i,(feature_subset, tree) in enumerate(forest):
            X_selected_features = X[:, feature_subset]

            bootstrap_samples = choices(samples, k=N)

            X_bootstrap, y_bootstrap = X_selected_features[bootstrap_samples, :], y[bootstrap_samples]

            training_forest.append((X_bootstrap, y_bootstrap, feature_subset, tree))
  
        num_proccesses = max(cpu_count() - 1, 1)
        forest = map_reduce(training_forest, num_proccesses, mapper, reducer)

        self._representation = forest
    
    def rank_estimators(self, X, y):
        if len(self._representation) == 0:
            error_msg = "Error: must call fit before calling predict"
            print(error_msg)
            return error_msg
        
        self._representation = sorted(self._representation, reverse = True, key = lambda tree: ((y - tree[1].predict(X[:,tree[0]]))**2).sum())
              


    
    def predict(self, X):
        N = X.shape[0]
        if len(self._representation) == 0:
            error_msg = "Error: must call fit before calling predict"
            print(error_msg)
            return error_msg

        preds =  zeros((N, self.num_estimators))

        for i, (feature_subset, tree) in enumerate(self._representation):
            preds[:,i] = tree.predict(X[:,feature_subset])

        return preds.mean(axis = 1)
    





     
            



    