from functools import partial
from numpy import unique, array

class RegressionNode:
    def __init__(self, val = None, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

    def isLeaf(self,):
        return (self.left == None) and (self.right == None)

# Model
class RegressionTree:

    def __init__(self, min_samples_split = 5, min_samples_leaf = 1):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._representation = None
        self.num_leaves = 0

# PUBLIC METHODS
#     
    def fit(self, X, y):
        self._representation = self.__grow(X,y)

    def predict(self, X):
        if self._representation is None:
            error_msg = "Error: must call fit before calling predict."
            print(error_msg)
            return error_msg

        traverse = partial(self.__predict_singleton, self._representation)

        preds = list(map(traverse, list(X)))

        return array(preds)
    
    def display(self,):
        if self._representation is None:
            error_msg = "Error: must call fit before calling predict."
            print(error_msg)
            return error_msg

        self.__display_model(self._representation)


# PRIVATE METHODS

# predict helper
    def __predict_singleton(self, current_node, x):

        if current_node.isLeaf():
            return current_node.val

        (j, s) = current_node.val
        
        if x[j] >= s:
            return self.__predict_singleton(current_node.left, x)
        else:
            return self.__predict_singleton(current_node.right, x)
     
# display helper
    def __display_model(self, node, level=0):
        if node is None:
            pass

        elif node.isLeaf():
            print(' ' * 10 * level + '-> ' + f"{round(node.val, 2)}")

        elif ~node.isLeaf():
            self.__display_model(node.left, level + 1)
            print(' ' * 10 * level + '-> ' + f"x_{node.val[0]}<{round(node.val[1], 2)}")
            self.__display_model(node.right, level + 1)
              
# fitting helpers
    def __node_loss(self, j, s, X, y):
        mask = (X[:,j] >= s)
        y_left, y_right = y[mask], y[~mask]

        if (mask.sum() == 0) or ((~mask).sum() == 0):
            return (float('inf'), None, None, mask)

        pred_left, pred_right = y_left.mean(), y_right.mean()
        
        left_loss, right_loss = (y_left - pred_left)**2, (y_right - pred_right)**2

        return (left_loss.sum() + right_loss.sum(), pred_left, pred_right, mask)

    def __get_best_split(self, X,y):
        start = self.__node_loss(0, X[0,0], X, y)
        min_loss = (0,0, start[0], start[1], start[2], start[3])
        for j in range(X.shape[1]):
            col = X[:,j]
            vals = unique(col)
            for s in vals:
                (loss, left, right, mask) = self.__node_loss(j, s, X, y)
                if loss < min_loss[2]:
                    min_loss = (j, s, loss, left, right, mask)
        
        return min_loss

    def __grow(self, X, y):

        n_samples = X.shape[0]

        if n_samples <= self.min_samples_split:
            self.num_leaves += 1
            return RegressionNode(val = y.mean())
        
        (j, s, loss, left, right, mask) = self.__get_best_split(X,y)
        
        if (mask.sum() <= self.min_samples_leaf) or ((~mask).sum() <= self.min_samples_leaf):
            self.num_leaves += 1
            return RegressionNode(val = y.mean())
            
        
        else:
            return RegressionNode(val=(j,s), 
                            left=self.__grow(X[mask, :], y[mask]),
                            right=self.__grow(X[~mask, :], y[~mask]))