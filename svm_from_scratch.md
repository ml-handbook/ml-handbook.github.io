# Support Vector Machine (SVM) From Scratch

This tutorial walks through building a **Support Vector Machine (SVM)** classifier **from scratch in Python**, without using `scikit-learn`'s SVM implementation.

We will:

* Understand the math behind SVMs
* Derive the optimization objective
* Implement a **linear SVM** using **gradient descent**
* Train and test it on a toy dataset

This article is **implementation-first** and suitable for ML learners who want to truly understand how SVMs work internally.

---

## 1. What Is a Support Vector Machine?

A Support Vector Machine is a supervised learning algorithm used for **classification** and **regression**.

For **binary classification**, SVM finds a **hyperplane** that:

* Separates the data into two classes
* Maximizes the **margin** (distance to the closest data points)

Those closest points are called **support vectors**.

---

## 2. Linear SVM Model

For a linear SVM, the decision function is:

[ f(x) = w^T x + b ]

Where:

* ( w ) → weight vector
* ( b ) → bias term

Prediction:

[
\hat{y} = \text{sign}(w^T x + b)
]

---

## 3. Margin and Constraints

For labeled data ( (x_i, y_i) ), where ( y_i \in {-1, +1} ):

Hard-margin constraint:

[
y_i (w^T x_i + b) \ge 1
]

This ensures:

* Correct classification
* Points are outside the margin

---

## 4. Soft-Margin SVM (Practical Version)

Real data is rarely perfectly separable. We introduce **hinge loss** and **regularization**.

### Optimization Objective

[
\min_{w,b} ; \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b))
]

Where:

* First term → margin maximization
* Second term → hinge loss
* ( C ) → regularization strength

---

## 5. Gradient Descent Updates

### Hinge Loss Gradient

If:

[
y_i (w^T x_i + b) \ge 1
]

Then:

* No loss
* Gradient = 0

Else:

[
\frac{\partial L}{\partial w} = w - C y_i x_i
]
[
\frac{\partial L}{\partial b} = -C y_i
]

---

## 6. Implementing SVM From Scratch

### Step 1: Imports

```python
import numpy as np
```

---

### Step 2: SVM Class

```python
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)
```

---

## 7. Training on a Toy Dataset

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = LinearSVM()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
```

---

## 8. Evaluating Accuracy

```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("Accuracy:", accuracy(y_test, predictions))
```

---

## 9. Visualizing the Decision Boundary (Optional)

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

plot_decision_boundary(X_train, y_train, svm)
```

---

## 10. Key Takeaways

* SVM maximizes the margin between classes
* Hinge loss penalizes margin violations
* Regularization controls overfitting
* Building SVM from scratch demystifies kernel methods

---

## 11. Where to Go Next

* Add **kernel functions** (RBF, polynomial)
* Implement **dual formulation**
* Compare with `sklearn.svm.SVC`
* Extend to **multiclass SVM**

---

📘 *This article is ready to publish on* **ml-handbook.github.io**
