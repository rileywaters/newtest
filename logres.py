import os
from numpy import ndarray, array, exp, log, diag, identity, zeros, load, array, save
from numpy.linalg import norm, eigh
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV

def get_scikit_lamda(X, Y):
    """
    Gets the optimal lambda as determined by scikit-learn's LogisticRegressionCV
    """
    logres = LogisticRegressionCV(fit_intercept=False).fit(X, Y)
    lambda_opt = 1/logres.C_
    return lambda_opt

def obj(X, Y, lam, beta):
    n= X.shape[0]
    return 1/n * sum(log(1 + exp(-Y * (X @ beta)))) + lam*(beta.T @ beta)

def computegrad(X, Y, lam, beta):
    n = X.shape[0]
    p_vector = 1/(1+exp(-Y * (X @ beta)))
    P = identity(n) - diag(p_vector)
    return (2*lam*beta) - (1/n * X.T @ P @ Y)

def backtracking(X, Y, lam, betas, grad_betas, init_t=1, alpha=0.5, beta=0.5, max_iter=1000):
    t = init_t
    norm_grad_betas = norm(grad_betas)
    for i in range(max_iter):
        if obj(X, Y, lam, betas - t*grad_betas) >= (obj(X, Y, lam, betas) - alpha*t*(norm_grad_betas**2)):
            t *= beta
        else:
            return t
    print('Max iterations of backtracking reached')
    return t

def fastgrad(X, Y, lam, epsilon, max_iter=100):

    n = X.shape[0]
    p = X.shape[1]

    # Determine the best initial step size
    eigenvalues, eigenvectors = eigh(1/n * X.T @ X)
    lipschitz =  max(eigenvalues) + lam
    init_stepsize = 1/lipschitz

    beta = zeros(p)
    theta = zeros(p)

    beta_hist = [beta]


    i = 0
    t = init_stepsize
    grad_beta = computegrad(X, Y, lam, beta)
    while norm(grad_beta) > epsilon:
        if i > max_iter:
            raise ValueError("Maximum iterations reached")
        t = backtracking(X, Y, lam, beta, grad_beta, init_t=t)
        beta_new = theta - t*_grad(X, Y, lam, theta)
        theta = beta_new + i/(i+3)*(beta_new - beta)
        beta = beta_new
        grad_beta = computegrad(X, Y, lam, beta_new)

        beta_hist.append(beta)


        i += 1
    return beta, beta_hist

def compute_misclass_error(beta_opt, X, Y):
    y_pred = expit(X @ beta_opt)
    return np.mean(y_pred != Y)

def plot_misclass_error(beta_hist, X_train, Y_train, X_test, Y_test):
    niter_fg = np.size(beta_hist, 0)
    error_train = np.zeros(niter_fg)
    error_test = np.zeros(niter_fg)

    for i in range(niter_fg):
        error_train[i] = compute_misclass_error(beta_hist[i, :], X_train, Y_train)
        error_test[i] = compute_misclass_error(beta_hist[i, :], X_test, Y_test)

    fig, ax = plt.subplots()
    ax.plot(range(1, niter_fg + 1), error_train, c='red', label='Training Set')
    ax.plot(range(1, niter_fg + 1), error_test, c='blue', label='Validation Set')

    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    ax.legend(loc='upper right')

# Load the data
data_dir = 'competition2data'
X_train = load(os.path.join(data_dir, 'train_features.npy'))
Y_train = load(os.path.join(data_dir, 'train_labels.npy'))
X_test = load(os.path.join(data_dir, 'val_features.npy'))
Y_test = load(os.path.join(data_dir, 'val_labels.npy'))
kaggle_X_test = load(os.path.join(data_dir, 'test_features.npy'))


# Find optimal lambda with scikit learn
lam_opt = get_scikit_lamda(X_train, Y_train)

# Train logistic model
beta, beta_hist = fastgrad(X_train, Y_train, lam_opt, 0.01)

plot_misclass_error(beta, X_train, Y_train, X_test, Y_test)

# Save predictions to submit to kaggle
kaggle_predictions = expit(kaggle_X_test @ beta)
pd.DataFrame(kaggle_predictions).to_csv("kaggle2submit.csv", header = ['Category'])
