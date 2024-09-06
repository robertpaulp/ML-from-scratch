import numpy as np

class EigenvalVectors(object):
    
    # Power Method -> Find the largest eigenvalue
    def power_method(matrix, x, tol=None, max_iter=100):
        """Parameters
        ----------
        x : numpy array
            Initial guess for the eigenvector
        tol : float, optional
            Tolerance for the eigenvalue (default is 1e-6)
        max_iter : int, optional
            Maximum number of iterations (default is 100)

        Returns
        -------
        float
            The largest eigenvalue
        numpy array
            The corresponding eigenvector"""
        lambda_prev = 0

        for i in range(max_iter):
            x = matrix @ x / np.linalg.norm(x)

            # Using the Rayleigh quotient
            lambda_ = (x.T @ matrix @ x) / (x.T @ x)

            if tol is not None and abs(lambda_ - lambda_prev) < tol:
                break
            lambda_prev = lambda_

        return lambda_, x

    # Inverse Power Method -> Find the smallest eigenvalue
    def inverse_power_method(matrix, x, tol=None, max_iter=100):
        """Parameters
        ----------
        x : numpy array
            Initial guess for the eigenvector
        tol : float (optional)
            Tolerance for the eigenvalue
        max_iter : int (optional)
            Maximum number of iterations

        Returns
        -------
        float       -> Eigenvalue
        numpy array -> Eigenvector"""
        inv_matrix = np.linalg.inv(matrix)
        lambda_prev = 0

        for i in range(max_iter):
            x = inv_matrix @ x / np.linalg.norm(x)

            # Using the Rayleigh quotient
            lambda_ = (x.T @ matrix @ x) / (x.T @ x)

            if tol is not None and abs(lambda_ - lambda_prev) < tol:
                break
            
            lambda_prev = lambda_

        return lambda_, x


    # Inverse Power Method with Shift
    # Plus Rayleigh Quotient Iteration
    def inverse_power_method_shift(matrix, x, u, tol=None, max_iter=100):
        """Parameters
        ----------
        x : numpy array
            Initial guess for the eigenvector
        u : float
            Shift value
        tol : float (optional)
            Tolerance for the eigenvalue
        max_iter : int (optional)
            Maximum number of iterations

        Returns
        -------
        float       -> Eigenvalue
        numpy array -> Eigenvector"""
        lambda_prev = 0

        for i in range(max_iter):
            shifted_matrix = matrix - u * np.eye(matrix.shape[0])
            inv_shifted_matrix = np.linalg.inv(shifted_matrix)

            x = inv_shifted_matrix @ x / np.linalg.norm(inv_shifted_matrix @ x)

            lambda_ = (x.T @ matrix @ x) / (x.T @ x)
            u = lambda_

            if tol is not None and abs(lambda_ - lambda_prev) < tol:
                break
            
            lambda_prev = lambda_

        return lambda_, x


# Other methods
## QR Method, Cholesky Decomposition, Schur decomposition

