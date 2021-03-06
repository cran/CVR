# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @title Canonical Variate Regression.
#' 
#' @description  Perform canonical variate regression with a set of fixed tuning parameters.
#'  
#' @usage cvrsolver(Y, Xlist, rank, eta, Lam, family, Wini, penalty, opts) 
#' 
#' @param Y A response matrix. The response can be continuous, binary or Poisson. 
#' @param Xlist A list of covariate matrices. Cannot contain missing values.
#' @param rank Number of pairs of canonical variates.
#' @param eta Weight parameter between 0 and 1.  
#' @param Lam A vector of penalty parameters \eqn{\lambda} for regularizing the loading matrices 
#'         corresponding to the covariate matrices in \code{Xlist}.  
#' @param family Type of response. \code{"gaussian"} if Y is continuous, \code{"binomial"} if Y is binary, and \code{"poisson"} if Y is Poisson. 
#' @param Wini A list of initial loading matrices W's. It must be provided. See \code{cvr} and \code{scca} for using sCCA solution as the default.
#' @param penalty Type of penalty on W's. "GL1" for rowwise sparsity and 
#'                   "L1" for entrywise sparsity.  
#' @param opts A list of options for controlling the algorithm. Some of the options are: 
#'         
#'         \code{standardization}:  need to standardize the data? Default is TRUE.
#'                        
#'         \code{maxIters}:         maximum number of iterations allowed in the algorithm. The default is 300. 
#'         
#'         \code{tol}:              convergence criterion. Stop iteration if the relative change in the objective is less than \code{tol}.
#'         
#' @details CVR is used for extracting canonical variates and also predicting the response 
#'           for multiple sets of covariates (Xlist = list(X1, X2)) and response (Y). 
#'           The covariates can be, for instance, gene expression, SNPs or DNA methylation data. 
#'           The response can be, for instance, quantitative measurement or binary phenotype.            
#'           The criterion minimizes the objective function 
#'                                 
#'   \deqn{(\eta/2)\sum_{k < j} ||X_kW_k - X_jW_j||_F^2 + (1-\eta)\sum_{k} l_k(\alpha, \beta, Y,X_kW_k)
#'   + \sum_k \rho_k(\lambda_k, W_k),}{%
#'   (\eta/2) \Sigma_\{k<j\}||X_kW_k - X_jW_j||_F^2 + (1 - \eta) \Sigma_k l_k(\alpha, \beta, Y, X_kW_k) + \Sigma_k \rho_k(\lambda_k, W_k),}
#'      s.t. \eqn{W_k'X_k'X_kW_k = I_r,}    for  \eqn{k = 1, 2, \ldots, K}. 
#'      \eqn{l_k()} are general loss functions with intercept \eqn{\alpha} and coefficients \eqn{\beta}. \eqn{\eta} is the weight parameter and 
#'           \eqn{\lambda_k} are the regularization parameters. \eqn{r} is the rank, i.e. the number of canonical pairs.          
#'           By adjusting \eqn{\eta}, one can change the weight of the first correlation term and the second prediction term. 
#'           \eqn{\eta=0} is reduced rank regression and \eqn{\eta=1} is sparse CCA (with orthogonal constrained W's). By choosing appropriate \eqn{\lambda_k} 
#'           one can induce sparsity of \eqn{W_k}'s to select useful variables for predicting Y.                       
#'           \eqn{W_k}'s with \eqn{B_k}'s and (\eqn{\alpha, \beta}) are iterated using an ADMM algorithm. See the reference for details.
#'           
#' @return An object containing the following components
#'   \item{iter}{The number of iterations the algorithm takes.}
#' @return \item{W}{A list of fitted loading matrices.}
#' @return \item{B}{A list of fitted \eqn{B_k}'s.}
#' @return \item{Z}{A list of fitted \eqn{B_kW_k}'s.}
#' @return \item{alpha}{Fitted intercept term in the general loss term.}
#' @return \item{beta}{Fitted regression coefficients in the general loss term.}
#' @return \item{objvals}{A sequence of the objective values.}
#' @author Chongliang Luo, Kun Chen.
#' @references Chongliang Luo, Jin Liu, Dipak D. Dey and Kun Chen (2016) Canonical variate regression. 
#'             Biostatistics, doi: 10.1093/biostatistics/kxw001.
#' @examples ## see  SimulateCVR for simulation examples, see CVR for parameter tuning.
#' @seealso \code{\link{SimulateCVR}}, \code{\link{CVR}}.
#' @export
cvrsolver <- function(Y, Xlist, rank, eta, Lam, family, Wini, penalty, opts) {
    .Call('CVR_cvrsolver', PACKAGE = 'CVR', Y, Xlist, rank, eta, Lam, family, Wini, penalty, opts)
}

