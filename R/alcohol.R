#' Data sets for the alcohol dependence example
#'
#' A list of 3 data frames that contains the gene expression, 
#' DNA methylation and AUD (alcohol use disorder) of 46 human subjects. 
#' The data is already screened for quality control. 
#' For the raw data see the link below. For more details see the reference. 
#'
#' @format A list of 3 data frames:
#' \describe{
#'   \item{gene}{Human  gene expression. A data frame of 46 rows and 300 columns.}
#'   \item{meth}{Human DNA methylation. A data frame of 46 rows and 500 columns.}
#'   \item{disorder}{Human AUD indicator. A data frame of 46 rows and 1 column.
#'              The first 23 subjects are AUDs and the others are matched controls.}
#' }
#' 
#' @examples
#' ############## Alcohol dependence example ######################
#' data(alcohol)
#' gene <- scale(as.matrix(alcohol$gene))
#' meth <- scale(as.matrix(alcohol$meth))
#' disorder <- as.matrix(alcohol$disorder)
#' alcohol.X <- list(X1 = gene, X2 = meth)
#' \dontrun{
#'   foldid <- c(rep(1:5, 4), c(3,4,5), rep(1:5, 4), c(1,2,5))
#'   ##  table(foldid, disorder)
#'   ## there maybe warnings due to the glm refitting with small sample size
#'   alcohol.cvr <- CVR(disorder, alcohol.X, rankseq = 2, etaseq = 0.02, 
#'                      family = "b", penalty = "L1", foldid = foldid )
#'   plot(alcohol.cvr)
#'   plot(gene %*% alcohol.cvr$solution$W[[1]][, 1], meth %*% alcohol.cvr$solution$W[[2]][, 1])
#'   cor(gene %*% alcohol.cvr$solution$W[[1]], meth %*% alcohol.cvr$solution$W[[2]])
#' }
#' 
#' @references Chongliang Luo, Jin Liu, Dipak D. Dey and Kun Chen (2016) Canonical variate regression. 
#'               Biostatistics, doi: 10.1093/biostatistics/kxw001.
#' @source  Alcohol dependence: \url{http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49393}.
"alcohol"
