#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>
#include <cmath>

// [[Rcpp::export]]
void forward_loop_cpp(
    const arma::mat& x,
    const arma::mat& R,
    const arma::mat& xx,
    const arma::vec& ini_sel,
    Rcpp::NumericMatrix G,
    Rcpp::List Rho,
    const double logalpha,
    const bool use_gram
) {
  const int n = static_cast<int>(x.n_rows);
  const int d = static_cast<int>(x.n_cols);

  // Empirical cross-products, computed only once.
  // Used when the supplied R is not the empirical correlation matrix.
  arma::mat C;

  if (!use_gram) {
    C = x.t() * x;
  }

  std::vector<int> selected;
  selected.reserve(d);

  std::vector<unsigned char> available(d);

  std::vector<double> rho;
  rho.reserve(d + 1);

  arma::vec beta;
  arma::mat A;
  arma::vec b;
  arma::vec correlations(d);

  for (int j = 0; j < d; ++j) {

    selected.clear();
    rho.clear();

    std::fill(available.begin(), available.end(), 1);

    // Initial predictor.
    const int first = static_cast<int>(ini_sel[j]) - 1;

    selected.push_back(first);
    available[j] = 0;
    available[first] = 0;

    const arma::vec y = x.col(j);

    // Initial residual, matching the original implementation.
    arma::vec res = y - x.col(first) * R(first, j);
    double sse2 = arma::dot(res, res);

    const double df0 = n - 2.0;
    const double stat0 =
      (n - 1.0 - sse2) / sse2 * df0;

    rho.push_back(0.0);
    rho.push_back(
      R::pf(stat0, 1.0, df0, false, true)
    );

    // Initialize predictor-residual correlations.
    if (use_gram) {
      for (int a = 0; a < d; ++a) {
        correlations[a] =
          xx(a, j) - xx(a, first) * R(first, j);
      }
    } else {
      correlations =
        C.col(j) - C.col(first) * R(first, j);
    }

    int i = 2;

    while (rho[i - 1] < logalpha && i < n) {

      ++i;

      // Select the available predictor with the largest
      // absolute correlation with the current residual.
      int best = -1;
      double best_abs = -1.0;

      for (int a = 0; a < d; ++a) {

        if (available[a]) {

          const double abs_value =
            std::abs(correlations[a]);

          if (abs_value > best_abs) {
            best_abs = abs_value;
            best = a;
          }
        }
      }

      if (best < 0) {
        break;
      }

      selected.push_back(best);
      available[best] = 0;

      const int p = static_cast<int>(selected.size());
      const double sse1 = sse2;

      // Construct regression system using xx.
      A.set_size(p, p);
      b.set_size(p);

      for (int a = 0; a < p; ++a) {

        const int sa = selected[a];

        b[a] = xx(sa, j);

        for (int c = 0; c < p; ++c) {
          A(a, c) = xx(sa, selected[c]);
        }
      }

      // Solve the regression system.
      const bool solved = arma::solve(
        beta,
        A,
        b,
        arma::solve_opts::no_approx
      );

      if (!solved || !beta.is_finite()) {

        // Preserve the original fallback behavior.
        rho.push_back(0.0);

      } else {

        if (use_gram) {

          // Fast path: xx = crossprod(x).
          sse2 = xx(j, j) - arma::dot(beta, b);

          for (int a = 0; a < d; ++a) {

            double value = xx(a, j);

            for (int c = 0; c < p; ++c) {
              value -=
                xx(a, selected[c]) * beta[c];
            }

            correlations[a] = value;
          }

        } else {

          // Non-Gram path: use empirical cross-products
          // for residual SSE and predictor correlations.
          arma::uvec idx(p);

          for (int a = 0; a < p; ++a) {
            idx[a] =
              static_cast<arma::uword>(selected[a]);
          }

          const arma::vec C_sj = C.submat(
            idx,
            arma::uvec{static_cast<arma::uword>(j)}
          );

          const arma::mat C_ss = C.submat(idx, idx);

          sse2 =
            C(j, j)
            - 2.0 * arma::dot(beta, C_sj)
            + arma::as_scalar(beta.t() * C_ss * beta);

            // Update correlations for all predictors.
            correlations =
            C.col(j) - C.cols(idx) * beta;
        }

        const double df = n - p - 1.0;

        const double stat =
          (sse1 - sse2) / sse2 * df;

        rho.push_back(
          R::pf(stat, 1.0, df, false, true)
        );
      }
    }

    // Store p-values and predictor names.
    const int len = static_cast<int>(selected.size());

    Rcpp::NumericVector rho_out(len);
    Rcpp::CharacterVector rho_names(len);

    for (int a = 0; a < len; ++a) {
      rho_out[a] = rho[a];
    }

    rho_names[0] = "0";

    for (int a = 1; a < len; ++a) {
      rho_names[a] =
        std::to_string(selected[a - 1] + 1);
    }

    rho_out.attr("names") = rho_names;
    Rho[j] = rho_out;

    // Record edges, matching the original implementation.
    for (int a = 0; a < len - 1; ++a) {
      G(j, selected[a]) = 1.0;
    }
  }
}
