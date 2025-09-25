//    cFuncs.h - array and mathematical operations C functions of fathon package
//    Copyright (C) 2019-  Stefano Bianchi
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy
#include <math.h>
#include <gsl/gsl_multifit.h>

/*****************************************************************************
 * Small helper: Solve a (p+1) x (p+1) linear system via naive Gauss elimination
 * M is row-major, b is length (p+1). On return, b[] contains the solution.
 *****************************************************************************/
static void solveLinearSystem(double *M, double *b, int p)
{
    // We solve in-place:
    //   for i in [0..p]:
    //     pivot M[i][i], scale row, eliminate below, then back-substitution
    for(int i = 0; i <= p; i++)
    {
        // Pivot
        double pivot = M[i*(p+1) + i];
        for(int c = i; c <= p; c++)
            M[i*(p+1) + c] /= pivot;
        b[i] /= pivot;

        // Eliminate below
        for(int r = i+1; r <= p; r++)
        {
            double mult = M[r*(p+1) + i];
            for(int c = i; c <= p; c++)
                M[r*(p+1) + c] -= mult * M[i*(p+1) + c];
            b[r] -= mult * b[i];
        }
    }
    // Back-substitution
    for(int i = p; i >= 0; i--)
    {
        for(int r = i-1; r >= 0; r--)
        {
            double mult = M[r*(p+1) + i];
            M[r*(p+1) + i] = 0.0;
            b[r] -= mult * b[i];
        }
    }
}

/*****************************************************************************
 * Compute x^k for small k quickly. In practice, one might use pow() or a
 * small lookup table. This is just a simple repeated multiply for demonstration.
 *****************************************************************************/
static inline double quickPow(double x, int k)
{
    double val = 1.0;
    for(int i = 0; i < k; i++)
        val *= x;
    return val;
}

static void computeSavitzkyGolayCoeffs(int windowSize, int polyOrder, double *coeffs)
{
    // For an odd window W, we typically center the polynomial
    // around index m = (W-1)/2. We solve the least squares problem
    // for a polynomial of order `polyOrder` that passes as close
    // as possible to the “unit impulse” at m, i.e., the row of the
    // pseudoinverse corresponding to derivative=0 at the center.

    // This code implements a direct approach:
    //   G = (X^T X)^{-1} X^T
    // where X[i,j] = i^j, for i from -m..m, j from 0..polyOrder
    // We then read row m of G to get the filter for smoothing.

    // Because the full symbolic method is quite involved, below is
    // a straightforward method that forms and inverts the normal
    // equation. For large window sizes, consider using a robust
    // linear algebra library.

    int W = windowSize;
    int d = polyOrder;
    int m = (W - 1) / 2; // center

    // Construct X: W x (d+1) matrix
    // row i => x_i^0, x_i^1, ..., x_i^d
    // x_i = (i - m)
    double *X = (double *)calloc(W * (d + 1), sizeof(double));
    for(int i = 0; i < W; i++)
    {
        double x = (double)(i - m);
        for(int j = 0; j <= d; j++)
        {
            // X[i, j]
            X[i * (d + 1) + j] = pow(x, j);
        }
    }

    // We want G = (X^T X)^{-1} X^T, then the center row of G is the filter.
    // Let's build normal equation N = X^T X (size (d+1) x (d+1)),
    // then solve for inv(N)*X^T. We'll only solve for the row that
    // corresponds to the “impulse at the center”, i.e. the center row in X^T.

    // N = (d+1) x (d+1)
    double *N = (double *)calloc((d + 1) * (d + 1), sizeof(double));
    // B = (d+1) x 1 (the right-hand side for the center row)
    double *B = (double *)calloc((d + 1), sizeof(double));

    // Compute N = X^T X
    for(int i = 0; i < W; i++)
    {
        for(int j = 0; j <= d; j++)
        {
            for(int k = 0; k <= d; k++)
            {
                N[j * (d+1) + k] += X[i * (d + 1) + j] * X[i * (d + 1) + k];
            }
        }
    }

    // We want the center row of X^T, i.e. row m of X => X[m, j] for j=0..d
    // But actually, "center row" in terms of the impulse shape is i=m in X.
    // So B[j] = X[m, j].
    for(int j = 0; j <= d; j++)
    {
        B[j] = X[m * (d + 1) + j];
    }

    // Solve N * a = B  for a => (d+1)x1
    // We'll do a naive Gaussian elimination for demonstration.

    // Copy N into M so we don't override it
    double *M = (double *)malloc((d + 1)*(d + 1)*sizeof(double));
    for(int i = 0; i < (d+1)*(d+1); i++)
        M[i] = N[i];
    double *a = (double *)calloc((d + 1), sizeof(double));
    for(int i = 0; i <= d; i++)
        a[i] = B[i];

    // Forward elimination
    for(int i = 0; i <= d; i++)
    {
        // Pivot
        double piv = M[i*(d+1) + i];
        for(int c = i; c <= d; c++)
            M[i*(d+1) + c] /= piv;
        a[i] /= piv;

        for(int r = i+1; r <= d; r++)
        {
            double mult = M[r*(d+1) + i];
            for(int c = i; c <= d; c++)
                M[r*(d+1) + c] -= mult * M[i*(d+1) + c];
            a[r] -= mult * a[i];
        }
    }
    // Back substitution
    for(int i = d; i >= 0; i--)
    {
        for(int r = i-1; r >= 0; r--)
        {
            double mult = M[r*(d+1) + i];
            M[r*(d+1) + i] = 0.0;
            a[r] -= mult * a[i];
        }
    }

    // Now "a" holds the row of G that, when convolved with the data,
    // yields the smoothed value at the center. Since we want a linear
    // filter (length W), those are exactly the coefficients in "a"
    // but we have to apply them at each shift around the center.

    // The final filter mask for the entire window is in "filter[i] = a0 + a1*(i-m) + ..."
    // However, the standard approach is to apply the same set of
    // coefficients “a” shifted properly. Another direct approach: each
    // output at index i is the dot product of the window with the filter
    // centered at i. Since we just want a single set of symmetrical
    // coefficients for smoothing, we can store them by direct polynomial evaluation:

    // For demonstration: coeffs[i] = polynomial(i - m)
    // i = 0..W-1
    for(int i = 0; i < W; i++)
    {
        double x = (double)(i - m);
        double val = 0.0;
        for(int j = 0; j <= d; j++)
            val += a[j] * pow(x, j);
        coeffs[i] = val;
    }

    free(X);
    free(N);
    free(B);
    free(M);
    free(a);
}

/* -------------------------------------------------------------------------
      Apply precomputed Savitzky-Golay filter to a local window of data.
      'windowSize' must match length of 'sgCoeffs'. We assume the data
      length here is exactly 'windowSize'.
   ------------------------------------------------------------------------- */
static void applySavitzkyGolayWindow(const double *dataWindow,
                                     double *filtered,
                                     int windowSize,
                                     const double *sgCoeffs)
{
    // The dataWindow has length windowSize
    // The filter mask sgCoeffs also has length windowSize
    // For a standard SG smoothing, the output at the center is the dot product
    // of dataWindow[i] * sgCoeffs[i]. But in practice for a "moving average" style,
    // you might shift the window. Below we just do a direct dot product to yield a
    // single "center" value for the entire window. Another approach is to compute
    // a smoothed value at each sample inside the window. For your advanced DMA,
    // you might want just the local "trend" (a single polynomial at each sub-block).
    // The simplest approach is to sum up data[i]*coeffs[i].

    double val = 0.0;
    for(int i = 0; i < windowSize; i++)
    {
        val += dataWindow[i] * sgCoeffs[i];
    }

    // We'll fill the entire "filtered" array with the same local trend estimate
    // if we want to mimic a single polynomial fit. Another option is a full SG,
    // giving a unique trend for each point. Adjust to your usage as needed.
    for(int i = 0; i < windowSize; i++)
    {
        filtered[i] = val; 
    }
}

//polynomial fit
void polynomialFit(int obs, int degree, double *dx, double *dy, double *store)
{
    gsl_matrix *X = gsl_matrix_alloc(obs, degree);
    gsl_vector *y = gsl_vector_alloc(obs);
    gsl_vector *c = gsl_vector_alloc(degree);
    gsl_matrix *cov = gsl_matrix_alloc(degree, degree);

    for(int i = 0; i < obs; i++)
    {
        for(int j = 0; j < degree; j++)
        {
            gsl_matrix_set(X, i, j, pow(dx[i], j));
        }
        gsl_vector_set(y, i, dy[i]);
    }

    double chisq;
    gsl_multifit_linear_workspace *ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    for(int i = 0; i < degree; i++)
    {
        store[i] = gsl_vector_get(c, i);
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);
}

static inline void mf_build_moments_S(int len, int p, double *S) {
    // S[k] = sum_{w=0}^{len-1} (w^k), for k=0..2p
    // This depends only on window length and polynomial order.
    for (int k = 0; k <= 2*p; ++k) S[k] = 0.0;
    for (int w = 0; w < len; ++w) {
        double x = (double)w;
        for (int k = 0; k <= 2*p; ++k) {
            S[k] += quickPow(x, k);
        }
    }
}

static inline void mf_build_M_template(const double *S, int p, double *M0) {
    // M0[a,b] = S[a+b], a,b = 0..p (normal equations' moment matrix)
    for (int a = 0; a <= p; ++a) {
        for (int b = 0; b <= p; ++b) {
            M0[a*(p+1) + b] = S[a+b];
        }
    }
}

static inline double mf_block_mean_square(
    const double *y, int start, int len, int p,
    const double *M0, double *M_scratch, double *b_scratch)
{
    // Build right-hand side vector T[a] = sum y(w)*x^a, a=0..p, reuse b_scratch for T
    for (int a = 0; a <= p; ++a) b_scratch[a] = 0.0;

    for (int w = 0; w < len; ++w) {
        double x   = (double)w;
        double val = y[start + w];
        for (int a = 0; a <= p; ++a)
            b_scratch[a] += val * quickPow(x, a);
    }

    // Copy M0 to M_scratch (solve modifies it)
    int dim = (p + 1) * (p + 1);
    memcpy(M_scratch, M0, (size_t)dim * sizeof(double));

    // Solve for coefficients in-place: M_scratch * b_scratch = T
    solveLinearSystem(M_scratch, b_scratch, p);

    // Accumulate residuals for this block
    double blockSum = 0.0;
    for (int w = 0; w < len; ++w) {
        double x = (double)w;
        double fitVal = 0.0;
        for (int a = 0; a <= p; ++a)
            fitVal += b_scratch[a] * quickPow(x, a);
        double diff = y[start + w] - fitVal;
        blockSum += diff * diff;
    }

    // Return mean square over the block
    return blockSum / (double)len;
}
