//    cLoops.c - C loops of fathon package
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

#include "cFuncs.h"
#include "cLoops.h"
#include "omp.h"

#define LQ -3.0e-15
#define HQ 3.0e-15

//main loop for unbiased DFA
void flucUDFACompute(double *y_vec, double *t_vec, int y_len, int *wins_vec, int num_wins, int pol, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < num_wins; i++)
#else
    for(int i = 0; i < num_wins; i++)
#endif
    {
        int s = wins_vec[i];
        int n_wins = y_len - s + 1;

        double *fit_coeffs = malloc((pol + 1) * sizeof(double));
        double *df = malloc(s * sizeof(double));

        double f = 0.0;
#ifdef _WIN64
        int start = 0;
        for(start = 0; start < n_wins; start++)
#else
        for(int start = 0; start < n_wins; start++)
#endif
        {
            polynomialFit(s, pol + 1, t_vec + start, y_vec + start, fit_coeffs);
            for(int j = 0; j < s; j++)
            {
                df[j] = y_vec[start + j];
                for(int k = 0; k < (pol + 1); k++)
                {
                    df[j] -= fit_coeffs[k] * pow(t_vec[start + j], k);
                }
            }
        
            double df_sum = 0.0, df_2_sum = 0.0, df_even_sum = 0.0, df_odd_sum = 0.0, df_shift_sum = 0.0;
            for(int j = 0; j < s; j++)
            {
                df_sum += df[j];
                df_2_sum += df[j] * df[j];
            }
            for(int j = 0; j < s; j += 2)
            {
                df_odd_sum += df[j];
            }
            for(int j = 1; j < s; j += 2)
            {
                df_even_sum += df[j];
            }
            for(int j = 0; j < (s - 1); j++)
            {
                df_shift_sum += (df[j] * df[j + 1]);
            }
    
            double df_neg_mean = (df_odd_sum - df_even_sum) / (double)s;
            double df_neg_var = df_2_sum / (double)s - df_neg_mean * df_neg_mean;
            double df_pos_mean = df_sum / (double)s;
            double df_pos_var = df_2_sum / (double)s - df_pos_mean * df_pos_mean;
        
            double df_pos_shift = (df_shift_sum + df_pos_mean * (df[0] + df[s - 1] - df_pos_mean * (s + 1))) / df_pos_var;
            double df_neg_shift = (-df_shift_sum + df_neg_mean * (df[0] + pow(-1.0, s + 1) * df[s - 1] - df_neg_mean * (s + 1))) / df_neg_var;
            double rho_A = (s + df_pos_shift) / (double)(2 * s - 1);
            double rho_B = (s + df_neg_shift) / (double)(2 * s - 1);
        
            double rho_A_star = rho_A + (1 + 3 * rho_A) / (double)(2 * s);
            double rho_B_star = rho_B + (1 + 3 * rho_B) / (double)(2 * s);
            f += ((rho_A_star + rho_B_star) * (1 - 1.0 / (double)(2 * s)) * df_pos_var);
        }
        f_vec[i] = sqrt(f * sqrt((s - 1) / (double)s) / (double)(n_wins));

        free(fit_coeffs);
        free(df);
    }
}

//main loop for DFA (computes fluctuations starting from the beginning of the array y)
void flucDFAForwCompute(double *y, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var = y[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var -= fit_coeffs[k] * pow(t[start_lim + j], k);
                }
                f += pow(var, 2.0);
            }
    
            free(fit_coeffs);
        }
    
        f_vec[i] = sqrt(f / (N_s * curr_win_size));
    }
}

//main loop for DFA (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the array y)
void flucDFAForwBackwCompute(double *y, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs[k] * pow(t[start_lim + j], k);
                }
                f += pow(var_1, 2.0);
            }

            start_lim = v * curr_win_size + (N - N_s * curr_win_size);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_2 = y[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_2 -= fit_coeffs[k] * pow(t[start_lim + j], k);
                }
                f += pow(var_2, 2.0);
            }
    
            free(fit_coeffs);
        }
    
        f_vec[i] = sqrt(f / (2.0 * N_s * curr_win_size));
    }
}

//main loop for unbiased DMA
void flucUDMACompute(double *y_vec, int y_len, int *wins_vec, int num_wins, int sgPolyOrder, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < num_wins; i++)
#else
    for(int i=0; i < num_wins; i++)
#endif
    {
        int s = wins_vec[i];               // subwindow size
        int p = sgPolyOrder;               // polynomial order
        if(s <= p) {
            // degenerate case
            f_vec[i] = 0.0;
            continue;
        }

        // We'll accumulate a corrected sum into "f"
        double f = 0.0;
        // The number of subwindows of length s
        int n_wins = y_len - s + 1;

        // For each subwindow [start..start+s-1]
        for(int start=0; start < n_wins; start++)
        {
            // 1) Build partial sums S(k), T(k) for polynomial fit
            double *S = (double *)calloc(2*p+1, sizeof(double));
            double *T = (double *)calloc(p+1,     sizeof(double));

            for(int w=0; w < s; w++)
            {
                double x = (double)w;  // local coordinate
                double y = y_vec[start + w];
                // accumulate for X^T X
                for(int kPow=0; kPow <= 2*p; kPow++){
                    S[kPow] += quickPow(x, kPow);
                }
                // accumulate for X^T y
                for(int kPow=0; kPow <= p; kPow++){
                    T[kPow] += (y * quickPow(x, kPow));
                }
            }

            // form normal eq M(a,b) = S(a+b), b[a] = T[a]
            double *M = (double *)calloc((p+1)*(p+1), sizeof(double));
            double *coef = (double *)calloc(p+1, sizeof(double));
            for(int a=0; a<=p; a++){
                coef[a] = T[a];
                for(int bcol=0; bcol<=p; bcol++){
                    M[a*(p+1) + bcol] = S[a + bcol];
                }
            }

            solveLinearSystem(M, coef, p);

            free(S);
            free(T);
            free(M);

            // 2) Compute the residual df[j] = y_vec[...] - polyFit(x=j)
            double *df = (double *)calloc(s, sizeof(double));
            for(int w=0; w<s; w++)
            {
                double x = (double)w;
                double fitVal = 0.0;
                for(int mPow=0; mPow<=p; mPow++){
                    fitVal += coef[mPow]*quickPow(x, mPow);
                }
                df[w] = y_vec[start + w] - fitVal;
            }
            free(coef);

            // 3) We do the "unbiased correction" from cDMA4:
            //    sums of df, df^2, df(odd), df(even), shift sums, etc.
            double df_sum=0.0, df_2_sum=0.0;
            double df_odd_sum=0.0, df_even_sum=0.0;
            double df_shift_sum=0.0;

            for(int j=0; j<s; j++){
                double d = df[j];
                df_sum += d;
                df_2_sum += d*d;
            }
            // odd/even in cDMA usage typically means j=0 is "even" or "odd"?
            // In the original code, it was j=0.., with j%2=0 => "df_odd_sum".
            // We'll do the same:
            for(int j=0; j<s; j+=2){
                df_odd_sum += df[j];
            }
            for(int j=1; j<s; j+=2){
                df_even_sum += df[j];
            }
            for(int j=0; j<(s-1); j++){
                df_shift_sum += (df[j]*df[j+1]);
            }

            // 4) from cDMA4 logic:
            double df_neg_mean = (df_odd_sum - df_even_sum) / (double)s;
            double df_neg_var = df_2_sum/(double)s - df_neg_mean*df_neg_mean;

            double df_pos_mean = df_sum / (double)s;
            double df_pos_var  = df_2_sum/(double)s - df_pos_mean*df_pos_mean;

            double df_pos_shift = ( df_shift_sum
                                    + df_pos_mean*(df[0] + df[s-1]
                                    - df_pos_mean*(s+1)) ) / df_pos_var;

            double df_neg_shift = ( - df_shift_sum
                                    + df_neg_mean*( df[0] + pow(-1.0, s+1)*df[s-1]
                                    - df_neg_mean*(s+1)) ) / df_neg_var;

            // from cDMA4:
            double rho_A = (s + df_pos_shift) / (2.0*s - 1.0);
            double rho_B = (s + df_neg_shift) / (2.0*s - 1.0);

            double rho_A_star = rho_A + (1 + 3*rho_A)/(2.0*s);
            double rho_B_star = rho_B + (1 + 3*rho_B)/(2.0*s);

            double local_contrib = (rho_A_star + rho_B_star)
                                    * (1.0 - 1.0/(2.0*s))
                                    * df_pos_var;

            f += local_contrib;

            free(df);
        }

        // 5) final unbiased scaling => same as cDMA4
        //    sqrt( f * sqrt((s-1)/s) / n_wins )
        double factor = sqrt( (double)(s-1) / (double)s );
        f_vec[i] = sqrt( f * factor / (double)n_wins );
    }
}


//main loop for DMA (computes fluctuations starting from the beginning of the array y)
void flucDMAForwCompute(double *y, int N, int *wins, int n_wins, int sgPolyOrder, double *f_vec)
{
    #ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int p = sgPolyOrder;

        if(curr_win_size <= p) {
            f_vec[i] = 0.0;
            continue;
        }

        // Number of segments we can fit in the data
        int n_segments = N / curr_win_size;

        double sumSq = 0.0; // accumulate sum of squared residuals

        for(int seg = 0; seg < n_segments; seg++)
        {
            int start = seg * curr_win_size;

            // Build partial sums S, T for sub-block [start..start+curr_win_size-1]
            double *S = (double *)calloc(2*p+1, sizeof(double));
            double *T = (double *)calloc(p+1,     sizeof(double));

            for(int w = 0; w < curr_win_size; w++)
            {
                double x = (double)w; // local coordinate
                double val = y[start + w];
                for(int kPow = 0; kPow <= 2*p; kPow++)
                    S[kPow] += quickPow(x, kPow);

                for(int kPow = 0; kPow <= p; kPow++)
                    T[kPow] += val * quickPow(x, kPow);
            }

            // Solve normal eq
            double *M = (double *)calloc((p+1)*(p+1), sizeof(double));
            double *b = (double *)calloc(p+1,         sizeof(double));
            for(int a=0; a<=p; a++){
                b[a] = T[a];
                for(int bcol=0; bcol<=p; bcol++){
                    M[a*(p+1) + bcol] = S[a+bcol];
                }
            }
            solveLinearSystem(M, b, p);

            // Accumulate residuals
            double blockSum = 0.0;
            for(int w = 0; w < curr_win_size; w++)
            {
                double x = (double)w;
                double fitVal = 0.0;
                for(int kk=0; kk<=p; kk++)
                    fitVal += b[kk]*quickPow(x, kk);

                double diff = y[start + w] - fitVal;
                blockSum += diff*diff;
            }

            sumSq += blockSum;

            free(S);
            free(T);
            free(M);
            free(b);
        }

        // final fluctuation for this window size
        double meanSq = sumSq / (n_segments*curr_win_size);
        f_vec[i] = sqrt(meanSq);
    }
}

//main loop for DMA (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the array y)
void flucDMAForwBackwCompute(double *y, int N, int *wins, int n_wins, int sgPolyOrder, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int p = sgPolyOrder;

        if(curr_win_size <= p) {
            f_vec[i] = 0.0;
            continue;
        }

        int n_segments = N / curr_win_size;
        double sumSq = 0.0;

        for(int seg = 0; seg < n_segments; seg++)
        {
            // === Forward block ===
            int start_forw = seg*curr_win_size;
            {
                double *S = (double *)calloc(2*p+1, sizeof(double));
                double *T = (double *)calloc(p+1,     sizeof(double));

                for(int w=0; w<curr_win_size; w++){
                    double x = (double)w;
                    double val = y[start_forw + w];
                    for(int kPow=0; kPow <= 2*p; kPow++)
                        S[kPow] += quickPow(x, kPow);
                    for(int kPow=0; kPow <= p; kPow++)
                        T[kPow] += val * quickPow(x, kPow);
                }
                double *M = (double *)calloc((p+1)*(p+1), sizeof(double));
                double *b = (double *)calloc(p+1,         sizeof(double));
                for(int a=0; a<=p; a++){
                    b[a] = T[a];
                    for(int bcol=0; bcol<=p; bcol++)
                        M[a*(p+1)+bcol] = S[a+bcol];
                }
                solveLinearSystem(M, b, p);

                double localSum = 0.0;
                for(int w=0; w<curr_win_size; w++){
                    double x = (double)w;
                    double fitVal = 0.0;
                    for(int kk=0; kk<=p; kk++)
                        fitVal += b[kk]*quickPow(x, kk);
                    double diff = y[start_forw + w] - fitVal;
                    localSum += diff*diff;
                }
                sumSq += localSum;

                free(S); free(T);
                free(M); free(b);
            }

            // === Backward block ===
            //   Start from the end of the array, stepping in segments of size curr_win_size
            //   Typically (N - n_segments*curr_win_size) is leftover. So your code does:
            int start_back = seg*curr_win_size + (N - n_segments*curr_win_size);
            {
                double *S = (double *)calloc(2*p+1, sizeof(double));
                double *T = (double *)calloc(p+1,     sizeof(double));

                for(int w=0; w<curr_win_size; w++){
                    double x = (double)w;
                    double val = y[start_back + w];
                    for(int kPow=0; kPow <= 2*p; kPow++)
                        S[kPow] += quickPow(x, kPow);
                    for(int kPow=0; kPow <= p; kPow++)
                        T[kPow] += val*quickPow(x, kPow);
                }

                double *M = (double *)calloc((p+1)*(p+1), sizeof(double));
                double *b = (double *)calloc(p+1,         sizeof(double));
                for(int a=0; a<=p; a++){
                    b[a] = T[a];
                    for(int bcol=0; bcol<=p; bcol++)
                        M[a*(p+1) + bcol] = S[a+bcol];
                }
                solveLinearSystem(M, b, p);

                double localSum = 0.0;
                for(int w=0; w<curr_win_size; w++){
                    double x = (double)w;
                    double fitVal = 0.0;
                    for(int kk=0; kk<=p; kk++)
                        fitVal += b[kk]*quickPow(x, kk);
                    double diff = y[start_back + w] - fitVal;
                    localSum += diff*diff;
                }
                sumSq += localSum;

                free(S); free(T);
                free(M); free(b);
            }
        }

        // Combine forward/backward residual => RMS
        // The total # of blocks is 2*n_segments (forward + backward),
        // each block size = curr_win_size
        double meanSq = sumSq / (2.0 * n_segments * curr_win_size);
        f_vec[i] = sqrt(meanSq);
    }
}   

//main loop for MFDFA (computes fluctuations starting from the beginning of the array y)
void flucMFDFAForwCompute(double *y, double *t, int N, int *wins, int n_wins, double *qs, int n_q, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int iq = 0;
#endif

#ifdef _WIN64
#pragma omp parallel for
    for(iq = 0; iq < n_q; iq++)
    {
        int i = 0;
        for(i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for(int iq = 0; iq < n_q; iq++)
    {
        for(int i = 0; i < n_wins; i++)
#endif
        {
            double q = qs[iq];
            int curr_win_size = wins[i];
            int N_s = N / curr_win_size;
            double f = 0.0;
#ifdef _WIN64
            int v = 0;
            for(v = 0; v < N_s; v++)
#else
            for(int v = 0; v < N_s; v++)
#endif
            {
                double rms = 0.0;
                int start_lim = v * curr_win_size;
                double *fit_coeffs = malloc((pol_ord + 1) * sizeof(double));
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var = y[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var -= fit_coeffs[k] * pow(t[start_lim + j], k);
                    }
                    rms += pow(var, 2.0);
                }
        
                if((q >= LQ) && (q <= HQ))
                {
                    f += log(rms / (double)curr_win_size);
                }
                else
                {
                    f += pow(rms / (double)curr_win_size, 0.5 * q);
                }
        
                free(fit_coeffs);
            }
    
            if((q >= LQ) && (q <= HQ))
            {
                f_vec[iq * n_wins + i] = exp(f / (double)(2 * N_s));
            }
            else
            {
                f_vec[iq * n_wins + i] = pow(f / (double)N_s, 1 / (double)q);
            }
        }
    }
}

//main loop for MFDFA (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the array y)
void flucMFDFAForwBackwCompute(double *y, double *t, int N, int *wins, int n_wins, double *qs, int n_q, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int iq = 0;
#endif

#ifdef _WIN64
#pragma omp parallel for
    for(iq = 0; iq < n_q; iq++)
    {
        int i = 0;
        for(i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for(int iq = 0; iq < n_q; iq++)
    {
        for(int i = 0; i < n_wins; i++)
#endif
        {
            double q = qs[iq];
            int curr_win_size = wins[i];
            int N_s = N / curr_win_size;
            double f = 0.0;
#ifdef _WIN64
            int v = 0;
            for(v = 0; v < N_s; v++)
#else
            for(int v = 0; v < N_s; v++)
#endif
            {
                double rms1 = 0.0;
                double rms2 = 0.0;
                int start_lim = v * curr_win_size;
                double *fit_coeffs = malloc((pol_ord + 1) * sizeof(double));
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var_1 = y[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var_1 -= fit_coeffs[k] * pow(t[start_lim + j], k);
                    }
                    rms1 += pow(var_1, 2.0);
                }
        
                start_lim = v * curr_win_size + (N - N_s * curr_win_size);
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y + start_lim, fit_coeffs);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var_2 = y[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var_2 -= fit_coeffs[k] * pow(t[start_lim + j], k);
                    }
                    rms2 += pow(var_2, 2.0);
                }
        
                if((q >= LQ) && (q <= HQ))
                {
                    f += (log(rms1 / (double)curr_win_size) + log(rms2 / (double)curr_win_size));
                }
                else
                {
                    f += (pow(rms1 / (double)curr_win_size, 0.5 * q) + pow(rms2 / (double)curr_win_size, 0.5 * q));
                }
        
                free(fit_coeffs);
            }
        
            if((q >= LQ) && (q <= HQ))
            {
                f_vec[iq * n_wins + i] = exp(f / (double)(4 * N_s));
            }
            else
            {
                f_vec[iq * n_wins + i] = pow(f / (double)(2 * N_s), 1 / (double)q);
            }
        }
    }
}

//main loop for MFDMA (computes fluctuations starting from the beginning of the array y)
void flucMFDMAForwCompute(double *y, int N,
                          int *wins, int n_wins,
                          double *qs, int n_q,
                          int pol_ord,
                          double *f_vec)
{
#ifdef _WIN64
#pragma omp parallel for
    for (int iq = 0; iq < n_q; iq++)
    {
        for (int i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for (int iq = 0; iq < n_q; iq++)
    {
        for (int i = 0; i < n_wins; i++)
#endif
        {
            const double q = qs[iq];
            const int curr_win_size = wins[i];
            const int p = pol_ord;

            if (curr_win_size <= p) {
                f_vec[iq * n_wins + i] = 0.0;
                continue;
            }

            const int n_segments = N / curr_win_size;

            // Precompute S and M template once per (window size)
            double *S  = (double *)calloc(2*p + 1, sizeof(double));
            double *M0 = (double *)malloc((size_t)(p+1)*(p+1)*sizeof(double));
            mf_build_moments_S(curr_win_size, p, S);
            mf_build_M_template(S, p, M0);

            // Scratch reused across segments
            double *M = (double *)malloc((size_t)(p+1)*(p+1)*sizeof(double));
            double *b = (double *)malloc((size_t)(p+1) * sizeof(double));

            double acc = 0.0; // accumulator of either logs or power-means
#ifdef _WIN64
            for (int seg = 0; seg < n_segments; seg++)
#else
            for (int seg = 0; seg < n_segments; seg++)
#endif
            {
                const int start = seg * curr_win_size;
                const double ms = mf_block_mean_square(y, start, curr_win_size, p, M0, M, b);
                if ((q >= LQ) && (q <= HQ)) {
                    // geometric mean of RMS -> 0.5 factor applies outside via the final exp
                    // (keep log of mean square, we'll divide by 2 later)
                    acc += log(ms);
                } else {
                    acc += pow(ms, 0.5 * q);
                }
            }

            double result;
            if ((q >= LQ) && (q <= HQ)) {
                // RMS geometric mean: exp( (1/(2*n_segments)) * sum log(ms) )
                result = exp(acc / (2.0 * (double)n_segments));
            } else {
                // Power mean of RMS: ( (1/n_segments) * sum ms^{q/2} )^{1/q}
                result = pow(acc / (double)n_segments, 1.0 / q);
            }

            f_vec[iq * n_wins + i] = result;

            free(S);
            free(M0);
            free(M);
            free(b);
        }
    }
}

//main loop for MFDMA (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the array y)
void flucMFDMAForwBackwCompute(double *y, int N,
                               int *wins, int n_wins,
                               double *qs, int n_q,
                               int pol_ord,
                               double *f_vec)
{
#ifdef _WIN64
#pragma omp parallel for
    for (int iq = 0; iq < n_q; iq++)
    {
        for (int i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for (int iq = 0; iq < n_q; iq++)
    {
        for (int i = 0; i < n_wins; i++)
#endif
        {
            const double q = qs[iq];
            const int curr_win_size = wins[i];
            const int p = pol_ord;

            if (curr_win_size <= p) {
                f_vec[iq * n_wins + i] = 0.0;
                continue;
            }

            const int n_segments = N / curr_win_size;
            const int leftover   = N - n_segments * curr_win_size;

            // Precompute S and M template once per (window size)
            double *S  = (double *)calloc(2*p + 1, sizeof(double));
            double *M0 = (double *)malloc((size_t)(p+1)*(p+1)*sizeof(double));
            mf_build_moments_S(curr_win_size, p, S);
            mf_build_M_template(S, p, M0);

            // Scratch reused across segments
            double *M = (double *)malloc((size_t)(p+1)*(p+1)*sizeof(double));
            double *b = (double *)malloc((size_t)(p+1) * sizeof(double));

            double acc = 0.0; // will sum both directions
#ifdef _WIN64
            for (int seg = 0; seg < n_segments; seg++)
#else
            for (int seg = 0; seg < n_segments; seg++)
#endif
            {
                // Forward block
                const int start_forw = seg * curr_win_size;
                const double ms1 = mf_block_mean_square(y, start_forw, curr_win_size, p, M0, M, b);

                // Backward block (from the end; align blocks as in your DMA)
                const int start_back = seg * curr_win_size + leftover;
                const double ms2 = mf_block_mean_square(y, start_back, curr_win_size, p, M0, M, b);

                if ((q >= LQ) && (q <= HQ)) {
                    acc += (log(ms1) + log(ms2));
                } else {
                    acc += (pow(ms1, 0.5 * q) + pow(ms2, 0.5 * q));
                }
            }

            double result;
            if ((q >= LQ) && (q <= HQ)) {
                // total blocks = 2*n_segments
                result = exp(acc / (4.0 * (double)n_segments));
            } else {
                result = pow(acc / (2.0 * (double)n_segments), 1.0 / q);
            }

            f_vec[iq * n_wins + i] = result;

            free(S);
            free(M0);
            free(M);
            free(b);
        }
    }
}

//main loop for DCCA (computes fluctuations using absolute values)
void flucDCCAAbsCompute(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N - curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            double *fit_coeffs1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size + 1, pol_ord + 1, t + v, y1 + v, fit_coeffs1);
            polynomialFit(curr_win_size + 1, pol_ord + 1, t + v, y2 + v, fit_coeffs2);
    
            for(int j = 0; j <= curr_win_size; j++)
            {
                double var_1 = y1[v + j];
                double var_2 = y2[v + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs1[k] * pow(t[v + j], k);
                    var_2 -= fit_coeffs2[k] * pow(t[v + j], k);
                }
                f += fabs(var_1 * var_2);
            }
    
            free(fit_coeffs1);
            free(fit_coeffs2);
        }

        f_vec[i] = sqrt(f / (N_s * (curr_win_size - 1)));
    }
}

//main loop for DCCA (computes fluctuations without using absolute values)
void flucDCCANoAbsCompute(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N - curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            double *fit_coeffs1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size + 1, pol_ord + 1, t + v, y1 + v, fit_coeffs1);
            polynomialFit(curr_win_size + 1, pol_ord + 1, t + v, y2 + v, fit_coeffs2);
    
            for(int j = 0; j <= curr_win_size; j++)
            {
                double var_1 = y1[v + j];
                double var_2 = y2[v + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs1[k] * pow(t[v + j], k);
                    var_2 -= fit_coeffs2[k] * pow(t[v + j], k);
                }
                f += var_1 * var_2;
            }
    
            free(fit_coeffs1);
            free(fit_coeffs2);
        }

        f_vec[i] = f / (N_s * (curr_win_size - 1));
    }
}

//main loop for HT (computes fluctuations)
double HTCompute(double *y, double *t, int scale, int N, int pol_ord, int v)
{
    double f = 0.0;
    double *fit_coeffs = malloc((pol_ord + 1) * sizeof(double));
    polynomialFit(scale, pol_ord + 1, t + v, y + v, fit_coeffs);

    for(int j = 0; j < scale; j++)
    {
        double var = y[v + j];
        for(int k = 0; k < (pol_ord + 1); k++)
        {
            var -= fit_coeffs[k] * pow(t[v + j], k);
        }
        f += pow(var, 2.0);
    }

    f = sqrt(f / (double)scale);

    free(fit_coeffs);

    return f;
}

//main loop for DCCA without overlap (computes fluctuations starting from the beginning
// of the array y and using absolute values)
void flucDCCAForwAbsComputeNoOverlap(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += fabs(var_1 * var_2);
            }
    
            free(fit_coeffs_1);
            free(fit_coeffs_2);
        }

        f_vec[i] = sqrt(f / (N_s * curr_win_size));
    }
}

//main loop for DCCA without overlap (ccomputes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the arrays y1 and y2, and using absolute values)
void flucDCCAForwBackwAbsComputeNoOverlap(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += fabs(var_1 * var_2);
            }
    
            start_lim = v * curr_win_size + (N - N_s * curr_win_size);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += fabs(var_1 * var_2);
            }
    
            free(fit_coeffs_1);
            free(fit_coeffs_2);
        }

        f_vec[i] = sqrt(f / (2.0 * N_s * curr_win_size));
    }
}

//main loop for DCCA without overlap (computes fluctuations starting from the beginning
// of the array y)
void flucDCCAForwNoAbsComputeNoOverlap(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += var_1 * var_2;
            }
    
            free(fit_coeffs_1);
            free(fit_coeffs_2);
        }

        f_vec[i] = f / (N_s * curr_win_size);
    }
}

//main loop for DCCA without overlap (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the arrays y1 and y2)
void flucDCCAForwBackwNoAbsComputeNoOverlap(double *y1, double *y2, double *t, int N, int *wins, int n_wins, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int i = 0;
#endif

#pragma omp parallel for
#ifdef _WIN64
    for(i = 0; i < n_wins; i++)
#else
    for(int i = 0; i < n_wins; i++)
#endif
    {
        int curr_win_size = wins[i];
        int N_s = N / curr_win_size;
        double f = 0.0;
#ifdef _WIN64
        int v = 0;
        for(v = 0; v < N_s; v++)
#else
        for(int v = 0; v < N_s; v++)
#endif
        {
            int start_lim = v * curr_win_size;
            double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
            double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += var_1 * var_2;
            }
    
            start_lim = v * curr_win_size + (N - N_s * curr_win_size);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
            polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
    
            for(int j = 0; j < curr_win_size; j++)
            {
                double var_1 = y1[start_lim + j];
                double var_2 = y2[start_lim + j];
                for(int k = 0; k < (pol_ord + 1); k++)
                {
                    var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                    var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                }
                f += var_1 * var_2;
            }
    
            free(fit_coeffs_1);
            free(fit_coeffs_2);
        }

        f_vec[i] = f / (2.0 * N_s * curr_win_size);
    }
}

//main loop for MFDCCA (computes fluctuations starting from the beginning of the array y)
void flucMFDCCAForwCompute(double *y1, double *y2, double *t, int N, int *wins, int n_wins, double *qs, int n_q, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int iq = 0;
#endif

#ifdef _WIN64
#pragma omp parallel for
    for(iq = 0; iq < n_q; iq++)
    {
        int i = 0;
        for(i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for(int iq = 0; iq < n_q; iq++)
    {
        for(int i = 0; i < n_wins; i++)
#endif
        {
            double q = qs[iq];
            int curr_win_size = wins[i];
            int N_s = N / curr_win_size;
            double f = 0.0;
#ifdef _WIN64
            int v = 0;
            for(v = 0; v < N_s; v++)
#else
            for(int v = 0; v < N_s; v++)
#endif
            {
                double rms = 0.0;
                int start_lim = v * curr_win_size;
                double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
                double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var_1 = y1[start_lim + j];
                    double var_2 = y2[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                        var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                    }
                    rms += fabs(var_1 * var_2);
                }
        
                if((q >= LQ) && (q <= HQ))
                {
                    f += log(rms / (double)curr_win_size);
                }
                else
                {
                    f += pow(rms / (double)curr_win_size, 0.5 * q);
                }
        
                free(fit_coeffs_1);
                free(fit_coeffs_2);
            }
        
            if((q >= LQ) && (q <= HQ))
            {
                f_vec[iq * n_wins + i] = exp(f / (double)(2 * N_s));
            }
            else
            {
                f_vec[iq * n_wins + i] = pow(f / (double)N_s, 1 / (double)q);
            }
        }
    }
}

//main loop for MFDCCA (computes fluctuations starting from the beginning of the array y
//and then computes fluctuations again starting from the end of the array y)
void flucMFDCCAForwBackwCompute(double *y1, double *y2, double *t, int N, int *wins, int n_wins, double *qs, int n_q, int pol_ord, double *f_vec)
{
#ifdef _WIN64
    int iq = 0;
#endif

#ifdef _WIN64
#pragma omp parallel for
    for(iq = 0; iq < n_q; iq++)
    {
        int i = 0;
        for(i = 0; i < n_wins; i++)
#else
#pragma omp parallel for collapse(2)
    for(int iq = 0; iq < n_q; iq++)
    {
        for(int i = 0; i < n_wins; i++)
#endif
        {
            double q = qs[iq];
            int curr_win_size = wins[i];
            int N_s = N / curr_win_size;
            double f = 0.0;
#ifdef _WIN64
            int v = 0;
            for(v = 0; v < N_s; v++)
#else
            for(int v = 0; v < N_s; v++)
#endif
            {
                double rms1 = 0.0;
                double rms2 = 0.0;
                int start_lim = v * curr_win_size;
                double *fit_coeffs_1 = malloc((pol_ord + 1) * sizeof(double));
                double *fit_coeffs_2 = malloc((pol_ord + 1) * sizeof(double));
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var_1 = y1[start_lim + j];
                    double var_2 = y2[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                        var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                    }
                    rms1 += fabs(var_1 * var_2);
                }
        
                start_lim = v * curr_win_size + (N - N_s * curr_win_size);
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y1 + start_lim, fit_coeffs_1);
                polynomialFit(curr_win_size, pol_ord + 1, t + start_lim, y2 + start_lim, fit_coeffs_2);
        
                for(int j = 0; j < curr_win_size; j++)
                {
                    double var_1 = y1[start_lim + j];
                    double var_2 = y2[start_lim + j];
                    for(int k = 0; k < (pol_ord + 1); k++)
                    {
                        var_1 -= fit_coeffs_1[k] * pow(t[start_lim + j], k);
                        var_2 -= fit_coeffs_2[k] * pow(t[start_lim + j], k);
                    }
                    rms2 += fabs(var_1 * var_2);
                }
        
                if((q >= LQ) && (q <= HQ))
                {
                    f += (log(rms1 / (double)curr_win_size) + log(rms2 / (double)curr_win_size));
                }
                else
                {
                    f += (pow(rms1 / (double)curr_win_size, 0.5 * q) + pow(rms2 / (double)curr_win_size, 0.5 * q));
                }
        
                free(fit_coeffs_1);
                free(fit_coeffs_2);
            }
        
            if((q >= LQ) && (q <= HQ))
            {
                f_vec[iq * n_wins + i] = exp(f / (double)(4 * N_s));
            }
            else
            {
                f_vec[iq * n_wins + i] = pow(f / (double)(2 * N_s), 1 / (double)q);
            }
        }
    }
}
