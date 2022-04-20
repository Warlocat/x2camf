#include"gsl_functions.h"
#include<cmath>
#include<iostream>

using namespace std;

inline
static
int locMax3(const int a, const int b, const int c)
{
  int d = max(a, b);
  return max(d, c);
}

inline
static
int locMin3(const int a, const int b, const int c)
{
  int d = min(a, b);
  return min(d, c);
}

inline
static
int locMin5(const int a, const int b, const int c, const int d, const int e)
{
  int f = min(a, b);
  int g = min(c, d);
  int h = min(f, g);
  return min(e, h);
}

/* coefficients for gamma=7, kmax=8  Lanczos method */
static double lanczos_7_c[9] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
 -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
 -176.61502916214059906584551354,
  12.507343278686904814458936853,
 -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};
inline
static
int
lngamma_1_pade(const double eps, gsl_sf_result * result)
{
  /* Use (2,2) Pade for Log[Gamma[1+eps]]/eps
   * plus a correction series.
   */
  const double n1 = -1.0017419282349508699871138440;
  const double n2 =  1.7364839209922879823280541733;
  const double d1 =  1.2433006018858751556055436011;
  const double d2 =  5.0456274100274010152489597514;
  const double num = (eps + n1) * (eps + n2);
  const double den = (eps + d1) * (eps + d2);
  const double pade = 2.0816265188662692474880210318 * num / den;
  const double c0 =  0.004785324257581753;
  const double c1 = -0.01192457083645441;
  const double c2 =  0.01931961413960498;
  const double c3 = -0.02594027398725020;
  const double c4 =  0.03141928755021455;
  const double eps5 = eps*eps*eps*eps*eps;
  const double corr = eps5 * (c0 + eps*(c1 + eps*(c2 + eps*(c3 + c4*eps))));
  result->val = eps * (pade + corr);
  result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
  return GSL_SUCCESS;
}
inline
static
int
lngamma_2_pade(const double eps, gsl_sf_result * result)
{
  /* Use (2,2) Pade for Log[Gamma[2+eps]]/eps
   * plus a correction series.
   */
  const double n1 = 1.000895834786669227164446568;
  const double n2 = 4.209376735287755081642901277;
  const double d1 = 2.618851904903217274682578255;
  const double d2 = 10.85766559900983515322922936;
  const double num = (eps + n1) * (eps + n2);
  const double den = (eps + d1) * (eps + d2);
  const double pade = 2.85337998765781918463568869 * num/den;
  const double c0 =  0.0001139406357036744;
  const double c1 = -0.0001365435269792533;
  const double c2 =  0.0001067287169183665;
  const double c3 = -0.0000693271800931282;
  const double c4 =  0.0000407220927867950;
  const double eps5 = eps*eps*eps*eps*eps;
  const double corr = eps5 * (c0 + eps*(c1 + eps*(c2 + eps*(c3 + c4*eps))));
  result->val = eps * (pade + corr);
  result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
  return GSL_SUCCESS;
}
static
int
lngamma_lanczos(double x, gsl_sf_result * result)
{
  int k;
  double Ag;
  double term1, term2;

  x -= 1.0; /* Lanczos writes z! instead of Gamma(z) */

  Ag = lanczos_7_c[0];
  for(k=1; k<=8; k++) { Ag += lanczos_7_c[k]/(x+k); }

  /* (x+0.5)*log(x+7.5) - (x+7.5) + LogRootTwoPi_ + log(Ag(x)) */
  term1 = (x+0.5)*log((x+7.5)/M_E);
  term2 = LogRootTwoPi_ + log(Ag);
  result->val  = term1 + (term2 - 7.0);
  result->err  = 2.0 * GSL_DBL_EPSILON * (fabs(term1) + fabs(term2) + 7.0);
  result->err += GSL_DBL_EPSILON * fabs(result->val);

  return GSL_SUCCESS;
}
static
int
lngamma_sgn_0(double eps, gsl_sf_result * lng, double * sgn)
{
  /* calculate series for g(eps) = Gamma(eps) eps - 1/(1+eps) - eps/2 */
  const double c1  = -0.07721566490153286061;
  const double c2  = -0.01094400467202744461;
  const double c3  =  0.09252092391911371098;
  const double c4  = -0.01827191316559981266;
  const double c5  =  0.01800493109685479790;
  const double c6  = -0.00685088537872380685;
  const double c7  =  0.00399823955756846603;
  const double c8  = -0.00189430621687107802;
  const double c9  =  0.00097473237804513221;
  const double c10 = -0.00048434392722255893;
  const double g6  = c6+eps*(c7+eps*(c8 + eps*(c9 + eps*c10)));
  const double g   = eps*(c1+eps*(c2+eps*(c3+eps*(c4+eps*(c5+eps*g6)))));

  /* calculate Gamma(eps) eps, a positive quantity */
  const double gee = g + 1.0/(1.0+eps) + 0.5*eps;

  lng->val = log(gee/fabs(eps));
  lng->err = 4.0 * GSL_DBL_EPSILON * fabs(lng->val);
  *sgn = GSL_SIGN(eps);

  return GSL_SUCCESS;
}
int gsl_sf_psi_int_e(const int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n <= 0) {
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else if(n <= PSI_TABLE_NMAX) {
    result->val = psi_table[n];
    result->err = GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    /* Abramowitz+Stegun 6.3.18 */
    const double c2 = -1.0/12.0;
    const double c3 =  1.0/120.0;
    const double c4 = -1.0/252.0;
    const double c5 =  1.0/240.0;
    const double ni2 = (1.0/n)*(1.0/n);
    const double ser = ni2 * (c2 + ni2 * (c3 + ni2 * (c4 + ni2*c5)));
    result->val  = log(n) - 0.5/n + ser;
    result->err  = GSL_DBL_EPSILON * (fabs(log(n)) + fabs(0.5/n) + fabs(ser));
    result->err += GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
}
int gsl_sf_psi_1_int_e(const int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */
  if(n <= 0) {
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else if(n <= PSI_1_TABLE_NMAX) {
    result->val = psi_1_table[n];
    result->err = GSL_DBL_EPSILON * result->val;
    return GSL_SUCCESS;
  }
  else {
    /* Abramowitz+Stegun 6.4.12
     * double-precision for n > 100
     */
    const double c0 = -1.0/30.0;
    const double c1 =  1.0/42.0;
    const double c2 = -1.0/30.0;
    const double ni2 = (1.0/n)*(1.0/n);
    const double ser =  ni2*ni2 * (c0 + ni2*(c1 + c2*ni2));
    result->val = (1.0 + 0.5/n + 1.0/(6.0*n*n) + ser) / n;
    result->err = GSL_DBL_EPSILON * result->val;
    return GSL_SUCCESS;
  }
}
static inline int
cheb_eval_e(const cheb_series * cs,
            const double x,
            gsl_sf_result * result)
{
  int j;
  double d  = 0.0;
  double dd = 0.0;

  double y  = (2.0*x - cs->a - cs->b) / (cs->b - cs->a);
  double y2 = 2.0 * y;

  double e = 0.0;

  for(j = cs->order; j>=1; j--) {
    double temp = d;
    d = y2*d - dd + cs->c[j];
    e += fabs(y2*temp) + fabs(dd) + fabs(cs->c[j]);
    dd = temp;
  }

  { 
    double temp = d;
    d = y*d - dd + 0.5 * cs->c[0];
    e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs->c[0]);
  }

  result->val = d;
  result->err = GSL_DBL_EPSILON * e + fabs(cs->c[cs->order]);

  return GSL_SUCCESS;
}
int gsl_sf_lnfact_e(const unsigned int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n <= GSL_SF_FACT_NMAX){
    result->val = log(fact_table[n].f);
    result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    gsl_sf_lngamma_e(n+1.0, result);
    return GSL_SUCCESS;
  }
}
static int
psi_x(const double x, gsl_sf_result * result)
{
  const double y = fabs(x);

  if(x == 0.0 || x == -1.0 || x == -2.0) {
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else if(y >= 2.0) {
    const double t = 8.0/(y*y)-1.0;
    gsl_sf_result result_c;
    cheb_eval_e(&apsi_cs, t, &result_c);
    if(x < 0.0) {
      const double s = sin(M_PI*x);
      const double c = cos(M_PI*x);
      if(fabs(s) < 2.0*GSL_SQRT_DBL_MIN) {
        cout << "GSL:Input domain error!" << endl;
        exit(99);
      }
      else {
        result->val  = log(y) - 0.5/x + result_c.val - M_PI * c/s;
        result->err  = M_PI*fabs(x)*GSL_DBL_EPSILON/(s*s);
        result->err += result_c.err;
        result->err += GSL_DBL_EPSILON * fabs(result->val);
        return GSL_SUCCESS;
      }
    }
    else {
      result->val  = log(y) - 0.5/x + result_c.val;
      result->err  = result_c.err;
      result->err += GSL_DBL_EPSILON * fabs(result->val);
      return GSL_SUCCESS;
    }
  }
  else { /* -2 < x < 2 */
    gsl_sf_result result_c;

    if(x < -1.0) { /* x = -2 + v */
      const double v  = x + 2.0;
      const double t1 = 1.0/x;
      const double t2 = 1.0/(x+1.0);
      const double t3 = 1.0/v;
      cheb_eval_e(&psi_cs, 2.0*v-1.0, &result_c);
      
      result->val  = -(t1 + t2 + t3) + result_c.val;
      result->err  = GSL_DBL_EPSILON * (fabs(t1) + fabs(x/(t2*t2)) + fabs(x/(t3*t3)));
      result->err += result_c.err;
      result->err += GSL_DBL_EPSILON * fabs(result->val);
      return GSL_SUCCESS;
    }
    else if(x < 0.0) { /* x = -1 + v */
      const double v  = x + 1.0;
      const double t1 = 1.0/x;
      const double t2 = 1.0/v;
      cheb_eval_e(&psi_cs, 2.0*v-1.0, &result_c);
      
      result->val  = -(t1 + t2) + result_c.val;
      result->err  = GSL_DBL_EPSILON * (fabs(t1) + fabs(x/(t2*t2)));
      result->err += result_c.err;
      result->err += GSL_DBL_EPSILON * fabs(result->val);
      return GSL_SUCCESS;
    }
    else if(x < 1.0) { /* x = v */
      const double t1 = 1.0/x;
      cheb_eval_e(&psi_cs, 2.0*x-1.0, &result_c);
      
      result->val  = -t1 + result_c.val;
      result->err  = GSL_DBL_EPSILON * t1;
      result->err += result_c.err;
      result->err += GSL_DBL_EPSILON * fabs(result->val);
      return GSL_SUCCESS;
    }
    else { /* x = 1 + v */
      const double v = x - 1.0;
      return cheb_eval_e(&psi_cs, 2.0*v-1.0, result);
    }
  }
}
int gsl_sf_psi_e(const double x, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */
  return psi_x(x, result);
}
int gsl_sf_hzeta_e(const double s, const double q, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  const double max_bits = 54.0;
  const double ln_term0 = -s * log(q);  

  if((s > max_bits && q < 1.0) || (s > 0.5*max_bits && q < 0.25)) {
    result->val = pow(q, -s);
    result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else if(s > 0.5*max_bits && q < 1.0) {
    const double p1 = pow(q, -s);
    const double p2 = pow(q/(1.0+q), s);
    const double p3 = pow(q/(2.0+q), s);
    result->val = p1 * (1.0 + p2 + p3);
    result->err = GSL_DBL_EPSILON * (0.5*s + 2.0) * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    /* Euler-Maclaurin summation formula 
     * [Moshier, p. 400, with several typo corrections]
     */
    const int jmax = 12;
    const int kmax = 10;
    int j, k;
    const double pmax  = pow(kmax + q, -s);
    double scp = s;
    double pcp = pmax / (kmax + q);
    double ans = pmax*((kmax+q)/(s-1.0) + 0.5);

    for(k=0; k<kmax; k++) {
      ans += pow(k + q, -s);
    }

    for(j=0; j<=jmax; j++) {
      double delta = hzeta_c[j+1] * scp * pcp;
      ans += delta;
      if(fabs(delta/ans) < 0.5*GSL_DBL_EPSILON) break;
      scp *= (s+2*j+1)*(s+2*j+2);
      pcp /= (kmax + q)*(kmax + q);
    }

    result->val = ans;
    result->err = 2.0 * (jmax + 1.0) * GSL_DBL_EPSILON * fabs(ans);
    return GSL_SUCCESS;
  }
}
int gsl_sf_exp_mult_err_e(const double x, const double dx,
                             const double y, const double dy,
                             gsl_sf_result * result)
{
  const double ay  = fabs(y);

  if(y == 0.0) {
    result->val = 0.0;
    result->err = fabs(dy * exp(x));
    return GSL_SUCCESS;
  }
  else if(   ( x < 0.5*GSL_LOG_DBL_MAX   &&   x > 0.5*GSL_LOG_DBL_MIN)
          && (ay < 0.8*GSL_SQRT_DBL_MAX  &&  ay > 1.2*GSL_SQRT_DBL_MIN)
    ) {
    double ex = exp(x);
    result->val  = y * ex;
    result->err  = ex * (fabs(dy) + fabs(y*dx));
    result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    const double ly  = log(ay);
    const double lnr = x + ly;
    const double sy  = GSL_SIGN(y);
    const double M   = floor(x);
    const double N   = floor(ly);
    const double a   = x  - M;
    const double b   = ly - N;
    const double eMN = exp(M+N);
    const double eab = exp(a+b);
    result->val  = sy * eMN * eab;
    result->err  = eMN * eab * 2.0*GSL_DBL_EPSILON;
    result->err += eMN * eab * fabs(dy/y);
    result->err += eMN * eab * fabs(dx);
    return GSL_SUCCESS;
  }
}
static int
psi_n_xg0(const int n, const double x, gsl_sf_result * result)
{
  if(n == 0) {
    return gsl_sf_psi_e(x, result);
  }
  else {
    /* Abramowitz + Stegun 6.4.10 */
    gsl_sf_result ln_nf;
    gsl_sf_result hzeta;
    int stat_hz = gsl_sf_hzeta_e(n+1.0, x, &hzeta);
    int stat_nf = gsl_sf_lnfact_e((unsigned int) n, &ln_nf);
    int stat_e  = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
                                           hzeta.val, hzeta.err,
                                           result);
    if(GSL_IS_EVEN(n)) result->val = -result->val;
    return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
  }
}
int gsl_sf_psi_1_e(const double x, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(x == 0.0 || x == -1.0 || x == -2.0) {
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else if(x > 0.0)
  {
    return psi_n_xg0(1, x, result);
  }
  else if(x > -5.0)
  {
    /* Abramowitz + Stegun 6.4.6 */
    int M = -floor(x);
    double fx = x + M;
    double sum = 0.0;
    int m;

    if(fx == 0.0)
      cout << "GSL:Input domain error!" << endl;
      exit(99);

    for(m = 0; m < M; ++m)
      sum += 1.0/((x+m)*(x+m));

    {
      int stat_psi = psi_n_xg0(1, fx, result);
      result->val += sum;
      result->err += M * GSL_DBL_EPSILON * sum;
      return stat_psi;
    }
  }
  else
  {
    /* Abramowitz + Stegun 6.4.7 */
    const double sin_px = sin(M_PI * x);
    const double d = M_PI*M_PI/(sin_px*sin_px);
    gsl_sf_result r;
    int stat_psi = psi_n_xg0(1, 1.0-x, &r);
    result->val = d - r.val;
    result->err = r.err + 2.0*GSL_DBL_EPSILON*d;
    return stat_psi;
  }
}
int gsl_sf_psi_n_e(const int n, const double x, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n == 0)
  {
    return gsl_sf_psi_e(x, result);
  }
  else if(n == 1)
  {
    return gsl_sf_psi_1_e(x, result);
  }
  else if(n < 0 || x <= 0.0) {
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else {
    gsl_sf_result ln_nf;
    gsl_sf_result hzeta;
    int stat_hz = gsl_sf_hzeta_e(n+1.0, x, &hzeta);
    int stat_nf = gsl_sf_lnfact_e((unsigned int) n, &ln_nf);
    int stat_e  = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
                                           hzeta.val, hzeta.err,
                                           result);
    if(GSL_IS_EVEN(n)) result->val = -result->val;
    return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
  }
}
static
int
lngamma_sgn_sing(int N, double eps, gsl_sf_result * lng, double * sgn)
{
  if(eps == 0.0) {
    lng->val = 0.0;
    lng->err = 0.0;
    *sgn = 0.0;
    cout << "GSL:Input domain error!" << endl;
    exit(99);
  }
  else if(N == 1) {
    /* calculate series for
     * g = eps gamma(-1+eps) + 1 + eps/2 (1+3eps)/(1-eps^2)
     * double-precision for |eps| < 0.02
     */
    const double c0 =  0.07721566490153286061;
    const double c1 =  0.08815966957356030521;
    const double c2 = -0.00436125434555340577;
    const double c3 =  0.01391065882004640689;
    const double c4 = -0.00409427227680839100;
    const double c5 =  0.00275661310191541584;
    const double c6 = -0.00124162645565305019;
    const double c7 =  0.00065267976121802783;
    const double c8 = -0.00032205261682710437;
    const double c9 =  0.00016229131039545456;
    const double g5 = c5 + eps*(c6 + eps*(c7 + eps*(c8 + eps*c9)));
    const double g  = eps*(c0 + eps*(c1 + eps*(c2 + eps*(c3 + eps*(c4 + eps*g5)))));

    /* calculate eps gamma(-1+eps), a negative quantity */
    const double gam_e = g - 1.0 - 0.5*eps*(1.0+3.0*eps)/(1.0 - eps*eps);

    lng->val = log(fabs(gam_e)/fabs(eps));
    lng->err = 2.0 * GSL_DBL_EPSILON * fabs(lng->val);
    *sgn = ( eps > 0.0 ? -1.0 : 1.0 );
    return GSL_SUCCESS;
  }
  else {
    double g;

    /* series for sin(Pi(N+1-eps))/(Pi eps) modulo the sign
     * double-precision for |eps| < 0.02
     */
    const double cs1 = -1.6449340668482264365;
    const double cs2 =  0.8117424252833536436;
    const double cs3 = -0.1907518241220842137;
    const double cs4 =  0.0261478478176548005;
    const double cs5 = -0.0023460810354558236;
    const double e2  = eps*eps;
    const double sin_ser = 1.0 + e2*(cs1+e2*(cs2+e2*(cs3+e2*(cs4+e2*cs5))));

    /* calculate series for ln(gamma(1+N-eps))
     * double-precision for |eps| < 0.02
     */
    double aeps = fabs(eps);
    double c1, c2, c3, c4, c5, c6, c7;
    double lng_ser;
    gsl_sf_result c0;
    gsl_sf_result psi_0;
    gsl_sf_result psi_1;
    gsl_sf_result psi_2;
    gsl_sf_result psi_3;
    gsl_sf_result psi_4;
    gsl_sf_result psi_5;
    gsl_sf_result psi_6;
    psi_2.val = 0.0;
    psi_3.val = 0.0;
    psi_4.val = 0.0;
    psi_5.val = 0.0;
    psi_6.val = 0.0;
    gsl_sf_lnfact_e(N, &c0);
    gsl_sf_psi_int_e(N+1, &psi_0);
    gsl_sf_psi_1_int_e(N+1, &psi_1);
    if(aeps > 0.00001) gsl_sf_psi_n_e(2, N+1.0, &psi_2);
    if(aeps > 0.0002)  gsl_sf_psi_n_e(3, N+1.0, &psi_3);
    if(aeps > 0.001)   gsl_sf_psi_n_e(4, N+1.0, &psi_4);
    if(aeps > 0.005)   gsl_sf_psi_n_e(5, N+1.0, &psi_5);
    if(aeps > 0.01)    gsl_sf_psi_n_e(6, N+1.0, &psi_6);
    c1 = psi_0.val;
    c2 = psi_1.val/2.0;
    c3 = psi_2.val/6.0;
    c4 = psi_3.val/24.0;
    c5 = psi_4.val/120.0;
    c6 = psi_5.val/720.0;
    c7 = psi_6.val/5040.0;
    lng_ser = c0.val-eps*(c1-eps*(c2-eps*(c3-eps*(c4-eps*(c5-eps*(c6-eps*c7))))));

    /* calculate
     * g = ln(|eps gamma(-N+eps)|)
     *   = -ln(gamma(1+N-eps)) + ln(|eps Pi/sin(Pi(N+1+eps))|)
     */
    g = -lng_ser - log(sin_ser);

    lng->val = g - log(fabs(eps));
    lng->err = c0.err + 2.0 * GSL_DBL_EPSILON * (fabs(g) + fabs(lng->val));

    *sgn = ( GSL_IS_ODD(N) ? -1.0 : 1.0 ) * ( eps > 0.0 ? 1.0 : -1.0 );

    return GSL_SUCCESS;
  }
}
int gsl_sf_lngamma_e(double x, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(fabs(x - 1.0) < 0.01) {
    /* Note that we must amplify the errors
     * from the Pade evaluations because of
     * the way we must pass the argument, i.e.
     * writing (1-x) is a loss of precision
     * when x is near 1.
     */
    int stat = lngamma_1_pade(x - 1.0, result);
    result->err *= 1.0/(GSL_DBL_EPSILON + fabs(x - 1.0));
    return stat;
  }
  else if(fabs(x - 2.0) < 0.01) {
    int stat = lngamma_2_pade(x - 2.0, result);
    result->err *= 1.0/(GSL_DBL_EPSILON + fabs(x - 2.0));
    return stat;
  }
  else if(x >= 0.5) {
    return lngamma_lanczos(x, result);
  }
  else if(x == 0.0) {
    cout << "GSL:Domain error!" << endl;
    exit(99);
  }
  else if(fabs(x) < 0.02) {
    double sgn;
    return lngamma_sgn_0(x, result, &sgn);
  }
  else if(x > -0.5/(GSL_DBL_EPSILON*M_PI)) {
    /* Try to extract a fractional
     * part from x.
     */
    double z  = 1.0 - x;
    double s  = sin(M_PI*z);
    double as = fabs(s);
    if(s == 0.0) {
      cout << "GSL:Domain error!" << endl;
      exit(99);
    }
    else if(as < M_PI*0.015) {
      /* x is near a negative integer, -N */
      if(x < INT_MIN + 2.0) {
        result->val = 0.0;
        result->err = 0.0;
        cout << "GSL error: x is too negative." << endl;
        exit(99);
      }
      else {
        int N = -(int)(x - 0.5);
        double eps = x + N;
        double sgn;
        return lngamma_sgn_sing(N, eps, result, &sgn);
      }
    }
    else {
      gsl_sf_result lg_z;
      lngamma_lanczos(z, &lg_z);
      result->val = M_LNPI - (log(as) + lg_z.val);
      result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val) + lg_z.err;
      return GSL_SUCCESS;
    }
  }
  else {
    /* |x| was too large to extract any fractional part */
    result->val = 0.0;
    result->err = 0.0;
    cout << "GSL error: |x| was too large to extract any fractional part." << endl;
    exit(99);
  }
}

int gsl_sf_lnchoose_e(unsigned int n, unsigned int m, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(m > n) {
    cout << "GSL:Domain error!" << endl;
    exit(99);
  }
  else if(m == n || m == 0) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    gsl_sf_result nf;
    gsl_sf_result mf;
    gsl_sf_result nmmf;
    if(m*2 > n) m = n-m;
    gsl_sf_lnfact_e(n, &nf);
    gsl_sf_lnfact_e(m, &mf);
    gsl_sf_lnfact_e(n-m, &nmmf);
    result->val  = nf.val - mf.val - nmmf.val;
    result->err  = nf.err + mf.err + nmmf.err;
    result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
}

int gsl_sf_fact_e(const unsigned int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n < 18) {
    result->val = fact_table[n].f;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else if(n <= GSL_SF_FACT_NMAX){
    result->val = fact_table[n].f;
    result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    cout << "GSL:Overflow error!" << endl;
    exit(99);
  }
}
/* See: [Thompson, Atlas for Computing Mathematical Functions] */

static
int
delta(int ta, int tb, int tc, gsl_sf_result * d)
{
  gsl_sf_result f1, f2, f3, f4;
  int status = 0;
  status += gsl_sf_fact_e((ta + tb - tc)/2, &f1);
  status += gsl_sf_fact_e((ta + tc - tb)/2, &f2);
  status += gsl_sf_fact_e((tb + tc - ta)/2, &f3);
  status += gsl_sf_fact_e((ta + tb + tc)/2 + 1, &f4);
  if(status != 0) {
    cout << "GSL:Overflow error!" << endl;
    exit(99);
  }
  d->val = f1.val * f2.val * f3.val / f4.val;
  d->err = 4.0 * GSL_DBL_EPSILON * fabs(d->val);
  return GSL_SUCCESS;
}


static
int
triangle_selection_fails(int two_ja, int two_jb, int two_jc)
{
  /*
   * enough to check the triangle condition for one spin vs. the other two
   */
  return ( (two_jb < abs(two_ja - two_jc)) || (two_jb > two_ja + two_jc) ||
           GSL_IS_ODD(two_ja + two_jb + two_jc) );
}


static
int
m_selection_fails(int two_ja, int two_jb, int two_jc,
                  int two_ma, int two_mb, int two_mc)
{
  return (
         abs(two_ma) > two_ja 
      || abs(two_mb) > two_jb
      || abs(two_mc) > two_jc
      || GSL_IS_ODD(two_ja + two_ma)
      || GSL_IS_ODD(two_jb + two_mb)
      || GSL_IS_ODD(two_jc + two_mc)
      || (two_ma + two_mb + two_mc) != 0
          );
}


int
gsl_sf_exp_err_e(const double x, const double dx, gsl_sf_result * result)
{
  const double adx = fabs(dx);
  const double ex  = exp(x);
  const double edx = exp(adx);
  result->val  = ex;
  result->err  = ex * max(GSL_DBL_EPSILON, edx - 1.0/edx);
  result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
  return GSL_SUCCESS;
}

/*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/
int
gsl_sf_coupling_3j_e (int two_ja, int two_jb, int two_jc,
                      int two_ma, int two_mb, int two_mc,
                      gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(two_ja < 0 || two_jb < 0 || two_jc < 0) {
    cout << "GSL:Domain error!" << endl;
    exit(99);
  }
  else if (   triangle_selection_fails(two_ja, two_jb, two_jc)
           || m_selection_fails(two_ja, two_jb, two_jc, two_ma, two_mb, two_mc)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else if ( two_ma == 0 && two_mb == 0 && two_mc == 0
            && ((two_ja + two_jb + two_jc) % 4 == 2) ) {
    /* Special case for (ja jb jc; 0 0 0) = 0 when ja+jb+jc=odd */
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    int jca  = (-two_ja + two_jb + two_jc) / 2,
        jcb  = ( two_ja - two_jb + two_jc) / 2,
        jcc  = ( two_ja + two_jb - two_jc) / 2,
        jmma = ( two_ja - two_ma) / 2,
        jmmb = ( two_jb - two_mb) / 2,
        jmmc = ( two_jc - two_mc) / 2,
        jpma = ( two_ja + two_ma) / 2,
        jpmb = ( two_jb + two_mb) / 2,
        jpmc = ( two_jc + two_mc) / 2,
        jsum = ( two_ja + two_jb + two_jc) / 2,
        kmin = locMax3 (0, jpmb - jmmc, jmma - jpmc),
        kmax = locMin3 (jcc, jmma, jpmb),
        k, sign = GSL_IS_ODD (kmin - jpma + jmmb) ? -1 : 1,
        status = 0;
    double sum_pos = 0.0, sum_neg = 0.0, sum_err = 0.0;
    gsl_sf_result bc1, bc2, bc3, bcn1, bcn2, bcd1, bcd2, bcd3, bcd4, term, lnorm;

    status += gsl_sf_lnchoose_e (two_ja, jcc , &bcn1);
    status += gsl_sf_lnchoose_e (two_jb, jcc , &bcn2);
    status += gsl_sf_lnchoose_e (jsum+1, jcc , &bcd1);
    status += gsl_sf_lnchoose_e (two_ja, jmma, &bcd2);
    status += gsl_sf_lnchoose_e (two_jb, jmmb, &bcd3);
    status += gsl_sf_lnchoose_e (two_jc, jpmc, &bcd4);

    lnorm.val = 0.5 * (bcn1.val + bcn2.val - bcd1.val - bcd2.val - bcd3.val
                       - bcd4.val - log(two_jc + 1.0));
    lnorm.err = 0.5 * (bcn1.err + bcn2.err + bcd1.err + bcd2.err + bcd3.err
                       + bcd4.err + GSL_DBL_EPSILON * log(two_jc + 1.0));

    for (k = kmin; k <= kmax; k++) {
      status += gsl_sf_lnchoose_e (jcc, k, &bc1);
      status += gsl_sf_lnchoose_e (jcb, jmma - k, &bc2);
      status += gsl_sf_lnchoose_e (jca, jpmb - k, &bc3);
      status += gsl_sf_exp_err_e(bc1.val + bc2.val + bc3.val + lnorm.val,
                                 bc1.err + bc2.err + bc3.err + lnorm.err, 
                                 &term);
      if (sign < 0) {
        sum_neg += term.val;
      } else {
        sum_pos += term.val;
      }

      sum_err += term.err;

      sign = -sign;
    }
    
    result->val  = sum_pos - sum_neg;
    result->err  = sum_err;
    result->err += 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += 2.0 * GSL_DBL_EPSILON * (kmax - kmin) * fabs(result->val);

    return GSL_SUCCESS;
  }
}

int
gsl_sf_coupling_6j_e(int two_ja, int two_jb, int two_jc,
                     int two_jd, int two_je, int two_jf,
                     gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(   two_ja < 0 || two_jb < 0 || two_jc < 0
     || two_jd < 0 || two_je < 0 || two_jf < 0
     ) {
    cout << "GSL:Domain error!" << endl;
    exit(99);
  }
  else if(   triangle_selection_fails(two_ja, two_jb, two_jc)
          || triangle_selection_fails(two_ja, two_je, two_jf)
          || triangle_selection_fails(two_jb, two_jd, two_jf)
          || triangle_selection_fails(two_je, two_jd, two_jc)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    gsl_sf_result n1;
    gsl_sf_result d1, d2, d3, d4, d5, d6;
    double norm;
    int tk, tkmin, tkmax;
    double phase;
    double sum_pos = 0.0;
    double sum_neg = 0.0;
    double sumsq_err = 0.0;
    int status = 0;
    status += delta(two_ja, two_jb, two_jc, &d1);
    status += delta(two_ja, two_je, two_jf, &d2);
    status += delta(two_jb, two_jd, two_jf, &d3);
    status += delta(two_je, two_jd, two_jc, &d4);
    if(status != GSL_SUCCESS) {
      cout << "GSL:Overflow error!" << endl;
      exit(99);
    }
    norm = sqrt(d1.val) * sqrt(d2.val) * sqrt(d3.val) * sqrt(d4.val);
    
    tkmin = locMax3(0,
                   two_ja + two_jd - two_jc - two_jf,
                   two_jb + two_je - two_jc - two_jf);

    tkmax = locMin5(two_ja + two_jb + two_je + two_jd + 2,
                    two_ja + two_jb - two_jc,
                    two_je + two_jd - two_jc,
                    two_ja + two_je - two_jf,
                    two_jb + two_jd - two_jf);

    phase = GSL_IS_ODD((two_ja + two_jb + two_je + two_jd + tkmin)/2)
            ? -1.0
            :  1.0;

    for(tk=tkmin; tk<=tkmax; tk += 2) {
      double term;
      double term_err;
      gsl_sf_result den_1, den_2;
      gsl_sf_result d1_a, d1_b;
      status = 0;

      status += gsl_sf_fact_e((two_ja + two_jb + two_je + two_jd - tk)/2 + 1, &n1);
      status += gsl_sf_fact_e(tk/2, &d1_a);
      status += gsl_sf_fact_e((two_jc + two_jf - two_ja - two_jd + tk)/2, &d1_b);
      status += gsl_sf_fact_e((two_jc + two_jf - two_jb - two_je + tk)/2, &d2);
      status += gsl_sf_fact_e((two_ja + two_jb - two_jc - tk)/2, &d3);
      status += gsl_sf_fact_e((two_je + two_jd - two_jc - tk)/2, &d4);
      status += gsl_sf_fact_e((two_ja + two_je - two_jf - tk)/2, &d5);
      status += gsl_sf_fact_e((two_jb + two_jd - two_jf - tk)/2, &d6);

      if(status != GSL_SUCCESS) {
        cout << "GSL:Overflow error!" << endl;
        exit(99);
      }

      d1.val = d1_a.val * d1_b.val;
      d1.err = d1_a.err * fabs(d1_b.val) + fabs(d1_a.val) * d1_b.err;

      den_1.val  = d1.val*d2.val*d3.val;
      den_1.err  = d1.err * fabs(d2.val*d3.val);
      den_1.err += d2.err * fabs(d1.val*d3.val);
      den_1.err += d3.err * fabs(d1.val*d2.val);

      den_2.val  = d4.val*d5.val*d6.val;
      den_2.err  = d4.err * fabs(d5.val*d6.val);
      den_2.err += d5.err * fabs(d4.val*d6.val);
      den_2.err += d6.err * fabs(d4.val*d5.val);

      term  = phase * n1.val / den_1.val / den_2.val;
      phase = -phase;
      term_err  = n1.err / fabs(den_1.val) / fabs(den_2.val);
      term_err += fabs(term / den_1.val) * den_1.err;
      term_err += fabs(term / den_2.val) * den_2.err;

      if(term >= 0.0) {
        sum_pos += norm*term;
      }
      else {
        sum_neg -= norm*term;
      }

      sumsq_err += norm*norm * term_err*term_err;
    }

    result->val  = sum_pos - sum_neg;
    result->err  = 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += sqrt(sumsq_err / (0.5*(tkmax-tkmin)+1.0));
    result->err += 2.0 * GSL_DBL_EPSILON * (tkmax - tkmin + 2.0) * fabs(result->val);

    return GSL_SUCCESS;
  }
}

int
gsl_sf_coupling_9j_e(int two_ja, int two_jb, int two_jc,
                     int two_jd, int two_je, int two_jf,
                     int two_jg, int two_jh, int two_ji,
                     gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(   two_ja < 0 || two_jb < 0 || two_jc < 0
     || two_jd < 0 || two_je < 0 || two_jf < 0
     || two_jg < 0 || two_jh < 0 || two_ji < 0
     ) {
    cout << "GSL:Domain error!" << endl;
    exit(99);
  }
  else if(   triangle_selection_fails(two_ja, two_jb, two_jc)
          || triangle_selection_fails(two_jd, two_je, two_jf)
          || triangle_selection_fails(two_jg, two_jh, two_ji)
          || triangle_selection_fails(two_ja, two_jd, two_jg)
          || triangle_selection_fails(two_jb, two_je, two_jh)
          || triangle_selection_fails(two_jc, two_jf, two_ji)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    int tk;
    int tkmin = locMax3(abs(two_ja-two_ji), abs(two_jh-two_jd), abs(two_jb-two_jf));
    int tkmax = locMin3(two_ja + two_ji, two_jh + two_jd, two_jb + two_jf);
    double sum_pos = 0.0;
    double sum_neg = 0.0;
    double sumsq_err = 0.0;
    double phase;
    for(tk=tkmin; tk<=tkmax; tk += 2) {
      gsl_sf_result s1, s2, s3;
      double term;
      double term_err;
      int status = 0;

      status += gsl_sf_coupling_6j_e(two_ja, two_ji, tk,  two_jh, two_jd, two_jg,  &s1);
      status += gsl_sf_coupling_6j_e(two_jb, two_jf, tk,  two_jd, two_jh, two_je,  &s2);
      status += gsl_sf_coupling_6j_e(two_ja, two_ji, tk,  two_jf, two_jb, two_jc,  &s3);

      if(status != GSL_SUCCESS) {
        cout << "GSL:Overflow error!" << endl;
        exit(99);
      }
      term = s1.val * s2.val * s3.val;
      term_err  = s1.err * fabs(s2.val*s3.val);
      term_err += s2.err * fabs(s1.val*s3.val);
      term_err += s3.err * fabs(s1.val*s2.val);

      if(term >= 0.0) {
        sum_pos += (tk + 1) * term;
      }
      else {
        sum_neg -= (tk + 1) * term;
      }

      sumsq_err += ((tk+1) * term_err) * ((tk+1) * term_err);
    }

    phase = GSL_IS_ODD(tkmin) ? -1.0 : 1.0;

    result->val  = phase * (sum_pos - sum_neg);
    result->err  = 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += sqrt(sumsq_err / (0.5*(tkmax-tkmin)+1.0));
    result->err += 2.0 * GSL_DBL_EPSILON * (tkmax-tkmin + 2.0) * fabs(result->val);

    return GSL_SUCCESS;
  }
}

double gsl_sf_coupling_3j(int two_ja, int two_jb, int two_jc,
                          int two_ma, int two_mb, int two_mc)
{
  gsl_sf_result result;
  gsl_sf_coupling_3j_e(two_ja, two_jb, two_jc,
                       two_ma, two_mb, two_mc,
                       &result);
  return result.val;
}

double gsl_sf_coupling_6j(int two_ja, int two_jb, int two_jc,
                          int two_jd, int two_je, int two_jf)
{
  gsl_sf_result result;
  gsl_sf_coupling_6j_e(two_ja, two_jb, two_jc,
                       two_jd, two_je, two_jf,
                       &result);
  return result.val;
}

double gsl_sf_coupling_9j(int two_ja, int two_jb, int two_jc,
                          int two_jd, int two_je, int two_jf,
                          int two_jg, int two_jh, int two_ji)
{
  gsl_sf_result result;
  gsl_sf_coupling_9j_e(two_ja, two_jb, two_jc,
                       two_jd, two_je, two_jf,
                       two_jg, two_jh, two_ji,
                       &result);
  return result.val;
}