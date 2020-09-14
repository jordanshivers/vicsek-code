#include "nrutil.h"
#define TOL 1.0e-12 //Tolerance passed to dbrent.
//#define TOL 1e-4 //Tolerance passed to dbrent.
int ncom; //Global variables communicate with df1dim.

double *pcom,*xicom,(*nrfunc)(double []);
void (*nrdfun)(double [], double []);
void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),
void (*dfunc)(double [], double []))
//Given an n-dimensional point p[1..n] and an n-dimensional direction xi[1..n], moves and
//resets p to where the function func(p) takes on a minimum along the direction xi from p,
//and replaces xi by the actual vector displacement that p was moved. Also returns as fret
//the value of func at the returned location p. This is actually all accomplished by calling the
//routines mnbrak and dbrent.
{

double dbrent(double ax, double bx, double cx,
double (*f)(double), double (*df)(double), double tol, double *xmin);
double f1dim(double x);
double df1dim(double x);
void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
double *fc, double (*func)(double));
int j;
double xx,xmin,fx,fb,fa,bx,ax;
ncom=n; //Define the global variables.

pcom=vector_1(1,n);

xicom=vector_1(1,n);

nrfunc=func;
nrdfun=dfunc;
for (j=1;j<=n;j++) {
pcom[j]=p[j];
xicom[j]=xi[j];
}
ax=0.0; //Initial guess for brackets.
xx=1.0;
mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
*fret=dbrent(ax,xx,bx,f1dim,df1dim,TOL,&xmin);
for (j=1;j<=n;j++) { //Construct the vector results to return.
xi[j] *= xmin;
p[j] += xi[j];
}
free_vector(xicom,1,n);
free_vector(pcom,1,n);
}
