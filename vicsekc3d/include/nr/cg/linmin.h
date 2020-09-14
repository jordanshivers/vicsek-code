#include "nrutil.h"
#include "brent.h"
#include "mnbrak.h"
#include "f1dim.h"
#define TOL 1.0e-12 //Tolerance passed to brent.
//int ncom; //Global variables communicate with f1dim.
//double *pcom,*xicom,(*nrfunc)(double []);
void linmin(double p[], double xi[], int n, double *fret, double (*func)(double []))
//Given an n-dimensional point p[1..n] and an n-dimensional direction xi[1..n], moves and
//resets p to where the function func(p) takes on a minimum along the direction xi from p,
//and replaces xi by the actual vector displacement that p was moved. Also returns as fret
//the value of func at the returned location p. This is actually all accomplished by calling the
//routines mnbrak and brent.
{
double brent(double ax, double bx, double cx, double (*f)(double), double tol, double *xmin);
double f1dim(double x);
void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc, double (*func)(double));
int j;
double xx,xmin,fx,fb,fa,bx,ax;
extern int ncom;

ncom=n; //Define the global variables.
//pcom = new double(n+1);
pcom=vector_1(1,n);
//xicom = new double(n+1);
xicom=vector_1(1,n);
nrfunc=func;
for (j=1;j<=n;j++) {
	pcom[j]=p[j];
	xicom[j]=xi[j];
}
ax=0.0; //Initial guess for brackets.
xx=1.0;
mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
*fret=brent(ax,xx,bx,f1dim,TOL,&xmin);
for (j=1;j<=n;j++) { //Construct the vector results to return.
	xi[j] *= xmin;
	p[j] += xi[j];
}

free_vector(xicom,1,n);
free_vector(pcom,1,n);
}



void linminTime(double p[], double xi[], int n, double *fret, double (*func)(double []),double &dt)
//Given an n-dimensional point p[1..n] and an n-dimensional direction xi[1..n], moves and
//resets p to where the function func(p) takes on a minimum along the direction xi from p,
//and replaces xi by the actual vector displacement that p was moved. Also returns as fret
//the value of func at the returned location p. This is actually all accomplished by calling the
//routines mnbrak and brent.
{
	double brent(double ax, double bx, double cx, double (*f)(double), double tol, double *xmin);
	double f1dim(double x);
	void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc, double (*func)(double));
	int j;
	double xx,xmin,fx,fb,fa,bx,ax;
extern int ncom;


	ncom=n; //Define the global variables.
	pcom=vector_1(1,n);
	//double pcom[1+n],xicom[1+n];
	xicom=vector_1(1,n);
	nrfunc=func;

	for (j=1;j<=n;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0; //Initial guess for brackets.
	xx=1.0;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
	*fret=brent(ax,xx,bx,f1dim,TOL,&xmin);

	for (j=1;j<=n;j++) { //Construct the vector results to return.
		xi[j] *= xmin;
		p[j] += xi[j];
	}

	free_vector(xicom,1,n);
	free_vector(pcom,1,n);
	dt=-xmin;
}




