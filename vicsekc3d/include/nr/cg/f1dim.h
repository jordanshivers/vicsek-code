#include "nrutil.h"
//extern int ncom; //Defined in linmin.
extern double *pcom,*xicom,(*nrfunc)(double []);
double f1dim(double x)
//Must accompany linmin.
{
int j;
double f,*xt;
xt=vector_1(1,ncom);
//double xt[1+n];
for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
f=(*nrfunc)(xt);
free_vector(xt,1,ncom);
return f;
}
