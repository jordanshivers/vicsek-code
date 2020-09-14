#include "nrutil.h"
//extern int ncom; //Defined in dlinmin.
//extern double *pcom,*xicom,(*nrfunc)(double []);
extern void (*nrdfun)(double [], double []);
double df1dim(double x)
{
int j;
double df1=0.0;
double *xt,*df;
xt=vector_1(1,ncom);
df=vector_1(1,ncom);
for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
(*nrdfun)(xt,df);
for (j=1;j<=ncom;j++) df1 += df[j]*xicom[j];
free_vector(df,1,ncom);
free_vector(xt,1,ncom);
return df1;
}
