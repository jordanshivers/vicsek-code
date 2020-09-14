#include <math.h>
#include "nrutil.h"
#include <iostream>
#include "linmin.h"
#include <stdlib.h>
using std::string;
using std::endl;
using std::cout;
using std::ofstream;
#define EPS 1.0e-16
#define ITMAX 100000

#define FREEALL free_vector(xi,1,n);free_vector(h,1,n);free_vector(g,1,n);
//Here ITMAX is the maximum allowed number of iterations, while EPS is a small number to
//rectify the special case of converging to exactly zero function value.
void frprmn(double p[], int n, double ftol, int *iter, double *fret,
		double (*func)(double []), void (*dfunc)(double [], double []))
//Given a starting point p[1..n], Fletcher-Reeves-Polak-Ribiere minimization is performed on a
//function func, using its gradient as calculated by a routine dfunc. The convergence tolerance
//on the function value is input as ftol. Returned quantities are p (the location of the minimum),
//iter (the number of iterations that were performed), and fret (the minimum value of the
//function). The routine linmin is called to perform line minimizations.
{
//	void linmin(double p[], double xi[], int n, double *fret,     //linmin is slower than dlinmin
//			double (*func)(double []));

	void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),
				 void (*dfunc)(double [], double []));

	int j,its;
	double gg,gam,fp,dgg;
	double *g,*h,*xi;
	double res;

	g=vector_1(1,n);
	h=vector_1(1,n);
	xi=vector_1(1,n);

	fp=(*func)(p); //Initializations.
	(*dfunc)(p,xi);



	for (j=1;j<=n;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}

	for (its=1;its<=ITMAX;its++)
	{ //Loop over iterations.
		*iter=its;
//		linmin(p,xi,n,fret,func);//Next statement is the normal return:

		dlinmin(p, xi, n, fret, func, dfunc);

		if (2.0*fabs(*fret-fp) < ftol*(fabs(*fret)+fabs(fp)+EPS))  //Energy exit
		{
			FREEALL
			//cout <<"Good enough "<<fabs(*fret-fp)<<endl;
			return;
		}

		fp=*fret;
		(*dfunc)(p,xi);
		dgg=gg=0.0;
		res = 0.0;

		for (j=1;j<=n;j++)
		{
			gg += g[j]*g[j];
			//dgg += xi[j]*xi[j];  //This statement for Fletcher-Reeves.
			dgg += (xi[j]+g[j])*xi[j]; //This statement for Polak-Ribiere.
            // if (fabs(xi[j])>res) res=fabs(xi[j]); // Finding maximum component of the gradient
		}

   //      if (res<=ftol){
   //      	FREEALL
			// cout <<"Finished in "<< *iter << " iterations." << endl;
   //          return;
   //      }

		if (*iter%1000==0) {
			cout << "Number of iterations: " << *iter << endl;
			cout << "gradient magnitude " << sqrt(gg/n) << "|||||"<< "current energy "<<*fret<<endl; ;
			// cout << "Largest component of the gradient: " << res << endl;
		}


		// if (dgg == 0.0)
		// { //Unlikely. If gradient is exactly zero then
		// 	FREEALL //we are already done.
		// 	cout <<"Bingo! Zero Gradient! " << endl;
		// 	return;
		// }

		gam=dgg/gg;
		for (j=1;j<=n;j++)
		{
			g[j] = -xi[j];
			xi[j]=h[j]=g[j]+gam*h[j];
		}
	}
	cout <<"Not good! "<< endl;
	FREEALL
	return;
}



