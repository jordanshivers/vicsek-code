#define NR_END 1
#define FREE_ARG char*

double *vector_1(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
double *v;
//v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
v=new double[nh-nl+2];
return v;//-nl+NR_END;
}


void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{

free((FREE_ARG) (v+nl-NR_END));

}
