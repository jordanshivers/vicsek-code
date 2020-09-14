// Written by Jordan Shivers 
// MacKintosh Group, Rice University
// 
// Vicsek model
// 
// Dependencies
// 
// Compilation:
// To compile locally, run "make all". 

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "include/Eigen/Core"
#include "include/Eigen/Dense"
#include <random>
#include <list>
#include <omp.h>
#include <chrono>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

#define tiny 1.0e-16
#define NUMTHREADS 4

std::uniform_real_distribution<> dist(0, 1);

int n_ran_calls = 0, nbins;
bool print_birds, print_state;
double r_cutoff;
std::vector<std::vector<std::vector<std::list<int> > > > bins;

#include "functions.h"

std::string out_dir;

int main(int argc, char* argv[]) {

    omp_set_num_threads(NUMTHREADS);

    int N, seed, n_T_steps, Tnum;
    double rho, L, dt, v, tmax, t_equil, rc, J, minT, maxT;

    // // // Parameters
    string dir = "./out";
    seed = 1;
    N =  2000; //number of birds
    rho = 1; // number density of birds
    dt = 0.1; // time step
    v = 1; // magnitude of bird velocity
    tmax = 20; // max time to simulate
    t_equil = 1; // equilibration time
    rc = 1; // characteristic interaction distance
    J = 0.1; // interaction strength
    n_T_steps = 50;
    Tnum = 0;
    minT = 1e-2;
    maxT = 100;

    //  // // // Parameters
    // string dir = argv[1];
    // seed = atoi(argv[2]);
    // N = atoi(argv[3]); //number of birds
    // rho = atof(argv[4]); // number density of birds
    // dt = atof(argv[5]); // time step
    // v = atof(argv[6]); // magnitude of bird velocity
    // tmax = atof(argv[7]); // max time to simulate
    // t_equil = atof(argv[8]); // equilibration time
    // rc  = atof(argv[9]); // characteristic interaction distance
    // J  = atof(argv[10]); // interaction strength
    // n_T_steps = atoi(argv[11]);
    // Tnum = atoi(argv[12]);
    // minT = atof(argv[13]);
    // maxT = atof(argv[14]);

    bool print_time_series = false;
    print_birds = false;
    print_state = true;

    L = pow(N/rho,1.0/3.0); // box side length
    r_cutoff = 10; // cutoff radius for interactions
    nbins = ceil(L/r_cutoff);
    nbins = max(3, nbins); // # x and y bins

    cout << "Nbins = " << nbins << endl;



    double *T = new double[n_T_steps];

    for (int i = 0; i < n_T_steps; i++) {
        T[i] = pow(10.0,log10(minT) + (log10(maxT) - log10(minT))/(n_T_steps-1)*i);
        // cout << "T[i] = " << T[i] << endl;
    }

    // https://stackoverflow.com/questions/1340729/how-do-you-generate-a-random-double-uniformly-distributed-between-0-and-1-from-c/1340762
    std::mt19937 gen(seed);




    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


    simulation_3D(dt, T[Tnum], rc, N, L, v, tmax, J, seed, t_equil, Tnum, gen, dir, print_time_series);
    // simulation_3D(dt, 0.1, rc, N, L, v, tmax, J, seed, t_equil, Tnum, gen, dir, print_time_series);


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    return 0;
}