
double randDoub(mt19937& generator) {
    double ans = dist(generator); n_ran_calls++;
    return ans;
}

std::string doubleToString(double d)
{
    std::ostringstream ss;
    ss << d;
    return ss.str();
}

std::string intToString(int d)
{
    std::ostringstream ss;
    ss << d;
    return ss.str();
}

void CopyArray(double *source, double *copy, int length)
{
    for (int i = 0; i < length; i++){
        copy[i] = source[i];
    }
    return;
}

double normRand(double m, double s, mt19937& generator)  /* normal random variate generator */
{                       /* mean m, standard deviation s */
    // copied from https://www.taygeta.com/random/boxmuller.html

    double x1, x2, w, y1;

    do {
        x1 = 2.0 * randDoub(generator) - 1.0;
        x2 = 2.0 * randDoub(generator) - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;

    return( m + y1 * s );
}

double dot(double ux, double uy, double uz, double vx, double vy, double vz) {
    // dot product of vectors u and v
    return ux*vx + uy*vy + uz*vz;
}


double angle3D(double ux, double uy, double uz, double vx, double vy, double vz) {
    // angle between vectors u and v (in radians)
    double temp = dot(ux, uy, uz, vx, vy, vz)/sqrt(dot(ux,uy,uz,ux,uy,uz)*dot(vx,vy,vz,vx,vy,vz));
    if (temp > 1.0) temp = 1.0;
    if (temp < -1.0) temp = -1.0;
    return acos(temp);
}

double prob3d(double phi, double z0, double sigma) {

    // % modified from solange's code
    // function p = prob3D(phi,z0,s)
    // a = z0.*cos(phi)./(s*sqrt(2)); % large number for small T, dt
    // % p = 2*sin(phi)./ (sqrt(pi)* 2* s^2) .* exp(-z0.^2.* sin(phi).^2./(2* s.^2)).* 
    // % (-a.* s.^2 .* exp(-a.^2)+ s.^2*sqrt(pi)./2.*erfc(-a) + z0.* cos(phi).* sqrt(2*s.^2).* exp(-a.^2) + z0.^2.* cos(phi).^2 .* sqrt(pi)./2 .* erfc(-a) );
    // p = 2.*sin(phi)./s .* exp(-z0.^2.* sin(phi).^2./(2.* s.^2)).* ( ...
    //     z0.* cos(phi).* exp(-a.^2)./(2*sqrt(2*pi)) + ...
    //     (s + z0.^2.* cos(phi).^2 ./s) .* erfc(-a)/4 );
    // end

    double a = z0*cos(phi)/(sigma*sqrt(2.0));    
    double p = 2.0*sin(phi)/sigma*exp(-z0*z0*sin(phi)*sin(phi)/(2.0*sigma*sigma))*(z0*cos(phi)*exp(-a*a)/(2.0*sqrt(2.0*M_PI))+(sigma+z0*z0*cos(phi)*cos(phi)/sigma)*erfc(-a)/4.0);

    return p;
}

double mean(double *x, int n) {
    // returns average of an array of n doubles
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum/n;
}

double compute_delS12_current(std::vector<double> pforward, std::vector<double> pbackward, int N) {

    double logpf, logpb;
    double sum = 0;

    for (int i = 0; i < N; i++) {
        logpf = log(pforward[i]);
        logpb = log(pbackward[i]);

        if (std::isinf(logpb)) { // case logpb = -inf
            logpb = log(1e-16);
        }

        if (std::isinf(logpf)) { // case logpf = -inf
            logpf = log(1e-16);
        }

        if (std::isnan(logpf) or std::isnan(logpb)) {
            // ignore this contribution
        }
        else {
            sum += logpf - logpb;

        }
    }

    return sum;
}
int xbin(double x, double L) {
    // side length is N
    // cutoff length is r_cutoff
    // minimum integer subdivision greater than or equal to r_cutoff is ceil(L/r_cutoff)

    return (int) ((x/L)*nbins);
}

int ybin(double y, double L) {
    // side length is N
    // cutoff length is r_cutoff
    // minimum integer subdivision greater than or equal to r_cutoff is ceil(L/r_cutoff)

    return (int) ((y/L)*nbins);
}

int zbin(double z, double L) {
    // side length is N
    // cutoff length is r_cutoff
    // minimum integer subdivision greater than or equal to r_cutoff is ceil(L/r_cutoff)

    return (int) ((z/L)*nbins);
}

void bin_birds(VectorXd x, VectorXd y, VectorXd z, int N, double L) {

    // side length is N
    // cutoff length is r_cutoff
    // minimum integer subdivision greater than or equal to r_cutoff is ceil(L/r_cutoff)
    
    int this_xbin, this_ybin, this_zbin;

    bins.clear();
    bins.resize(nbins);
    for (int i = 0; i < nbins; i++) {
        bins[i].resize(nbins);
        for (int j = 0; j < nbins; j++) {
            bins[i][j].resize(nbins);
        }
    }

    for (int i = 0; i < N; i++) {

        this_xbin = xbin(x(i), L);
        this_ybin = ybin(y(i), L);
        this_zbin = zbin(z(i), L);
        bins[this_xbin][this_ybin][this_zbin].push_back(i);

    }
}

void simulation_3D(double dt, double T, double rc, int N, double L, double v, double tmax, double J, double seed, double t_equil, int Tnum, mt19937& generator, string dir, bool print_time_series)
{

    double dt_sim = dt;
    double d = 3.0; // no. of dimensions
    double dt_record = 100;
    if (remainder(dt_record, dt) > 1e-10) {
        throw std::invalid_argument( "error: dt_record needs to be a multiple of dt" );
    }

    int record_every_n = dt_record/dt_sim;
    bool finished_already = false;

    int n_timesteps = ceil(tmax/dt);
    int tnum_current = 0, tnum_start = 0, count_recorded = 0;

    double *polarization;
    double *delS12;
    double *analytical_S_integrand;

    if (print_time_series) {
        polarization = new double[record_every_n];
        delS12 = new double[record_every_n];
        analytical_S_integrand = new double[record_every_n];
    }
    
    double sum_polarization = 0, sum_delS12 = 0;
    double mean_polarization = 0, mean_delS12_over_Ndt = 0;

    // set initial positions
    VectorXd x(N);
    VectorXd y(N);
    VectorXd z(N);
    VectorXd sum_dt_Jij_sx(N);
    VectorXd sum_dt_Jij_sy(N);
    VectorXd sum_dt_Jij_sz(N);
    MatrixXd Jij(N,N);    
    MatrixXd Jij_0(N,N);    
    MatrixXd Jij_prev(N,N);
    MatrixXd dJij_dt(N,N);    
    MatrixXd si_dot_sj(N,N); 
    MatrixXd si_dot_sj_0(N,N); 
    double rx, ry, rz, r2, rij, rxRound, ryRound, rzRound;

    MatrixXd s(N,3);
    MatrixXd s_prev(N,3);
    MatrixXd s_det(N,3);
    MatrixXd s_r(N,3);
    MatrixXd s_prev_r(N,3);
    MatrixXd s_det_r(N,3);
    MatrixXd eta(N,3);
    double mag, theta, sumsx, sumsy, sumsz, ux, uy, uz, vx, vy, vz, t, delS12_current, polarization_current;
    double sx, sy, sz, analytical_S_firstterm = 0, analytical_S_integrand_prev = 0, analytical_S_integrand_this, Jij_times_sidotsj_sum_t0, Jij_times_sidotsj_sum, analytical_S_second_term;
    std::vector<double> pforward(N);
    std::vector<double> pbackward(N);
    double *theta_noise = new double[N];
    double *l = new double[N];
    int this_xbin, this_ybin, this_zbin, that_xbin, that_ybin, that_zbin, j;
    std::vector<std::vector<int> > neighbors; 


    // Check state
    ifstream statein ((dir + "/seed" + intToString(seed) + "_Tnum" + intToString(Tnum) + "_state.txt").c_str());
    if (statein.is_open()) {

        // read it all in
        // Output state
        statein.precision(17);
        statein >> tnum_current;
        statein >> n_ran_calls;
        statein >> count_recorded;        
        statein >> mean_polarization;
        statein >> mean_delS12_over_Ndt;
        statein >> analytical_S_firstterm;
        statein >> analytical_S_integrand_prev;
        statein >> Jij_times_sidotsj_sum_t0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                statein >> Jij(i,j);
            }
        }

        for (int i = 0; i < N; i++) {
            statein >> x(i);
            statein >> y(i);
            statein >> z(i);
            statein >> s(i,0);
            statein >> s(i,1);
            statein >> s(i,2);
        }

        statein.close();

        tnum_start = tnum_current;

        // call the random number generator n_ran_calls times
        for (int i = 0; i < n_ran_calls; i++) {
            dist(generator);
        }

        sum_polarization = mean_polarization*tnum_current;
        sum_delS12 = mean_delS12_over_Ndt*tnum_current*N*dt;
    }

    else {

        cout << "No state file exists! Starting from time 0" << endl;

        for (int i = 0; i < N; i++) {
            x(i) = L*randDoub(generator);
            y(i) = L*randDoub(generator);
            z(i) = L*randDoub(generator);
        }


        for (int i = 0; i < N; i++) {
   
            sx = normRand(0.0, 1.0, generator);
            sy = normRand(0.0, 1.0, generator);
            sz = normRand(0.0, 1.0, generator);
            mag = sqrt(sx*sx+sy*sy+sz*sz);
            s(i,0) = sx/mag;
            s(i,1) = sy/mag;
            s(i,2) = sz/mag;
        }

        dt = 0.05; // use larger timestep for equilibration

        // equilibration
        cout << "equilibrating for " << ceil(t_equil/dt) << " timesteps" << endl;

        for (int tnum = 0; tnum < ceil(t_equil/dt); tnum++) {

            if (tnum%1000 == 0) {
                cout << "equilibrating... " << tnum*1.0/ceil(t_equil/dt)*100.0 << "%" << endl;
            }

            bin_birds(x, y, z, N, L);


            Jij = MatrixXd::Zero(N,N);
            sum_dt_Jij_sx = VectorXd::Zero(N);
            sum_dt_Jij_sy = VectorXd::Zero(N);
            sum_dt_Jij_sz = VectorXd::Zero(N);

            double sum = 0;

            #pragma omp parallel shared(x, y, z, Jij, sum_dt_Jij_sx, sum_dt_Jij_sy, sum_dt_Jij_sz) private(j, this_xbin, this_ybin, this_zbin, that_xbin, that_ybin, that_zbin, rx, ry, rz, rxRound, ryRound, rzRound, r2, rij)
            {

                #pragma omp for
                for (int i = 0; i < N; i++) {

                    // cout << i << endl;

                    // consider bird i; what are its bins?
                    this_xbin = xbin(x(i), L);
                    this_ybin = ybin(y(i), L);
                    this_zbin = zbin(z(i), L);

                    // what are the 27 neighboring bins?
                    std::vector<std::vector<int> > neighborBins_this; 
                    neighborBins_this.resize(27);
                    for (int b = 0; b < 27; b++) {
                        neighborBins_this[b].resize(3);
                    }

                    neighborBins_this[0][0] = (nbins+this_xbin-1)%nbins;  
                    neighborBins_this[1][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[2][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[3][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[4][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[5][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[6][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[7][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[8][0] = (nbins+this_xbin-1)%nbins; 
                    neighborBins_this[9][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[10][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[11][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[12][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[13][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[14][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[15][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[16][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[17][0] = (nbins+this_xbin)%nbins; 
                    neighborBins_this[18][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[19][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[20][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[21][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[22][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[23][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[24][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[25][0] = (nbins+this_xbin+1)%nbins; 
                    neighborBins_this[26][0] = (nbins+this_xbin+1)%nbins; 

                    neighborBins_this[0][1] = (nbins+this_ybin-1)%nbins;  
                    neighborBins_this[1][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[2][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[3][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[4][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[5][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[6][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[7][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[8][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[9][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[10][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[11][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[12][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[13][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[14][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[15][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[16][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[17][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[18][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[19][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[20][1] = (nbins+this_ybin-1)%nbins; 
                    neighborBins_this[21][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[22][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[23][1] = (nbins+this_ybin)%nbins; 
                    neighborBins_this[24][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[25][1] = (nbins+this_ybin+1)%nbins; 
                    neighborBins_this[26][1] = (nbins+this_ybin+1)%nbins; 

                    neighborBins_this[0][2] = (nbins+this_zbin-1)%nbins;  
                    neighborBins_this[1][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[2][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[3][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[4][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[5][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[6][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[7][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[8][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[9][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[10][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[11][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[12][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[13][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[14][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[15][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[16][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[17][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[18][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[19][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[20][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[21][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[22][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[23][2] = (nbins+this_zbin+1)%nbins; 
                    neighborBins_this[24][2] = (nbins+this_zbin-1)%nbins; 
                    neighborBins_this[25][2] = (nbins+this_zbin)%nbins; 
                    neighborBins_this[26][2] = (nbins+this_zbin+1)%nbins; 

                    // iterate through neighbor bins and compute distance between bird i and neighbors
                    for (int bin = 0; bin < 27; bin ++) {
                        that_xbin = neighborBins_this[bin][0];
                        that_ybin = neighborBins_this[bin][1];
                        that_zbin = neighborBins_this[bin][2];

                        for (std::list<int>::iterator it = bins[that_xbin][that_ybin][that_zbin].begin(); it != bins[that_xbin][that_ybin][that_zbin].end(); ++it){
                            j = *it;

                            if (i != j) {

                                rx = x(j) - x(i);
                                ry = y(j) - y(i);
                                rz = z(j) - z(i);

                                // take into account periodic boundary conditions
                                rxRound = L*(round(rx/L));
                                ryRound = L*(round(ry/L));
                                rzRound = L*(round(rz/L));
                                rx = rx - rxRound;
                                ry = ry - ryRound;
                                rz = rz - rzRound;
                                r2 = rx*rx + ry*ry + rz*rz;
                                rij = sqrt(r2);

                                if (rij < r_cutoff) {
                                    Jij(i,j) = J*exp(-rij/rc);

                                    sum_dt_Jij_sx(i) += dt*Jij(i,j)*s(j,0);
                                    sum_dt_Jij_sy(i) += dt*Jij(i,j)*s(j,1);
                                    sum_dt_Jij_sz(i) += dt*Jij(i,j)*s(j,2);
                                }
                            }
                        }
                    }

                }
            
            }

            #pragma omp parallel shared(eta, s, x, y, z, sum_dt_Jij_sx, sum_dt_Jij_sy, sum_dt_Jij_sz) private(mag)
            {

                #pragma omp for
                for (int i = 0; i < N; i++) {

                    eta(i, 0) = normRand(0.0, 1.0, generator);
                    eta(i, 1) = normRand(0.0, 1.0, generator);
                    eta(i, 2) = normRand(0.0, 1.0, generator);
                    s(i,0) = s(i,0) + sum_dt_Jij_sx(i) + sqrt(2.0*T*dt)*eta(i, 0);
                    s(i,1) = s(i,1) + sum_dt_Jij_sy(i) + sqrt(2.0*T*dt)*eta(i, 1);
                    s(i,2) = s(i,2) + sum_dt_Jij_sz(i) + sqrt(2.0*T*dt)*eta(i, 2);
                    mag = sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));
                    s(i, 0) = s(i,0)/mag;
                    s(i, 1) = s(i,1)/mag;
                    s(i, 2) = s(i,2)/mag;
                    x(i) = x(i) + v*s(i,0)*dt;
                    y(i) = y(i) + v*s(i,1)*dt;
                    z(i) = z(i) + v*s(i,2)*dt;
                    if (x(i) >= L) x(i) = x(i) - L;
                    if (x(i) < 0) x(i) = x(i) + L;
                    if (y(i) >= L) y(i) = y(i) - L;
                    if (y(i) < 0) y(i) = y(i) + L;
                    if (z(i) >= L) z(i) = z(i) - L;
                    if (z(i) < 0) z(i) = z(i) + L;
                }
            }


            s_prev = s;
            Jij_prev = Jij;

        }
    }

    dt = dt_sim;

    cout << "simulating for " << n_timesteps << " timesteps" << endl;

    if (tnum_current > n_timesteps) {
        finished_already = true;
        cout << "finished already!" << endl;
    }


    for (int tnum = tnum_current; tnum <= n_timesteps; tnum++) {

        t = tnum*dt;


        if (tnum%1000 == 0) {
            cout << "simulating... " << tnum*1.0/n_timesteps*100.0 << "%" << endl;
        }

        s_prev = s;
        Jij_prev = Jij;

        bin_birds(x, y, z, N, L);

        Jij = MatrixXd::Zero(N,N);
        dJij_dt = MatrixXd::Zero(N,N);
        sum_dt_Jij_sx = VectorXd::Zero(N);
        sum_dt_Jij_sy = VectorXd::Zero(N);
        sum_dt_Jij_sz = VectorXd::Zero(N);
        neighbors.clear();
        neighbors.resize(N);


        #pragma omp parallel shared(x, y, z, Jij, sum_dt_Jij_sx, sum_dt_Jij_sy, sum_dt_Jij_sz) private(j, this_xbin, this_ybin, this_zbin, that_xbin, that_ybin, that_zbin, rx, ry, rz, rxRound, ryRound, rzRound, r2, rij)
        {

            #pragma omp for
            for (int i = 0; i < N; i++) {

                // cout << i << endl;

                // consider bird i; what are its bins?
                this_xbin = xbin(x(i), L);
                this_ybin = ybin(y(i), L);
                this_zbin = zbin(z(i), L);

                // what are the 27 neighboring bins?
                std::vector<std::vector<int> > neighborBins_this; 
                neighborBins_this.resize(27);
                for (int b = 0; b < 27; b++) {
                    neighborBins_this[b].resize(3);
                }

                neighborBins_this[0][0] = (nbins+this_xbin-1)%nbins;  
                neighborBins_this[1][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[2][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[3][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[4][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[5][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[6][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[7][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[8][0] = (nbins+this_xbin-1)%nbins; 
                neighborBins_this[9][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[10][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[11][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[12][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[13][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[14][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[15][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[16][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[17][0] = (nbins+this_xbin)%nbins; 
                neighborBins_this[18][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[19][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[20][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[21][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[22][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[23][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[24][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[25][0] = (nbins+this_xbin+1)%nbins; 
                neighborBins_this[26][0] = (nbins+this_xbin+1)%nbins; 

                neighborBins_this[0][1] = (nbins+this_ybin-1)%nbins;  
                neighborBins_this[1][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[2][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[3][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[4][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[5][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[6][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[7][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[8][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[9][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[10][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[11][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[12][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[13][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[14][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[15][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[16][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[17][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[18][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[19][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[20][1] = (nbins+this_ybin-1)%nbins; 
                neighborBins_this[21][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[22][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[23][1] = (nbins+this_ybin)%nbins; 
                neighborBins_this[24][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[25][1] = (nbins+this_ybin+1)%nbins; 
                neighborBins_this[26][1] = (nbins+this_ybin+1)%nbins; 

                neighborBins_this[0][2] = (nbins+this_zbin-1)%nbins;  
                neighborBins_this[1][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[2][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[3][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[4][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[5][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[6][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[7][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[8][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[9][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[10][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[11][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[12][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[13][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[14][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[15][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[16][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[17][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[18][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[19][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[20][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[21][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[22][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[23][2] = (nbins+this_zbin+1)%nbins; 
                neighborBins_this[24][2] = (nbins+this_zbin-1)%nbins; 
                neighborBins_this[25][2] = (nbins+this_zbin)%nbins; 
                neighborBins_this[26][2] = (nbins+this_zbin+1)%nbins; 

                // iterate through neighbor bins and compute distance between bird i and neighbors
                for (int bin = 0; bin < 27; bin ++) {
                    that_xbin = neighborBins_this[bin][0];
                    that_ybin = neighborBins_this[bin][1];
                    that_zbin = neighborBins_this[bin][2];

                    for (std::list<int>::iterator it = bins[that_xbin][that_ybin][that_zbin].begin(); it != bins[that_xbin][that_ybin][that_zbin].end(); ++it){
                        j = *it;

                        if (i != j) {

                            rx = x(j) - x(i);
                            ry = y(j) - y(i);
                            rz = z(j) - z(i);

                            // take into account periodic boundary conditions
                            rxRound = L*(round(rx/L));
                            ryRound = L*(round(ry/L));
                            rzRound = L*(round(rz/L));
                            rx = rx - rxRound;
                            ry = ry - ryRound;
                            rz = rz - rzRound;
                            r2 = rx*rx + ry*ry + rz*rz;
                            rij = sqrt(r2);

                            if (rij < r_cutoff) {
                                neighbors[i].push_back(j);
                                Jij(i,j) = J*exp(-rij/rc);

                                sum_dt_Jij_sx(i) += dt*Jij(i,j)*s(j,0);
                                sum_dt_Jij_sy(i) += dt*Jij(i,j)*s(j,1);
                                sum_dt_Jij_sz(i) += dt*Jij(i,j)*s(j,2);
                            }
                        }
                    }
                }

            }
        
        }

        sumsx = 0;
        sumsy = 0;
        sumsz = 0;



        #pragma omp parallel shared(eta, s, x, y, z, sum_dt_Jij_sx, sum_dt_Jij_sy, sum_dt_Jij_sz) private(mag)
        {

            #pragma omp for
            for (int i = 0; i < N; i++) {

                eta(i, 0) = normRand(0.0, 1.0, generator);
                eta(i, 1) = normRand(0.0, 1.0, generator);
                eta(i, 2) = normRand(0.0, 1.0, generator);
                s(i,0) = s(i,0) + sum_dt_Jij_sx(i) + sqrt(2.0*T*dt)*eta(i, 0);
                s(i,1) = s(i,1) + sum_dt_Jij_sy(i) + sqrt(2.0*T*dt)*eta(i, 1);
                s(i,2) = s(i,2) + sum_dt_Jij_sz(i) + sqrt(2.0*T*dt)*eta(i, 2);
                mag = sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));
                s(i, 0) = s(i,0)/mag;
                s(i, 1) = s(i,1)/mag;
                s(i, 2) = s(i,2)/mag;
                x(i) = x(i) + v*s(i,0)*dt;
                y(i) = y(i) + v*s(i,1)*dt;
                z(i) = z(i) + v*s(i,2)*dt;
                if (x(i) >= L) x(i) = x(i) - L;
                if (x(i) < 0) x(i) = x(i) + L;
                if (y(i) >= L) y(i) = y(i) - L;
                if (y(i) < 0) y(i) = y(i) + L;
                if (z(i) >= L) z(i) = z(i) - L;
                if (z(i) < 0) z(i) = z(i) + L;
                sumsx += s(i,0);
                sumsy += s(i,1);
                sumsz += s(i,2);
            }
        }

        polarization_current = sqrt(sumsx*sumsx + sumsy*sumsy + sumsz*sumsz)/N;

        s_det = MatrixXd::Zero(N,3);
        s_prev_r = -1.0*s_prev;
        s_r = -1.0*s;
        s_det_r = MatrixXd::Zero(N,3);

        delS12_current = 0.0;
        
        analytical_S_integrand_this = 0;

        #pragma omp parallel reduction(+:analytical_S_integrand_this) shared(theta_noise, l, pforward, pbackward, s_prev, s_det, s, s_r, s_det_r, s_prev_r) private(ux, uy, uz, vx, vy, vz, j)
        {
            #pragma omp for
            for (int i = 0; i < N; i++) {

                // calculate s_det and s_det_r (slightly faster to do it this way than directly multiplying Jij and s)

                for (int a = 0; a < neighbors[i].size(); a++) {
                    j = neighbors[i][a];
                    s_det(i,0) += dt*Jij(i,j)*s_prev(j,0);
                    s_det(i,1) += dt*Jij(i,j)*s_prev(j,1);
                    s_det(i,2) += dt*Jij(i,j)*s_prev(j,2);
                    s_det_r(i,0) += dt*Jij(i,j)*s_prev_r(j,0);
                    s_det_r(i,1) += dt*Jij(i,j)*s_prev_r(j,1);
                    s_det_r(i,2) += dt*Jij(i,j)*s_prev_r(j,2);

                    // if these were neighbors before and are neighbors now
                    if ((Jij(i,j) > tiny) and (Jij_prev(i,j) > tiny)) {
                        analytical_S_integrand_this += (s(i,0)*s(j,0) + s(i,1)*s(j,1) + s(i,2)*s(j,2))*(Jij(i,j) - Jij_prev(i,j))/dt/(2.0*T*(d-1.0));
                    }
                }

                ux = s_prev(i,0) + s_det(i,0);
                uy = s_prev(i,1) + s_det(i,1);
                uz = s_prev(i,2) + s_det(i,2);
                vx = s(i,0);
                vy = s(i,1);
                vz = s(i,2);
                theta_noise[i] = angle3D(ux, uy, uz, vx, vy, vz);
                l[i] = sqrt(dot(ux,uy,uz,ux,uy,uz));
                pforward[i] = prob3d(theta_noise[i], l[i], sqrt(2*T*dt));

                ux = s_r(i,0) + s_det_r(i,0);
                uy = s_r(i,1) + s_det_r(i,1);
                uz = s_r(i,2) + s_det_r(i,2);
                vx = s_prev_r(i,0);
                vy = s_prev_r(i,1);
                vz = s_prev_r(i,2);
                theta_noise[i] = angle3D(ux, uy, uz, vx, vy, vz);
                l[i] = sqrt(dot(ux,uy,uz,ux,uy,uz));
                pbackward[i] = prob3d(theta_noise[i], l[i], sqrt(2*T*dt));


            }
        }



        analytical_S_firstterm += dt*(analytical_S_integrand_this + analytical_S_integrand_prev)/2.0; // trapezoidal rule
        analytical_S_integrand_prev = analytical_S_integrand_this;
        if (print_time_series) {
            analytical_S_integrand[tnum-tnum_current] = analytical_S_integrand_this;
        }

        delS12_current = compute_delS12_current(pforward, pbackward, N);

        sum_polarization += polarization_current;
        sum_delS12 += delS12_current;

        if (print_time_series) {
            delS12[tnum-tnum_current] = delS12_current;
            polarization[tnum-tnum_current] = polarization_current;
        }

        if ((tnum > 0) and (tnum % record_every_n == 0)) {
            // cout << "t = " << t << endl;

            tnum_current = tnum + 1;

            cout << "Mean(P) = " << sum_polarization/tnum_current << ". Mean(delS12)/N/dt = " << sum_delS12/tnum_current/N/dt << endl;

            count_recorded++;
            // export current state: Jij, s, x, y 
            // then polarization, delS12, analytical_S_integrand = new double[n_timesteps];

            if (print_state) {
                // Output state
                ofstream stateout;
                stateout.open((dir + "/seed" + intToString(seed) + "_Tnum" + intToString(Tnum) + "_state.txt").c_str());
                stateout.precision(17);
                stateout << tnum_current << endl;
                stateout << n_ran_calls << endl;
                stateout << count_recorded << endl;
                stateout << sum_polarization/tnum_current << endl;
                stateout << sum_delS12/tnum_current/N/dt << endl;
                stateout << analytical_S_firstterm << endl;
                stateout << analytical_S_integrand_prev << endl;
                stateout << Jij_times_sidotsj_sum_t0 << endl;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        stateout << Jij(i,j) << " " ;
                    }
                    stateout << "\n" << endl;
                }

                for (int i = 0; i < N; i++) {
                    stateout << x(i) << " " << y(i) << " " << z(i) << " " << s(i,0) << " " << s(i,1) << " " << s(i,2) << " " << endl;
                }
                stateout.close();
            }

            // Output data
            ofstream dataout;
            dataout.open((dir + "/seed" + intToString(seed) + "_Tnum" + intToString(Tnum) + "_data.txt").c_str());
            dataout.precision(17);
            dataout << tnum << endl;
            dataout << t << endl;
            dataout << sum_polarization/tnum_current << endl;
            dataout << sum_delS12/tnum_current/N/dt << endl;
            dataout << 0 << endl;
            dataout << (-analytical_S_firstterm)/(tnum_current*dt*N) << endl;
            dataout << 0 << endl;
            dataout.close();

            if (print_time_series) {
                ofstream timeseriesout;
                timeseriesout.open((dir + "/seed" + intToString(seed) + "_Tnum" + intToString(Tnum) + "_timeseries" + intToString(count_recorded) + ".txt").c_str());
                timeseriesout.precision(8);

                for (int i = tnum_start; i < tnum_current; i++) {
                    timeseriesout << i << " " << polarization[i-tnum_start] << " " << delS12[i-tnum_start] << " " << analytical_S_integrand[i-tnum_start] << endl;
                }

                timeseriesout.close();
            }

            if (print_birds) {
                ofstream birdsout;
                birdsout.open((dir + "/seed" + intToString(seed) + "_birds" + intToString(count_recorded) + ".txt").c_str());
                birdsout.precision(8);

                birdsout << N << " " << L << " " << t << " " << dt_record << " 0 0" << endl;
                for (int i = 0; i < N; i++) {
                    birdsout << x(i) << " " << y(i) << " " << z(i) << " " << s(i,0) << " " << s(i,1) << " " << s(i,2) << endl;
                }

                birdsout.close();
            }

            tnum_start = tnum_current;
        }

    }

    // double P = mean(polarization, n_timesteps);

    // double delS = mean(delS12, n_timesteps)/N/dt;

    if (!finished_already and print_time_series) cout << "current polarization = " << polarization[record_every_n-1] << endl;



    return;
}
