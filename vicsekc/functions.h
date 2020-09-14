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

double dot(double ux, double uy, double vx, double vy) {
    // dot product of vectors u and v
    return ux*vx + uy*vy;
}

double angle2D(double ux, double uy, double vx, double vy) {
    // angle between vectors u and v (in radians)
    double temp = dot(ux, uy, vx, vy)/sqrt(dot(ux,uy,ux,uy)*dot(vx,vy,vx,vy));
    if (temp > 1.0) temp = 1.0;
    if (temp < -1.0) temp = -1.0;
    return acos(temp);
}


double prob2d(double theta, double x0, double sigma) {
    double a = x0*cos(theta)/sqrt(2.0*sigma*sigma);
    double p = 2.0 * 1.0/(2.0*M_PI) * (exp(-x0*x0/(2.0*sigma*sigma)) + sqrt(M_PI*(2.0*sigma*sigma))*x0*cos(theta)*exp(x0*x0*(cos(theta)*cos(theta)-1.0)/(2.0*sigma*sigma))*erfc(-a));
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

void bin_birds(VectorXd x, VectorXd y, int N, double L) {

    // side length is N
    // cutoff length is r_cutoff
    // minimum integer subdivision greater than or equal to r_cutoff is ceil(L/r_cutoff)
    
    int this_xbin, this_ybin;

    for (int i = 0; i < nbins*nbins; i++) {
        bins[i].clear();
    }

    for (int i = 0; i < N; i++) {

        this_xbin = xbin(x(i), L);
        this_ybin = ybin(y(i), L);
        bins[this_xbin + nbins*this_ybin].push_back(i);

    }
}

void simulation_2D(double dt, double T, double rc, int N, double L, double v, double tmax, double J, double seed, double t_equil, int Tnum, mt19937& generator, string dir, bool print_time_series)
{
    double d = 2.0; // no. of dimensions
    // double dt_record = 100;
    double dt_sim = dt;
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
    VectorXd sum_dt_Jij_sx(N);
    VectorXd sum_dt_Jij_sy(N);
    VectorXd delS_analytic_contribution(N);
    MatrixXd Jij(N,N);    
    MatrixXd Jij_0(N,N);    
    MatrixXd Jij_prev(N,N);
    MatrixXd dJij_dt(N,N);    
    double rx, ry, r2, rij, rxRound, ryRound;
    MatrixXd s(N,2);
    MatrixXd s_prev(N,2);
    MatrixXd s_det(N,2);
    MatrixXd s_r(N,2);
    MatrixXd s_prev_r(N,2);
    MatrixXd s_det_r(N,2);
    MatrixXd eta(N,2);
    double mag, theta, sumsx, sumsy, ux, uy, vx, vy, t, delS12_current, polarization_current;
    double sx, sy, analytical_S_firstterm = 0, analytical_S_integrand_prev = 0, analytical_S_integrand_this, Jij_times_sidotsj_sum_t0, Jij_times_sidotsj_sum;
    std::vector<double> pforward(N);
    std::vector<double> pbackward(N);
    double *theta_noise = new double[N];
    double *l = new double[N];
    int this_xbin, this_ybin, that_xbin, that_ybin, j;
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

        sum_polarization = mean_polarization*tnum_current;
        sum_delS12 = mean_delS12_over_Ndt*(tnum_current*N*dt);
        
        for (int i = 0; i < N; i++) {
            statein >> x(i);
            statein >> y(i);
            statein >> s(i,0);
            statein >> s(i,1);
        }

        statein.close();

        tnum_start = tnum_current;

        // call the random number generator n_ran_calls times
        for (int i = 0; i < n_ran_calls; i++) {
            dist(generator);
        }
    }

    else {

        cout << "No state file exists! Starting from time 0" << endl;

        for (int i = 0; i < N; i++) {
            x(i) = L*randDoub(generator);
            y(i) = L*randDoub(generator);
        }

        for (int i = 0; i < N; i++) {
            theta = randDoub(generator)*2.0*M_PI;
            s(i,0) = cos(theta);
            s(i,1) = sin(theta);
        }

        // equilibration
        cout << "equilibrating for " << ceil(t_equil/dt) << " timesteps" << endl;

        dt = 0.05; // use a larger timestep for equilibration

        for (int tnum = 0; tnum < ceil(t_equil/dt); tnum++) {

            if (tnum%100 == 0) {
                cout << "equilibrating... " << tnum*1.0/ceil(t_equil/dt)*100.0 << "%" << endl;
            }

            bin_birds(x, y, N, L);

            Jij = MatrixXd::Zero(N,N);
            sum_dt_Jij_sx = VectorXd::Zero(N);
            sum_dt_Jij_sy = VectorXd::Zero(N);


            double sum = 0;

            #pragma omp parallel shared(x, y, Jij, sum_dt_Jij_sx, sum_dt_Jij_sy) private(j, this_xbin, this_ybin, that_xbin, that_ybin, rx, ry, rxRound, ryRound, r2, rij)
            {

                #pragma omp for
                for (int i = 0; i < N; i++) {

                    // consider bird i; what are its bins?
                    this_xbin = xbin(x(i), L);
                    this_ybin = ybin(y(i), L);

                    // what are the 9 neighboring bins?
                    std::vector<std::pair<int,int> > neighborBins_this; 
                    neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin-1)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin+1)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin-1)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin+1)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin-1)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin)%nbins));
                    neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin+1)%nbins));

                    // iterate through neighbor bins and compute distance between bird i and neighbors
                    for (int bin = 0; bin < 9; bin ++) {
                        that_xbin = neighborBins_this[bin].first;
                        that_ybin = neighborBins_this[bin].second;

                        for (std::list<int>::iterator it = bins[that_xbin + nbins*that_ybin].begin(); it != bins[that_xbin + nbins*that_ybin].end(); ++it){
                            j = *it;

                            if (i != j) {

                                rx = x(j) - x(i);
                                ry = y(j) - y(i);

                                // take into account periodic boundary conditions
                                rxRound = L*(round(rx/L));
                                ryRound = L*(round(ry/L));
                                rx = rx - rxRound;
                                ry = ry - ryRound;
                                r2 = rx*rx + ry*ry;
                                rij = sqrt(r2);

                                if (rij < r_cutoff) {
                                    Jij(i,j) = J*exp(-rij/rc);

                                    sum_dt_Jij_sx(i) += dt*Jij(i,j)*s(j,0);
                                    sum_dt_Jij_sy(i) += dt*Jij(i,j)*s(j,1);
                                }
                            }
                        }
                    }

                }
            }

            #pragma omp parallel shared(eta, s, x, y, sum_dt_Jij_sx, sum_dt_Jij_sy) private(mag)
            {

                #pragma omp for
                for (int i = 0; i < N; i++) {

                    eta(i, 0) = normRand(0.0, 1.0, generator);
                    eta(i, 1) = normRand(0.0, 1.0, generator);
                    s(i,0) = s(i,0) + sum_dt_Jij_sx(i) + sqrt(2.0*T*dt)*eta(i, 0);
                    s(i,1) = s(i,1) + sum_dt_Jij_sy(i) + sqrt(2.0*T*dt)*eta(i, 1);
                    mag = sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1));
                    s(i, 0) = s(i,0)/mag;
                    s(i, 1) = s(i,1)/mag;
                    x(i) = x(i) + v*s(i,0)*dt;
                    y(i) = y(i) + v*s(i,1)*dt;
                    if (x(i) >= L) x(i) = x(i) - L;
                    if (x(i) < 0) x(i) = x(i) + L;
                    if (y(i) >= L) y(i) = y(i) - L;
                    if (y(i) < 0) y(i) = y(i) + L;
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

        bin_birds(x, y, N, L);

        Jij = MatrixXd::Zero(N,N);
        dJij_dt = MatrixXd::Zero(N,N);
        sum_dt_Jij_sx = VectorXd::Zero(N);
        sum_dt_Jij_sy = VectorXd::Zero(N);
        neighbors.clear();
        neighbors.resize(N);

        #pragma omp parallel shared(x, y, Jij, sum_dt_Jij_sx, sum_dt_Jij_sy, neighbors) private(j, this_xbin, this_ybin, that_xbin, that_ybin, rx, ry, rxRound, ryRound, r2, rij)
        {

            #pragma omp for
            for (int i = 0; i < N; i++) {

                // consider bird i; what are its bins?
                this_xbin = xbin(x(i), L);
                this_ybin = ybin(y(i), L);
                // what are the 9 neighboring bins?
                std::vector<std::pair<int,int> > neighborBins_this; 
                neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin-1)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin-1)%nbins, (nbins+this_ybin+1)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin-1)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin)%nbins, (nbins+this_ybin+1)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin-1)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin)%nbins));
                neighborBins_this.push_back(make_pair((nbins+this_xbin+1)%nbins, (nbins+this_ybin+1)%nbins));

                // iterate through neighbor bins and compute distance between bird i and neighbors
                for (int bin = 0; bin < 9; bin ++) {
                    that_xbin = neighborBins_this[bin].first;
                    that_ybin = neighborBins_this[bin].second;

                    for (std::list<int>::iterator it = bins[that_xbin + nbins*that_ybin].begin(); it != bins[that_xbin + nbins*that_ybin].end(); ++it){
                        j = *it;

                        if (i != j) {

                            rx = x(j) - x(i);
                            ry = y(j) - y(i);

                            // take into account periodic boundary conditions
                            rxRound = L*(round(rx/L));
                            ryRound = L*(round(ry/L));
                            rx = rx - rxRound;
                            ry = ry - ryRound;
                            r2 = rx*rx + ry*ry;
                            rij = sqrt(r2);

                            if (rij < r_cutoff) {
                                neighbors[i].push_back(j);

                                Jij(i,j) = J*exp(-rij/rc);

                                sum_dt_Jij_sx(i) += dt*Jij(i,j)*s(j,0);
                                sum_dt_Jij_sy(i) += dt*Jij(i,j)*s(j,1);
                            }
                        }
                    }
                }
            }
        }

        sumsx = 0;
        sumsy = 0;

        #pragma omp parallel reduction(+:sumsx, sumsy) shared(eta, s, x, y, sum_dt_Jij_sx, sum_dt_Jij_sy) private(mag)
        {
            #pragma omp for
            for (int i = 0; i < N; i++) {

                eta(i, 0) = normRand(0.0, 1.0, generator);
                eta(i, 1) = normRand(0.0, 1.0, generator);
                s(i,0) = s(i,0) + sum_dt_Jij_sx(i) + sqrt(2.0*T*dt)*eta(i, 0);
                s(i,1) = s(i,1) + sum_dt_Jij_sy(i) + sqrt(2.0*T*dt)*eta(i, 1);
                mag = sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1));
                s(i, 0) = s(i,0)/mag;
                s(i, 1) = s(i,1)/mag;
                x(i) = x(i) + v*s(i,0)*dt;
                y(i) = y(i) + v*s(i,1)*dt;
                if (x(i) >= L) x(i) = x(i) - L;
                if (x(i) < 0) x(i) = x(i) + L;
                if (y(i) >= L) y(i) = y(i) - L;
                if (y(i) < 0) y(i) = y(i) + L;
                sumsx += s(i,0);
                sumsy += s(i,1);
            }
        }

        polarization_current = sqrt(sumsx*sumsx + sumsy*sumsy)/N;


        s_det = MatrixXd::Zero(N,2);
        s_prev_r = -1.0*s_prev;
        s_r = -1.0*s;
        s_det_r = MatrixXd::Zero(N,2);

        delS12_current = 0.0;

        analytical_S_integrand_this = 0;

        #pragma omp parallel reduction(+:analytical_S_integrand_this) shared(theta_noise, l, pforward, pbackward, s_prev, s_det, s, s_r, s_det_r, s_prev_r) private(ux, uy, vx, vy, j)
        {
            #pragma omp for
            for (int i = 0; i < N; i++) {

                // calculate s_det and s_det_r (slightly faster to do it this way than directly multiplying Jij and s)

                for (int a = 0; a < neighbors[i].size(); a++) {
                    j = neighbors[i][a];
                    s_det(i,0) += dt*Jij(i,j)*s_prev(j,0);
                    s_det(i,1) += dt*Jij(i,j)*s_prev(j,1);
                    s_det_r(i,0) += dt*Jij(i,j)*s_prev_r(j,0);
                    s_det_r(i,1) += dt*Jij(i,j)*s_prev_r(j,1);

                    // if these were neighbors before and are neighbors now
                    if ((Jij(i,j) > tiny) and (Jij_prev(i,j) > tiny)) {
                        analytical_S_integrand_this += (s(i,0)*s(j,0) + s(i,1)*s(j,1))*(Jij(i,j) - Jij_prev(i,j))/dt/(2.0*T*(d-1.0));
                    }
                }

                ux = s_prev(i,0) + s_det(i,0);
                uy = s_prev(i,1) + s_det(i,1);
                vx = s(i,0);
                vy = s(i,1);
                theta_noise[i] = angle2D(ux, uy, vx, vy);
                l[i] = sqrt(dot(ux,uy,ux,uy));
                pforward[i] = prob2d(theta_noise[i], l[i], sqrt(2*T*dt));

                ux = s_r(i,0) + s_det_r(i,0);
                uy = s_r(i,1) + s_det_r(i,1);
                vx = s_prev_r(i,0);
                vy = s_prev_r(i,1);
                theta_noise[i] = angle2D(ux, uy, vx, vy);
                l[i] = sqrt(dot(ux,uy,ux,uy));
                pbackward[i] = prob2d(theta_noise[i], l[i], sqrt(2*T*dt));

            }
        }

        analytical_S_firstterm += dt*(analytical_S_integrand_this + analytical_S_integrand_prev)/2.0; // trapezoidal rule
        analytical_S_integrand_prev = analytical_S_integrand_this;
        if (print_time_series) {
            analytical_S_integrand[tnum-tnum_current] = analytical_S_integrand_this;
        }

        delS12_current = compute_delS12_current(pforward, pbackward, N);

        // % Compute probability of step t + 1 -> t (with reversed velocities)

        sum_polarization += polarization_current;
        sum_delS12 += delS12_current;

        if (print_time_series) {
            delS12[tnum-tnum_current] = delS12_current;
            polarization[tnum-tnum_current] = polarization_current;
        }


        if ((tnum > 0) and (tnum % record_every_n == 0)) {
            // cout << "t = " << t << endl;

            tnum_current = tnum + 1;

            cout << "Mean(P) = " << sum_polarization/tnum_current << ". S_prob = " << sum_delS12/tnum_current/N/dt << ". S_an = " << -(analytical_S_firstterm)/tnum_current/N/dt << endl;

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
                    stateout << x(i) << " " << y(i) << " " << s(i,0) << " " << s(i,1) << " " << endl;
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

                birdsout << N << " " << L << " " << t << " " << dt_record << " 0" << endl;
                for (int i = 0; i < N; i++) {
                    birdsout << x(i) << " " << y(i) << " " << s(i,0) << " " << s(i,1) << " " << delS_analytic_contribution(i)/(N*dt) << endl;
                    // birdsout << x(i) << " " << y(i) << " " << s(i,0) << " " << s(i,1) << endl;

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
