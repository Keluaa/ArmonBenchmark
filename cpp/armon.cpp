
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include <fenv.h>

#include <map>
#include <string>

#ifndef USE_SIMD
#define USE_SIMD 1
#endif

#ifndef USE_THREADING
#define USE_THREADING 1
#endif

#ifdef __clang__
#define INDEX_TYPE int
#else
#define INDEX_TYPE size_t
#endif

#if USE_SIMD

#if USE_THREADING
#define OMP_FOR _Pragma("omp parallel for simd")
#else
#define OMP_FOR _Pragma("omp for simd")
#endif // USE_THREADING

#else

#if USE_THREADING
#define OMP_FOR _Pragma("omp parallel for")
#else
#define OMP_FOR
#endif // USE_THREADING

#endif // USE_SIMD

#define DO_TIME_MEASUREMENT 1

#if DO_TIME_MEASUREMENT
std::map<std::string, double> time_contrib;
#define TIME_POS(label, expr) { \
    auto _t1 = std::chrono::high_resolution_clock::now(); \
    expr \
    auto _t2 = std::chrono::high_resolution_clock::now(); \
    double _expr_time = std::chrono::duration<double>(_t2 - _t1).count(); \
    time_contrib[label] += _expr_time; }
#else
#define TIME_POS(label, expr) { expr }
#endif


#ifndef USE_SINGLE_PRECISION
typedef double flt_t;
#else
typedef float flt_t;
#endif


enum class Test {
    Sod,
    Bizarrium
};

enum class Scheme {
    Godunov,
    GAD_minmod
};

enum class Riemann {
    Acoustic
};


Test test = Test::Sod;
Scheme scheme = Scheme::GAD_minmod;
Riemann riemann = Riemann::Acoustic;

bool euler_projection = false;

int nbcells = 1000, maxcycles = 100;
int verbose = 2;
bool do_write_output = false;
flt_t total_time = 0.0, max_time = 0.0, cfl = 0.6, Dt = 0.0;
int nbghost = 2;
int ideb, ifin;


std::vector<flt_t> x;
std::vector<flt_t> X;
std::vector<flt_t> rho;
std::vector<flt_t> umat;
std::vector<flt_t> emat;
std::vector<flt_t> Emat;
std::vector<flt_t> pmat;
std::vector<flt_t> cmat;
std::vector<flt_t> gmat;
std::vector<flt_t> ustar;
std::vector<flt_t> pstar;
std::vector<flt_t> ustar_1;
std::vector<flt_t> pstar_1;
std::vector<flt_t> tmp_rho;
std::vector<flt_t> tmp_urho;
std::vector<flt_t> tmp_Erho;


void BizarriumEOS()
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    const size_t ideb_ = ideb, ifin_ = ifin;

OMP_FOR
    for (INDEX_TYPE i = ideb_; i < ifin_; i++) {
        flt_t x = rho[i] / rho0 - 1;
        flt_t g = G0 * (1 - rho0 / rho[i]);

        flt_t f0 = (1+(s/3-2)*x+q*(x*x)+r*(x*x*x))/(1-s*x);
        flt_t f1 = (s/3-2+2*q*x+3*r*(x*x)+s*f0)/(1-s*x);
        flt_t f2 = (2*q+6*r*x+2*s*f1)/(1-s*x);
        flt_t f3 = (6*r+3*s*f2)/(1-s*x);

        flt_t epsk0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*(x*x)*f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*pow(1+x,2)*(2*f0+x*f1);
        flt_t pk0prime = -0.5*K0*pow(1+x,3)*rho0 * (2*(1+3*x)*f0 + 2*x*(2+3*x)*f1 + (x*x)*(1+x)*f2);
        flt_t pk0second = 0.5*K0*pow(1+x,4)*(rho0*rho0) * (12*(1+2*x)*f0 + 6*(1+6*x+6*(x*x))*f1 + 6*x*(1+x)*(1+2*x)*f2 + pow(x*(1+x),2)*f3);

        pmat[i] = pk0 + G0*rho0*(emat[i] - epsk0);
        cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i];

        gmat[i] = 0.5/(pow(rho[i],3)*pow(cmat[i],2))*(pk0second+pow(G0*rho0,2)*(pmat[i]-pk0));
    }
}


void perfectGasEOS(flt_t gamma)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR
    for (INDEX_TYPE i = ideb_; i < ifin_; i++) {
        pmat[i] = (gamma-1)*rho[i]*emat[i];
        cmat[i] = std::sqrt(gamma*pmat[i]/rho[i]);
        gmat[i] = (1+gamma)/2;
    }
}


flt_t phi(flt_t r)
{
    // Minmod
    return std::max(0., std::min(1., r));
}


void init_test()
{
    switch (test)
    {
    case Test::Sod:
    {
        if (max_time == 0.0) max_time = 0.20;
        if (cfl == 0.0) cfl = 0.95;

        for (int i = ideb; i <= ifin; i++) {
            x[i] = static_cast<flt_t>(i - nbghost) / nbcells;
            if (x[i] < 0.5) {
                rho[i] = 1.;
                pmat[i] = 1.;
                umat[i] = 0.;
            }
            else {
                rho[i] = 0.125;
                pmat[i] = 0.1;
                umat[i] = 0.0;
            }

            const flt_t gamma = 1.4;
            emat[i] = Emat[i] = pmat[i]/((gamma-1.)*rho[i]);
            cmat[i] = std::sqrt(gamma*pmat[i]/rho[i]);
            gmat[i] = 0.5*(1.0+gamma);
        }

        break;
    }
    case Test::Bizarrium:
    {
        if (max_time == 0.0) max_time = 80e-6;
        if (cfl == 0.0) cfl = 0.6;

        for (int i = ideb; i <= ifin; i++) {
            x[i] = static_cast<flt_t>(i - nbghost) / nbcells;
            if (x[i] < 0.5) {
                rho[i] = 1.42857142857e+4;
                pmat[i] = 0.;
                emat[i] = Emat[i] = 4.48657821135e+6;
            }
            else {
                rho[i] =  10000.;
			    umat[i] = 250.;
			    emat[i] = 0.;
			    Emat[i] = emat[i] + 0.5 * pow(umat[i], 2);
            }

            BizarriumEOS();
        }

        break;
    }
    }
}


void boundaryConditions()
{
    rho[ideb-1] = rho[ideb];    rho[ifin] = rho[ifin-1]; 
    umat[ideb-1] = -umat[ideb]; 
    pmat[ideb-1] = pmat[ideb];  pmat[ifin] = pmat[ifin-1]; 
    cmat[ideb-1] = cmat[ideb];  cmat[ifin] = cmat[ifin-1]; 
    gmat[ideb-1] = gmat[ideb];  gmat[ifin] = gmat[ifin-1];

    if (test == Test::Bizarrium) {
        umat[ifin] = umat[ifin-1];
    }
    else {
        umat[ifin] = -umat[ifin-1];
    }
}


void first_order_euler_remap(flt_t dt)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        flt_t dx = X[i+1] - X[i];
        flt_t L1 =  std::max(0., ustar[i]) * dt;
        flt_t L3 = -std::min(0., ustar[i+1]) * dt;
        flt_t L2 = dx - L1 - L3;

        tmp_rho[i]  = (L1 * rho[i-1]             + L2 * rho[i]           + L3 * rho[i+1]            ) / dx;
        tmp_urho[i] = (L1 * rho[i-1] * umat[i-1] + L2 * rho[i] * umat[i] + L3 * rho[i+1] * umat[i+1]) / dx;
        tmp_Erho[i] = (L1 * rho[i-1] * Emat[i-1] + L2 * rho[i] * Emat[i] + L3 * rho[i+1] * Emat[i+1]) / dx;
    };

OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        rho[i]  = tmp_rho[i];
        umat[i] = tmp_urho[i] / tmp_rho[i];
        Emat[i] = tmp_Erho[i] / tmp_rho[i];
    }
}


flt_t dtCFL(flt_t dta)
{
    flt_t dt = 1e99;

    if (euler_projection) {
        for (int i = ideb; i < ifin; i++) {
            dt = std::min(dt, ((x[i+1]-x[i]) / std::max(std::abs(umat[i] + cmat[i]), std::abs(umat[i] - cmat[i]))));
        }
    }
    else {
        for (int i = ideb; i < ifin; i++) {
            dt = std::min(dt, ((x[i+1]-x[i])/cmat[i]));
        }
    }

    if (dta == 0) {
        if (Dt != 0) {
            return Dt;
        }
        else {
            return cfl*dt;
        }
    }
    else {
        return std::min(cfl*dt, 1.05*dta);
    }
}


void acoustic()
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        flt_t rc_l = rho[i-1] * cmat[i-1];
        flt_t rc_r = rho[i]   * cmat[i];

        ustar[i] = (rc_l * umat[i-1] + rc_r * umat[i] + (pmat[i-1] - pmat[i])) / (rc_l + rc_r);
        pstar[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r);
    }
}


void acoustic_GAD(flt_t dt)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        flt_t rc_l = rho[i-1] * cmat[i-1];
        flt_t rc_r = rho[i]   * cmat[i];

        ustar_1[i] = (rc_l * umat[i-1] + rc_r * umat[i] + (pmat[i-1] - pmat[i])) / (rc_l + rc_r);
        pstar_1[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r);
    }

OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        flt_t r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6);
        flt_t r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6);
        flt_t r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6);
        flt_t r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6);
        // if (std::isnan(r_u_m)) r_u_m = 1;
        // if (std::isnan(r_p_m)) r_p_m = 1;
        // if (std::isnan(r_u_p)) r_u_p = 1;
        // if (std::isnan(r_p_p)) r_p_p = 1;

        flt_t dm_l = rho[i-1] * (x[i] - x[i-1]);
        flt_t dm_r = rho[i]   * (x[i+1] - x[i]);
        flt_t Dm = (dm_l + dm_r) / 2;
        flt_t theta = ((rho[i-1] * cmat[i-1]) + (rho[i] * cmat[i])) / 2 * (dt / Dm);

        ustar[i] = ustar_1[i] + 0.5 * (1 - theta) * (phi(r_u_p) * (umat[i] - ustar_1[i]) - phi(r_u_m) * (ustar_1[i] - umat[i-1]));
        pstar[i] = pstar_1[i] + 0.5 * (1 - theta) * (phi(r_p_p) * (pmat[i] - pstar_1[i]) - phi(r_p_m) * (pstar_1[i] - pmat[i-1]));
    }
}


void numericalFluxes(flt_t dt)
{
    switch (riemann) {
    case Riemann::Acoustic:
    {   
        switch (scheme) {
        case Scheme::Godunov:    acoustic();       break;
        case Scheme::GAD_minmod: acoustic_GAD(dt); break;
        }
        break;
    }
    }
}


void cellUpdate(flt_t dt)
{
    const size_t ideb_ = ideb, ifin_ = ifin;

OMP_FOR
    for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
        X[i] = x[i] + dt * ustar[i];
    }

OMP_FOR
    for (INDEX_TYPE i = ideb_; i < ifin_; i++) {
        flt_t dm = rho[i] * (x[i+1] - x[i]);
        rho[i] = dm / (X[i+1] - X[i]);
        umat[i] += dt / dm * (pstar[i] - pstar[i+1]);
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1]);
        emat[i] = Emat[i] - 0.5 * pow(umat[i], 2);
    }
    
    if (!euler_projection) {
OMP_FOR
        for (INDEX_TYPE i = ideb_; i <= ifin_; i++) {
            x[i] = X[i];
        }
    }
}


void update_EOS()
{
    switch (test)
    {
    case Test::Sod:
    {
        const flt_t gamma = 1.4;
        perfectGasEOS(gamma);
        break;
    }
    case Test::Bizarrium:
    {
        BizarriumEOS();
        break;
    }
    }
}


void time_loop()
{   
    int cycle = 0;
    flt_t t = 0.0, dta = 0.0;

    auto t1 = std::chrono::steady_clock::now();

    while (t < max_time && cycle < maxcycles) {
        TIME_POS("boundaryConditions", boundaryConditions();)
        flt_t dt;
        TIME_POS("dtCFL", dt = dtCFL(dta);)
        TIME_POS("numericalFluxes", numericalFluxes(dt);)
        TIME_POS("cellUpdate", cellUpdate(dt);)

        if (euler_projection) {
            TIME_POS("first_order_euler_remap", first_order_euler_remap(dt);)
        }

        TIME_POS("update_EOS", update_EOS();)

        dta = dt;
        cycle++;
        t += dt;
    }

    auto t2 = std::chrono::steady_clock::now();

    flt_t loop_time = std::chrono::duration<flt_t>(t2 - t1).count();
    flt_t grind_time = loop_time / (flt_t(cycle) * nbcells) * 1e6;
    printf("\n");
    printf("Time: %.4f seconds\n", loop_time);
    printf("Grind time: %.4f microseconds/cell/cycle\n", grind_time);
    printf("Cells/sec:  %.4f Mega cells/sec\n", 1. / grind_time);
    printf("Cycle: %d\n", cycle);
    printf("\n");
}


void write_output()
{
    std::ofstream file("output_cpp", std::ios_base::out);
    
    printf("Writing output file...\n");
    for (int i = ideb; i < ifin; i++) {
        file << 0.5 * (x[i] + x[i+1]) << ", "
             << rho[i] << ", "
             << umat[i] << ", "
             << pmat[i] << ", "
             << emat[i] << ", "
             << cmat[i] << ", "
             << gmat[i] << "\n";
    }
    printf("Done.\n");
}


void parse_arguments(int argc, const char* argv[])
{
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-t") == 0) {
            if (strcmp(argv[i+1], "Sod") == 0) {
                test = Test::Sod;
            }
            else if (strcmp(argv[i+1], "Bizarrium") == 0) {
                test = Test::Bizarrium;
            }
            else {
                printf("Wrong test: %s\n", argv[i+1]);
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-s") == 0) {
            if (strcmp(argv[i+1], "Godunov") == 0) {
                scheme = Scheme::Godunov;
            }
            else if (strcmp(argv[i+1], "GAD-minmod") == 0) {
                scheme = Scheme::GAD_minmod;
            }
            else {
                printf("Wrong scheme: %s\n", argv[i+1]);
                exit(1);
            }
        }
        else if (strcmp(argv[i], "--cells") == 0) {
            nbcells = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--cycle") == 0) {
            maxcycles = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--riemann") == 0) {
            if (strcmp(argv[i+1], "acoustic") == 0) {
                riemann = Riemann::Acoustic;
            }
            else {
                printf("Wrong riemann solver: %s\n", argv[i+1]);
                exit(1);
            }
        }
        else if (strcmp(argv[i], "--time") == 0) {
            max_time = atof(argv[i+1]);
        }
        else if (strcmp(argv[i], "--cfl") == 0) {
            cfl = atof(argv[i+1]);
        }
        else if (strcmp(argv[i], "--dt") == 0) {
            Dt = atof(argv[i+1]);
        }
        else if (strcmp(argv[i], "--write-output") == 0) {
            do_write_output = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--euler") == 0) {
            euler_projection = atoi(argv[i+1]);
        }
        else {
            printf("Wrong argument: %s\n", argv[i]);
            exit(1);
        }
    }
}


int main(int argc, const char* argv[])
{
    feenableexcept(FE_INVALID);

    printf("Num threads: %s (max: %d)\n", getenv("OMP_NUM_THREADS"), omp_get_max_threads());

    parse_arguments(argc, argv);

    int size = nbcells + 2 * nbghost;
    x = std::vector<flt_t>(size);
    X = std::vector<flt_t>(size);
    rho = std::vector<flt_t>(size);
    umat = std::vector<flt_t>(size);
    emat = std::vector<flt_t>(size);
    Emat = std::vector<flt_t>(size);
    pmat = std::vector<flt_t>(size);
    cmat = std::vector<flt_t>(size);
    gmat = std::vector<flt_t>(size);
    ustar = std::vector<flt_t>(size);
    pstar = std::vector<flt_t>(size);
    ustar_1 = std::vector<flt_t>(size);
    pstar_1 = std::vector<flt_t>(size);
    tmp_rho = std::vector<flt_t>(size);
    tmp_urho = std::vector<flt_t>(size);
    tmp_Erho = std::vector<flt_t>(size);

    ideb = nbghost;
    ifin = nbghost + nbcells;

    TIME_POS("init_test", init_test();)

    printf("Test problem: %s\n", (test == Test::Sod) ? "Sod" : "Bizarrium");
    printf("Hydro scheme: %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
    printf("Riemann: acoustic\n");
    printf("Cells: %d\n", nbcells);
    printf("Maxtime: %f\n", max_time);
    printf("cfl: %f\n", cfl);
    printf("dt: %f\n", Dt);

    time_loop();

    if (do_write_output) {
        write_output();
    }

#if DO_TIME_MEASUREMENT
    double total_time = 0.;
    for (const auto& [_, time] : time_contrib) {
        total_time += time;
    }

    printf("Total time for each step:\n");
    for (const auto& [label, time] : time_contrib) {
        printf(" - %-25s %10.5f ms (%5.2f%%)\n", label.c_str(), time * 1e3, time / total_time * 100);
    }
#endif
}
