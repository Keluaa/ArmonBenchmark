
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>

#include <omp.h>


#ifndef USE_SIMD
#define USE_SIMD 1
#endif

#ifndef USE_THREADING
#define USE_THREADING 1
#endif

#if USE_SIMD
#define SIMD simd
#else
#define SIMD
#endif // USE_SIMD

#if USE_THREADING
#define _PRAGMA(V) _Pragma(#V)
#define OMP_FOR(...) _PRAGMA(omp parallel for SIMD default(none) __VA_ARGS__)
#else
#define OMP_FOR
#endif // USE_THREADING


// Program time contribution tracking
std::map<std::string, double> time_contribution;
#define CAT(a, b) a##b
#define TIC_IMPL(line_nb) auto CAT(tic_, line_nb) = std::chrono::steady_clock::now()
#define TAC_IMPL(label, line_nb) \
    auto CAT(tac_, line_nb) = std::chrono::steady_clock::now(); \
    double CAT(expr_time_, line_nb) = std::chrono::duration<double>(CAT(tac_, line_nb) - CAT(tic_, line_nb)).count(); \
    time_contribution[label]   += CAT(expr_time_, line_nb); \
    time_contribution["TOTAL"] += CAT(expr_time_, line_nb)
#define TIC() TIC_IMPL(__LINE__)
#define TAC(label) TAC_IMPL(label, __LINE__)


#ifndef USE_SINGLE_PRECISION
typedef float flt_t;
#else
typedef double flt_t;
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

int nb_cells = 1000, max_cycles = 100;
int verbose = 2;
flt_t max_time = 0.0, cfl = 0.6, Dt = 0.0;
bool cst_dt;
int nb_ghosts = 2;
int ideb, ifin;

bool do_write_output = false;
const char* output_file = "output_cpp";


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

OMP_FOR(firstprivate(ideb_, ifin_, rho0, G0, s, q, r, eps0, Cv0, T0, K0) shared(rho, pmat, emat, cmat, gmat))
    for (size_t i = ideb_; i < ifin_; i++) {
        flt_t x_ = rho[i] / rho0 - 1;
        flt_t g = G0 * (1 - rho0 / rho[i]);

        flt_t f0 = (1 + (s/3-2) * x_ + q * (x_ * x_) + r * (x_ * x_ * x_)) / (1 - s * x_);
        flt_t f1 = (s/3 - 2 + 2 * q * x_ + 3 * r * (x_ * x_) + s * f0) / (1 - s * x_);
        flt_t f2 = (2*q + 6 * r * x_ + 2 * s * f1) / (1 - s * x_);
        flt_t f3 = (6*r+3*s*f2)/(1- s * x_);

        flt_t eps_k0 = eps0 - Cv0 * T0 * (1 + g) + flt_t(0.5) * (K0 / rho0) * (x_ * x_) * f0;
        flt_t pk0 = -Cv0*T0*G0*rho0 + flt_t(0.5) * K0 * x_ * std::pow(1 + x_, flt_t(2)) * (2 * f0 + x_ * f1);
        flt_t pk0prime = -flt_t(0.5) * K0 * std::pow(1 + x_, flt_t(3)) * rho0 * (2 * (1 + 3 * x_) * f0 + 2 * x_ * (2 + 3 * x_) * f1 + (x_ * x_) * (1 + x_) * f2);
        flt_t pk0second = flt_t(0.5) * K0 * std::pow(1 + x_, flt_t(4)) * (rho0 * rho0) * (12 * (1 + 2 * x_) * f0 + 6 * (1 + 6 * x_ + 6 * (x_ * x_)) * f1 + 6 * x_ * (1 + x_) * (1 + 2 * x_) * f2 + std::pow(x_ * (1 + x_), flt_t(2)) * f3);

        pmat[i] = pk0 + G0*rho0*(emat[i] - eps_k0);
        cmat[i] = std::sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i];

        gmat[i] = flt_t(0.5)/(std::pow(rho[i],flt_t(3))*std::pow(cmat[i],flt_t(2)))*(pk0second+std::pow(G0*rho0,flt_t(2))*(pmat[i]-pk0));
    }
}


void perfectGasEOS(flt_t gamma)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR(firstprivate(ideb_, ifin_, gamma) shared(pmat, cmat, gmat, rho, emat))
    for (size_t i = ideb_; i < ifin_; i++) {
        pmat[i] = (gamma-1)*rho[i]*emat[i];
        cmat[i] = std::sqrt(gamma*pmat[i]/rho[i]);
        gmat[i] = (1+gamma)/2;
    }
}


flt_t phi(flt_t r)
{
    // Minmod
    return std::max(flt_t(0.), std::min(flt_t(1.), r));
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
            x[i] = flt_t(i - nb_ghosts) / flt_t(nb_cells);
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
            emat[i] = Emat[i] = pmat[i]/((gamma-1)*rho[i]);
            cmat[i] = std::sqrt(gamma*pmat[i]/rho[i]);
            gmat[i] = flt_t(0.5)*(1+gamma);
        }

        break;
    }
    case Test::Bizarrium:
    {
        if (max_time == 0.0) max_time = 80e-6;
        if (cfl == 0.0) cfl = 0.6;

        for (int i = ideb; i <= ifin; i++) {
            x[i] = flt_t(i - nb_ghosts) / flt_t(nb_cells);
            if (x[i] < 0.5) {
                rho[i] = 1.42857142857e+4;
                pmat[i] = 0.;
                emat[i] = Emat[i] = 4.48657821135e+6;
            }
            else {
                rho[i] =  10000.;
			    umat[i] = 250.;
			    emat[i] = 0.;
			    Emat[i] = emat[i] + flt_t(0.5) * std::pow(umat[i], flt_t(2));
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
OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(X, ustar, tmp_rho, tmp_urho, tmp_Erho, rho, umat, Emat))
    for (size_t i = ideb_; i <= ifin_; i++) {
        flt_t dx = X[i+1] - X[i];
        flt_t L1 =  std::max(flt_t(0.), ustar[i]) * dt;
        flt_t L3 = -std::min(flt_t(0.), ustar[i+1]) * dt;
        flt_t L2 = dx - L1 - L3;

        tmp_rho[i]  = (L1 * rho[i-1]             + L2 * rho[i]           + L3 * rho[i+1]            ) / dx;
        tmp_urho[i] = (L1 * rho[i-1] * umat[i-1] + L2 * rho[i] * umat[i] + L3 * rho[i+1] * umat[i+1]) / dx;
        tmp_Erho[i] = (L1 * rho[i-1] * Emat[i-1] + L2 * rho[i] * Emat[i] + L3 * rho[i+1] * Emat[i+1]) / dx;
    }

OMP_FOR(firstprivate(ideb_, ifin_) shared(tmp_rho, tmp_urho, tmp_Erho, rho, umat, Emat))
    for (size_t i = ideb_; i <= ifin_; i++) {
        rho[i]  = tmp_rho[i];
        umat[i] = tmp_urho[i] / tmp_rho[i];
        Emat[i] = tmp_Erho[i] / tmp_rho[i];
    }
}


flt_t dtCFL(flt_t dta)
{
    flt_t dt = INFINITY;

    if (cst_dt) {
        return Dt;
    }
    else if (euler_projection) {
OMP_FOR(firstprivate(ideb, ifin) shared(x, umat, cmat) reduction(min:dt))
        for (int i = ideb; i < ifin; i++) {
            dt = std::min(dt, ((x[i+1] - x[i]) / std::max(std::abs(umat[i] + cmat[i]), std::abs(umat[i] - cmat[i]))));
        }
    }
    else {
        for (int i = ideb; i < ifin; i++) {
            dt = std::min(dt, ((x[i+1] - x[i]) / cmat[i]));
        }
    }

    if (dta == 0) {
        if (Dt != 0) {
            return Dt;
        }
        else {
            return cfl * dt;
        }
    }
    else {
        return std::min(cfl * dt, flt_t(1.05) * dta);
    }
}


void acoustic()
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR(firstprivate(ideb_, ifin_) shared(rho, cmat, ustar, umat, pmat, pstar))
    for (size_t i = ideb_; i <= ifin_; i++) {
        flt_t rc_l = rho[i-1] * cmat[i-1];
        flt_t rc_r = rho[i]   * cmat[i];

        ustar[i] = (rc_l * umat[i-1] + rc_r * umat[i] + (pmat[i-1] - pmat[i])) / (rc_l + rc_r);
        pstar[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r);
    }
}


void acoustic_GAD(flt_t dt)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR(firstprivate(ideb_, ifin_) shared(rho, cmat, umat, pmat, ustar_1, pstar_1))
    for (size_t i = ideb_; i <= ifin_; i++) {
        flt_t rc_l = rho[i-1] * cmat[i-1];
        flt_t rc_r = rho[i]   * cmat[i];

        ustar_1[i] = (rc_l * umat[i-1] + rc_r * umat[i] + (pmat[i-1] - pmat[i])) / (rc_l + rc_r);
        pstar_1[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r);
    }

OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(x, rho, cmat, umat, pmat, ustar, pstar, ustar_1, pstar_1))
    for (size_t i = ideb_; i <= ifin_; i++) {
        flt_t r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + flt_t(1e-6));
        flt_t r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + flt_t(1e-6));
        flt_t r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + flt_t(1e-6));
        flt_t r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + flt_t(1e-6));
        // if (std::isnan(r_u_m)) r_u_m = 1;
        // if (std::isnan(r_p_m)) r_p_m = 1;
        // if (std::isnan(r_u_p)) r_u_p = 1;
        // if (std::isnan(r_p_p)) r_p_p = 1;

        flt_t dm_l = rho[i-1] * (x[i] - x[i-1]);
        flt_t dm_r = rho[i]   * (x[i+1] - x[i]);
        flt_t Dm = (dm_l + dm_r) / 2;
        flt_t theta = ((rho[i-1] * cmat[i-1]) + (rho[i] * cmat[i])) / 2 * (dt / Dm);

        ustar[i] = ustar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_u_p) * (umat[i] - ustar_1[i]) - phi(r_u_m) * (ustar_1[i] - umat[i-1]));
        pstar[i] = pstar_1[i] + flt_t(0.5) * (1 - theta)
                * (phi(r_p_p) * (pmat[i] - pstar_1[i]) - phi(r_p_m) * (pstar_1[i] - pmat[i-1]));
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

OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(X, x, ustar))
    for (size_t i = ideb_; i <= ifin_; i++) {
        X[i] = x[i] + dt * ustar[i];
    }

OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(X, x, rho, umat, Emat, emat, pstar, ustar))
    for (size_t i = ideb_; i < ifin_; i++) {
        flt_t dm = rho[i] * (x[i+1] - x[i]);
        rho[i] = dm / (X[i+1] - X[i]);
        umat[i] += dt / dm * (pstar[i] - pstar[i+1]);
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1]);
        emat[i] = Emat[i] - flt_t(0.5) * std::pow(umat[i], flt_t(2));
    }
    
    if (!euler_projection) {
OMP_FOR(firstprivate(ideb_, ifin_) shared(X, x))
        for (size_t i = ideb_; i <= ifin_; i++) {
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

    auto time_loop_start = std::chrono::steady_clock::now();

    while (t < max_time && cycle < max_cycles) {
        TIC(); boundaryConditions();  TAC("boundaryConditions");
        TIC(); flt_t dt = dtCFL(dta); TAC("dtCFL");
        TIC(); numericalFluxes(dt);   TAC("numericalFluxes");
        TIC(); cellUpdate(dt);        TAC("cellUpdate");

        if (euler_projection) {
            TIC(); first_order_euler_remap(dt); TAC("first_order_euler");
        }

        TIC(); update_EOS();          TAC("update_EOS");

        dta = dt;
        cycle++;
        t += dt;

        if (verbose <= 1) {
            printf("Cycle = %d, dt = %.3g, t = %.3g\n", cycle, dt, t);
        }
    }

    auto time_loop_end = std::chrono::steady_clock::now();

    double loop_time = std::chrono::duration<double>(time_loop_end - time_loop_start).count();
    double grind_time = loop_time / (double(cycle) * nb_cells) * 1e6;

    printf("\n");
    printf("Time:       %.4f seconds\n", loop_time);
    printf("Grind time: %.4f µs/cell/cycle\n", grind_time);
    printf("Cells/sec:  %.4f Mega cells/sec\n", 1. / grind_time);
    printf("Cycles:     %d\n\n", cycle);
}


void write_output()
{
    printf("Writing to output file '%s'...\n", output_file);

    FILE* file = fopen(output_file, "w");
    for (int i = ideb; i < ifin; i++) {
        fprintf(file, "%f, %f, %f, %f, %f, %f, %f\n",
                (x[i] + x[i+1]) * 0.5, rho[i], umat[i], pmat[i], emat[i], cmat[i], gmat[i]);
    }
    fclose(file);

    printf("Done.\n\n");
}


const char USAGE[] = R"(
 == Armon ==
CFD 1D solver using the conservative Euler equations in the lagrangian description.
Parallelized using OpenMP

Options:
    -h or --help            Prints this message and exit
    -t <test>               Test case: 'Sod' or 'Bizarrium'
    -s <scheme>             Numeric scheme: 'Godunov' (first order) or 'GAD-minmod' (second order, minmod limiter)
    --cells N               Number of cells in the mesh
    --cycle N               Maximum number of iterations
    --riemann <solver>      Riemann solver: 'acoustic' only
    --euler 0-1             Enable the eulerian projection step after each iteration
    --time T                Maximum time (in seconds)
    --cfl C                 CFL number
    --dt T                  Initial time step (in seconds)
    --cst-dt 0-1            Constant time step mode
    --write-output 0-1      If the variables should be written to the output file
    --output <file>         The output file name/path
    --verbose 0-3           Verbosity (0: high, 3: low)
)";


void parse_arguments(int argc, const char* argv[])
{
    errno = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            if (strcmp(argv[i+1], "Sod") == 0) {
                test = Test::Sod;
            }
            else if (strcmp(argv[i+1], "Bizarrium") == 0) {
                test = Test::Bizarrium;
            }
            else {
                printf("Unknown test: %s\n", argv[i+1]);
                exit(1);
            }
            i++;
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
            i++;
        }
        else if (strcmp(argv[i], "--cells") == 0) {
            nb_cells = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--cycle") == 0) {
            max_cycles = (int) strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = (int) strtol(argv[i+1], nullptr, 10);
            i++;
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
            max_time = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--cfl") == 0) {
            cfl = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--dt") == 0) {
            Dt = flt_t(strtod(argv[i+1], nullptr));
            i++;
        }
        else if (strcmp(argv[i], "--write-output") == 0) {
            do_write_output = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--euler") == 0) {
            euler_projection = strtol(argv[i+1], nullptr, 10);
            i++;
        }
        else if (strcmp(argv[i], "--cst-dt") == 0) {
            cst_dt = strtol(argv[i+1], nullptr, 2);
            i++;
        }
        else if (strcmp(argv[i], "--output") == 0) {
            output_file = argv[i+1];
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            puts(USAGE);
            exit(0);
        }
        else {
            fprintf(stderr, "Wrong option: %s\n", argv[i]);
            exit(1);
        }
    }

    if (errno != 0) {
        fprintf(stderr, "Parsing error occurred: %s\n", std::strerror(errno));
        exit(1);
    }
}


int main(int argc, const char* argv[])
{
    parse_arguments(argc, argv);

    int size = nb_cells + 2 * nb_ghosts;
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

    ideb = nb_ghosts;
    ifin = nb_ghosts + nb_cells;

    TIC(); init_test(); TAC("init_test");

    if (verbose < 3) {
        printf("Parameters:\n");
#if USE_THREADING
        printf(" - multithreading: 1 (%d threads)\n", omp_get_max_threads());
#else
        printf(" - multithreading: 0\n");
#endif
        printf(" - use simd:   %d\n", USE_SIMD);
        printf(" - ieee bits:  %lu\n", 8 * sizeof(flt_t));
        printf("\n");
        printf(" - test:       %s\n", (test == Test::Sod) ? "Sod" : "Bizarrium");
        printf(" - riemann:    %s\n", "acoustic");
        printf(" - scheme:     %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
        printf(" - nb cells:   %g\n", double(nb_cells));
        printf(" - Dt init:    %g\n", Dt);
        printf(" - euler proj: %d\n", euler_projection);
        printf(" - cst dt:     %d\n", cst_dt);
        printf(" - cfl:        %g\n", cfl);
        printf(" - max time:   %g\n", max_time);
        printf(" - max cycles: %d\n", max_cycles);
        if (do_write_output) {
            printf(" - output:     '%s'\n", output_file);
        }
        else {
            printf(" - no output\n");
        }
    }

    time_loop();

    if (do_write_output) {
        write_output();
    }

    if (verbose < 3) {
        double total_time = time_contribution["TOTAL"];
        time_contribution.erase(time_contribution.find("TOTAL"));

        printf("Total time for each step:\n");
        for (const auto& [label, time] : time_contribution) {
            printf(" - %-20s %10.5f ms (%5.2f%%)\n", label.c_str(), time * 1e3, time / total_time * 100);
        }
    }
}
