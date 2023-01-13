
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

#ifndef USE_STD_VECTOR
#define USE_STD_VECTOR 0
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
#define OMP_FOR(...)
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


#if USE_STD_VECTOR
template<typename T>
using Vector = std::vector<T>;
#else
template<typename T>
class Vector
{
    T* _ptr = nullptr;
public:
    Vector() = default;

    explicit Vector(size_t size)
    { _ptr = (T*) malloc(size * sizeof(T)); }

    ~Vector()
    { free(_ptr); }

    Vector(Vector<T>&& other) noexcept
    { std::swap(_ptr, other._ptr); }

    Vector& operator=(Vector<T>&& other) noexcept
    { std::swap(_ptr, other._ptr); return *this; }

    inline T operator[](int i) const
    { return _ptr[i]; }

    inline T operator[](size_t i) const
    { return _ptr[i]; }

    inline T& operator[](int i)
    { return _ptr[i]; }

    inline T& operator[](size_t i)
    { return _ptr[i]; }
};
#endif // USE_STD_VECTOR


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


Vector<flt_t> x;
Vector<flt_t> X;
Vector<flt_t> rho;
Vector<flt_t> umat;
Vector<flt_t> emat;
Vector<flt_t> Emat;
Vector<flt_t> pmat;
Vector<flt_t> cmat;
Vector<flt_t> gmat;
Vector<flt_t> ustar;
Vector<flt_t> pstar;
Vector<flt_t> ustar_1;
Vector<flt_t> pstar_1;
Vector<flt_t> tmp_rho;
Vector<flt_t> tmp_urho;
Vector<flt_t> tmp_Erho;


void BizarriumEOS()
{
    const flt_t rho0 = 1e4, K0 = 1e11, Cv0 = 1e3, T0 = 300, eps0 = 0;
    const flt_t G0 = 1.5, s = 1.5, q = -42080895./14941154., r = 727668333./149411540.;

    const size_t ideb_ = ideb, ifin_ = ifin;

OMP_FOR(firstprivate(ideb_, ifin_, rho0, G0, s, q, r, eps0, Cv0, T0, K0) shared(rho, pmat, Emat, cmat, gmat, umat))
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

        flt_t e = Emat[i] - flt_t(0.5) * std::pow(umat[i], flt_t(2));
        pmat[i] = pk0 + G0*rho0*(e - eps_k0);
        cmat[i] = std::sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i];
        gmat[i] = flt_t(0.5) / (std::pow(rho[i],flt_t(3)) * std::pow(cmat[i],flt_t(2)))
                * (pk0second+std::pow(G0*rho0,flt_t(2))*(pmat[i]-pk0));
    }
}


void perfectGasEOS(flt_t gamma)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR(firstprivate(ideb_, ifin_, gamma) shared(pmat, cmat, gmat, rho, Emat, umat))
    for (size_t i = ideb_; i < ifin_; i++) {
        flt_t e = Emat[i] - flt_t(0.5) * std::pow(umat[i], flt_t(2));
        pmat[i] = (gamma - 1) * rho[i] * e;
        cmat[i] = std::sqrt(gamma * pmat[i] / rho[i]);
        gmat[i] = (1 + gamma) / 2;
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

OMP_FOR(firstprivate(ideb, ifin, nb_ghosts, nb_cells) shared(x, rho, pmat, umat, Emat, cmat, gmat))
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
            Emat[i] = pmat[i]/((gamma-1)*rho[i]);
            cmat[i] = std::sqrt(gamma*pmat[i]/rho[i]);
            gmat[i] = flt_t(0.5)*(1+gamma);
        }
        break;
    }
    case Test::Bizarrium:
    {
        if (max_time == 0.0) max_time = 80e-6;
        if (cfl == 0.0) cfl = 0.6;

OMP_FOR(firstprivate(ideb, ifin, nb_ghosts, nb_cells) shared(x, rho, pmat, umat, Emat))
        for (int i = ideb; i <= ifin; i++) {
            x[i] = flt_t(i - nb_ghosts) / flt_t(nb_cells);
            if (x[i] < 0.5) {
                rho[i] = 1.42857142857e+4;
                umat[i] = 0.;
                pmat[i] = 0.;
                Emat[i] = 4.48657821135e+6;
            }
            else {
                rho[i] =  10000.;
                umat[i] = 250.;
                pmat[i] = 0.;
                Emat[i] = flt_t(0.5) * std::pow(umat[i], flt_t(2));
            }
        }
        BizarriumEOS();
        break;
    }
    }
}


void boundaryConditions()
{
    int i_l = ideb - 1;
    int ip_l = ideb;
    int i_r = ifin;
    int ip_r = ifin - 1;

    flt_t u_factor = (test == Test::Bizarrium) ? 1 : -1;

    for (int w = 0; w < nb_ghosts; w++) {
        rho[i_l] = rho[ip_l];    rho[i_r] = rho[ip_r];
        umat[i_l] = -umat[ip_l]; umat[i_r] = umat[ip_r] * u_factor;
        pmat[i_l] = pmat[ip_l];  pmat[i_r] = pmat[ip_r];
        cmat[i_l] = cmat[ip_l];  cmat[i_r] = cmat[ip_r];
        gmat[i_l] = gmat[ip_l];  gmat[i_r] = gmat[ip_r];

        i_l -= 1;
        ip_l += 1;
        i_r += 1;
        ip_r -= 1;
    }
}


void first_order_euler_remap(flt_t dt)
{
    const size_t ideb_ = ideb, ifin_ = ifin;
OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(X, ustar, tmp_rho, tmp_urho, tmp_Erho, rho, umat, Emat))
    for (size_t i = ideb_; i <= ifin_; i++) {
        size_t idx = i;
        flt_t disp = dt * ustar[i];
        if (disp > 0) {
            idx = i - 1;
        }

        tmp_rho[i]  = disp * (rho[idx]            );
        tmp_urho[i] = disp * (rho[idx] * umat[idx]);
        tmp_Erho[i] = disp * (rho[idx] * Emat[idx]);
    }

OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(tmp_rho, tmp_urho, tmp_Erho, rho, umat, Emat, ustar, x))
    for (size_t i = ideb_; i <= ifin_; i++) {
        flt_t dx = x[i+1] - x[i];
        flt_t dX = dx + dt * (ustar[i+1] - ustar[i]);

        flt_t tmp_rho_  = (dX * rho[i]           - (tmp_rho[i+1]  - tmp_rho[i] )) / dx;
        flt_t tmp_urho_ = (dX * rho[i] * umat[i] - (tmp_urho[i+1] - tmp_urho[i])) / dx;
        flt_t tmp_Erho_ = (dX * rho[i] * Emat[i] - (tmp_Erho[i+1] - tmp_Erho[i])) / dx;

        rho[i]  = tmp_rho_;
        umat[i] = tmp_urho_ / tmp_rho_;
        Emat[i] = tmp_Erho_ / tmp_rho_;
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
OMP_FOR(firstprivate(ideb, ifin) shared(x, cmat) reduction(min:dt))
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


std::tuple<flt_t, flt_t> acoustic_Godunov(
        flt_t rho_i, flt_t rho_im, flt_t c_i, flt_t c_im,
        flt_t u_i,   flt_t u_im,   flt_t p_i, flt_t p_im)
{
    flt_t rc_l = rho_im * c_im;
    flt_t rc_r = rho_i  * c_i;
    flt_t ustar_i = (rc_l * u_im + rc_r * u_i +               (p_im - p_i)) / (rc_l + rc_r);
    flt_t pstar_i = (rc_r * p_im + rc_l * p_i + rc_l * rc_r * (u_im - u_i)) / (rc_l + rc_r);
    return std::make_tuple(ustar_i, pstar_i);
}


void acoustic()
{
    const size_t ideb_ = ideb - 1, ifin_ = ifin + 1;
OMP_FOR(firstprivate(ideb_, ifin_) shared(rho, cmat, ustar, umat, pmat, pstar))
    for (size_t i = ideb_; i <= ifin_; i++) {
        auto [ustar_i, pstar_i] = acoustic_Godunov(
             rho[i],  rho[i-1], cmat[i], cmat[i-1],
            umat[i], umat[i-1], pmat[i], pmat[i-1]);
        ustar[i] = ustar_i;
        pstar[i] = pstar_i;
    }
}


void acoustic_GAD(flt_t dt)
{
    const size_t ideb_ = ideb - 1, ifin_ = ifin + 1;
OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(x, rho, cmat, umat, pmat, ustar, pstar))
    for (size_t i = ideb_; i <= ifin_; i++) {
        // First order acoustic solver on the left cell
        auto [ustar_im, pstar_im] = acoustic_Godunov(
             rho[i-1],  rho[i-2], cmat[i-1], cmat[i-2],
            umat[i-1], umat[i-2], pmat[i-1], pmat[i-2]);

        // First order acoustic solver on the current cell
        auto [ustar_i, pstar_i] = acoustic_Godunov(
             rho[i],  rho[i-1], cmat[i], cmat[i-1],
            umat[i], umat[i-1], pmat[i], pmat[i-1]);

        // First order acoustic solver on the right cell
        auto [ustar_ip, pstar_ip] = acoustic_Godunov(
             rho[i+1],  rho[i], cmat[i+1], cmat[i],
            umat[i+1], umat[i], pmat[i+1], pmat[i]);

        flt_t r_um = phi((ustar_ip -   umat[i]) / (ustar_i - umat[i-1] + flt_t(1e-6)));
        flt_t r_pm = phi((pstar_ip -   pmat[i]) / (pstar_i - pmat[i-1] + flt_t(1e-6)));
        flt_t r_up = phi((umat[i-1] - ustar_im) / (umat[i] - ustar_i   + flt_t(1e-6)));
        flt_t r_pp = phi((pmat[i-1] - pstar_im) / (pmat[i] - pstar_i   + flt_t(1e-6)));

        flt_t dm_l = rho[i-1] * (x[i] - x[i-1]);
        flt_t dm_r = rho[i]   * (x[i+1] - x[i]);
        flt_t Dm = (dm_l + dm_r) / 2;

        flt_t rc_l = rho[i-1] * cmat[i-1];
        flt_t rc_r = rho[i]   * cmat[i];
        flt_t theta = flt_t(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm));

        ustar[i] = ustar_i + theta * (r_up * (umat[i] - ustar_i) - r_um * (ustar_i - umat[i-1]));
        pstar[i] = pstar_i + theta * (r_pp * (pmat[i] - pstar_i) - r_pm * (pstar_i - pmat[i-1]));
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

OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(x, rho, umat, Emat, pstar, ustar))
    for (size_t i = ideb_; i < ifin_; i++) {
        flt_t dx = x[i+1] - x[i];
        flt_t dm = rho[i] * dx;
        rho[i] = dm / (dx + dt * (ustar[i+1] - ustar[i]));
        umat[i] += dt / dm * (pstar[i] - pstar[i+1]);
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1]);
    }
    
    if (!euler_projection) {
OMP_FOR(firstprivate(ideb_, ifin_, dt) shared(x, ustar))
        for (size_t i = ideb_; i <= ifin_; i++) {
            x[i] += dt * ustar[i];
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
            printf("Cycle = %d, dt = %#18.15g, t = %#18.15g\n", cycle, dt, t);
        }

        if (!std::isfinite(dt) || dt <= 0.) {
            printf("Error: dt has an invalid value: %f\n", dt);
            exit(1);
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
    --nghost N              Number of ghost cells
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
        else if (strcmp(argv[i], "--nghost") == 0) {
            nb_ghosts = (int) strtol(argv[i+1], nullptr, 10);
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
    x = Vector<flt_t>(size);
    X = Vector<flt_t>(size);
    rho = Vector<flt_t>(size);
    umat = Vector<flt_t>(size);
    emat = Vector<flt_t>(size);
    Emat = Vector<flt_t>(size);
    pmat = Vector<flt_t>(size);
    cmat = Vector<flt_t>(size);
    gmat = Vector<flt_t>(size);
    ustar = Vector<flt_t>(size);
    pstar = Vector<flt_t>(size);
    ustar_1 = Vector<flt_t>(size);
    pstar_1 = Vector<flt_t>(size);
    tmp_rho = Vector<flt_t>(size);
    tmp_urho = Vector<flt_t>(size);
    tmp_Erho = Vector<flt_t>(size);

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
        printf(" - std vector: %d\n", USE_STD_VECTOR),
        printf("\n");
        printf(" - test:       %s\n", (test == Test::Sod) ? "Sod" : "Bizarrium");
        printf(" - riemann:    %s\n", "acoustic");
        printf(" - scheme:     %s\n", (scheme == Scheme::Godunov) ? "Godunov" : "GAD-minmod");
        printf(" - nb cells:   %g\n", double(nb_cells));
        printf(" - nb ghosts:  %d\n", nb_ghosts);
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
