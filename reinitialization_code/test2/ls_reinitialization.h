#ifndef LS_REINITIALIZATION_H
#define LS_REINITIALIZATION_H

#include "grid2d.h"
#include <vector>
#include <cf_2.h>

class LS_reinitialization
{

private:
    Grid2D grid;
    std::vector<double> solution;
    std::vector<double> phi0;
    double dt;
    int steps;

    void one_Step(double dt_in);
    void set_dt(double dxin);
    double get_phi(double sgnphi0, double phixp, double phixm);
    double cfl_cond();
    double signum(double phi_val, double sgn_band_tol);

public:
    LS_reinitialization();
    void set_solution(std::vector<double> &solution_in);
    void set_grid(Grid2D &grid_in);
    void set_steps(int steps_in);
    void get_solution(std::vector<double> &solution_in);
    void reinit();
    double norm_grad();
};

#endif // LS_REINITIALIZATION_H
