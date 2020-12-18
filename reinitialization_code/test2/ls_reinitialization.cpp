#include "ls_reinitialization.h"

LS_reinitialization::LS_reinitialization()
{

}

void LS_reinitialization::set_grid(Grid2D &grid_in)
{
    grid = grid_in;
}

void LS_reinitialization::set_solution(std::vector<double> &solution_in)
{
    solution = solution_in;
    phi0 = solution_in;
}

void LS_reinitialization::set_dt(double dxin)
{
    dt = dxin;
}

void LS_reinitialization::set_steps(int steps_in)
{
    steps = steps_in;
}

void LS_reinitialization::get_solution(std::vector<double> &solution_in)
{
    solution_in = solution;
}

double LS_reinitialization::get_phi(double sgnphi0, double phip, double phim)
{
    if((sgnphi0*phim <= 0.) && (sgnphi0*phip <= 0.)){
        return phip;
    }else if((sgnphi0*phim >= 0.) && (sgnphi0*phip >= 0.)){
        return phim;
    }else if((sgnphi0*phim <= 0.) && (sgnphi0*phip >= 0.)){
        return 0.;
    }else if((sgnphi0*phim >= 0.) && (sgnphi0*phip <= 0.)){
        if(abs(phim) >= abs(phip)){
            return phim;
        }else{
            return phip;
        }
    }
    return 0.;
}

double LS_reinitialization::signum(double phi_val, double sgn_band_tol)
{
    if (phi_val < 0){
        return -1.;
    } else if(phi_val >0){
        return 1.;
    } else{
        return 0.;
    }
}

void LS_reinitialization::one_Step(double dt_in)
{

#pragma omp parallel for
    for(int n = 0; n < (grid.get_N() * grid.get_M()); n++){

        int i = grid.i_from_n(n);
        int j = grid.j_from_n(n);

        double sgn_band_tol = 0.005*grid.get_dx();
        double sgnphi0 = signum(phi0[n],sgn_band_tol);

        int boundary_checker = 0;

        // check if on the boundary
        // first
        // use no flux boundary conditions
        if((i == 0) && (j == 0)){
            // bottom left corner
            boundary_checker = 1;
        }

        if((j == 0) && (i > 0) && (i < grid.get_N()-1)){
            // bottom wall but not a corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j == 0)){
            // bottom right corner
            boundary_checker = 1;
        }

        if((i == 0) && (j > 0) && (j < grid.get_M()-1)){
            // left wall but not a corner
            boundary_checker = 1;
        }

        if((i == 0) && (j == grid.get_M()-1)){
            // top left corner
            boundary_checker = 1;
        }

        if((j == grid.get_M()-1) && (i > 0) && (i < grid.get_N()-1)){
            // top wall but not a corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j == grid.get_M()-1)){
            // top right corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j > 0) && (j < grid.get_M()-1)){
            // right wall but not a corner
            boundary_checker = 1;
        }

        if(boundary_checker == 0){
            // interior points

            // first compute forward and backward differences
            double phixp = grid.dx_forward(solution,n);
            double phixm = grid.dx_backward(solution,n);
            double phiyp = grid.dy_forward(solution,n);
            double phiym = grid.dy_backward(solution,n);

            double phixc = get_phi(sgnphi0,phixp,phixm);
            double phiyc = get_phi(sgnphi0,phiyp,phiym);

            solution[n] += dt_in * sgnphi0 * (1 - sqrt( phixc*phixc + phiyc*phiyc ));

        }else{
            // boundary point
            solution[n] += dt_in*sgnphi0;
        }
    }

}

void LS_reinitialization::reinit()
{
    set_dt( 0.5 * cfl_cond() );

    for(int n = 0; n < steps; n++){
        one_Step(dt);
    }
}

double LS_reinitialization::cfl_cond()
{
    // returns the value dx/||v||_{\infty}
    // where v = sgn(phi_0) del(phi) / |del(phi)| is the velocity of the Hamilton Jacobi equation

    double v_inf = 0.;
    double v_inf_temp = 0.;
    double sgn_band_tol = 0.005*grid.get_dx();

#pragma omp parallel for
    for(int n = 0; n < (grid.get_N() * grid.get_M()); n++){
        v_inf_temp = abs( signum(phi0[n],sgn_band_tol) );
        if( v_inf_temp > v_inf){
            v_inf = v_inf_temp;
        }
    }

    double eps = 0.0001;
    return grid.get_dx() / ( v_inf + eps );
}


double LS_reinitialization::norm_grad()
{

int int_node_counter = 0;
double grad = 0.;

#pragma omp parallel for
    for(int n = 0; n < (grid.get_N() * grid.get_M()); n++){

        int i = grid.i_from_n(n);
        int j = grid.j_from_n(n);

        int boundary_checker = 0;

        // check if on the boundary first
        // use no flux boundary conditions
        if((i == 0) && (j == 0)){
            // bottom left corner
            boundary_checker = 1;
        }

        if((j == 0) && (i > 0) && (i < grid.get_N()-1)){
            // bottom wall but not a corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j == 0)){
            // bottom right corner
            boundary_checker = 1;
        }

        if((i == 0) && (j > 0) && (j < grid.get_M()-1)){
            // left wall but not a corner
            boundary_checker = 1;
        }

        if((i == 0) && (j == grid.get_M()-1)){
            // top left corner
            boundary_checker = 1;
        }

        if((j == grid.get_M()-1) && (i > 0) && (i < grid.get_N()-1)){
            // top wall but not a corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j == grid.get_M()-1)){
            // top right corner
            boundary_checker = 1;
        }

        if((i == grid.get_N()-1) && (j > 0) && (j < grid.get_M()-1)){
            // right wall but not a corner
            boundary_checker = 1;
        }

        if(boundary_checker == 0){
            // interior points

            // first compute forward and backward differences
            double phixp = grid.dx_forward(solution,n);
            double phixm = grid.dx_backward(solution,n);
            double phiyp = grid.dy_forward(solution,n);
            double phiym = grid.dy_backward(solution,n);

            double phix = 0.5*(phixp + phixm);
            double phiy = 0.5*(phiyp + phiym);

            grad += pow((phix*phix + phiy*phiy),0.5);
            int_node_counter ++;
        }
    }
    grad = grad/((double) int_node_counter);
    std::cout<<"int node counter "<<int_node_counter<<std::endl;
    return grad;


    /*
    int n = 30;
    double phixp = grid.dx_forward(phi0,n);
    double phixm = grid.dx_backward(phi0,n);
    double phiyp = grid.dy_forward(phi0,n);
    double phiym = grid.dy_backward(phi0,n);

    double phix = 0.5*(phixp + phixm);
    double phiy = 0.5*(phiyp + phiym);

    double grad = pow((phix*phix + phiy*phiy),0.5);
    return grad;
    */
}
