#include "grid2d.h"


Grid2D::Grid2D()
{

}

Grid2D::Grid2D(int N_, int M_, double xmin_, double xmax_, double ymin_, double ymax_)
{
    N = N_;
    M = M_;
    xmin = xmin_;
    ymin = ymin_;
    xmax = xmax_;
    ymax = ymax_;

    dx = (xmax - xmin) / (double) (N - 1);
    dy = (ymax - ymin) / (double) (M - 1);

}

double Grid2D::get_dx() const
{
    return dx;
}

double Grid2D::get_dy() const
{
    return dy;
}

double Grid2D::get_xmin() const
{
    return xmin;
}

double Grid2D::get_ymin() const
{
    return ymin;
}

int Grid2D::get_N() const
{
    return N;
}

int Grid2D::get_M() const
{
    return M;
}

int Grid2D::i_from_n(int n) const
{
    return n % N;
}

int Grid2D::j_from_n(int n) const
{
    return n / N;
}

int Grid2D::n_from_ij(int i, int j) const
{
    return i + j * N ;
}

double Grid2D::x_from_n(int n) const
{
    return  xmin + dx*i_from_n(n);
}

double Grid2D::y_from_n(int n) const
{
    return  ymin + dy*j_from_n(n);
}


double Grid2D::dx_forward(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M )
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    // if we're on the boundry the derivative is 0!
    //else if ( y_from_n(n) == ymin || y_from_n(n) == ymax ||  x_from_n(n) == xmin || x_from_n(n) == xmax)
    else if ( x_from_n(n) == xmin || x_from_n(n) == xmax)
    {
        return 0;
    }
    else
    {
        return ( function[n+1] - function[n] ) / dx;
    }
}

double Grid2D::dx_backward(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M)
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    // if we're on the boundry the derivative is 0!
    //else if ( y_from_n(n) == ymin || y_from_n(n) == ymax ||  x_from_n(n) == xmin || x_from_n(n) == xmax)
    else if ( x_from_n(n) == xmin || x_from_n(n) == xmax)
    {
        return 0;
    }
    else
    {
        return ( function[n] - function[n-1] ) / dx;
    }
}

double Grid2D::dy_forward(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M)
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    // if we're on the boundry the derivative is 0!
    //else if ( y_from_n(n) == ymin || y_from_n(n) == ymax ||  x_from_n(n) == xmin || x_from_n(n) == xmax)
    else if ( y_from_n(n) == ymin || y_from_n(n) == ymax)
    {
        return 0;
    }
    else
    {
        return ( function[n+N] - function[n] ) / dy;
    }
}

double Grid2D::dy_backward(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M)
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    // if we're on the boundry the derivative is 0!
    //else if ( y_from_n(n) == ymin || y_from_n(n) == ymax ||  x_from_n(n) == xmin || x_from_n(n) == xmax)
    else if ( y_from_n(n) == ymin || y_from_n(n) == ymax)
    {
        return 0;
    }
    else
    {
        return ( function[n] - function[n-N] ) / dy;
    }
}

double Grid2D::dx_2(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M)
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    else if (x_from_n(n) == xmin || x_from_n(n) == xmax)
    {
        return 0;
    }
    {
        return ( function[n+1] - 2* function[n]  + function[n-1] ) / dx*dx;
    }
}

double Grid2D::dy_2(std::vector <double> & function, int n)
{
    if ( (int) function.size() != N*M)
    {
        std::cout << "Error!" << std::endl;
        return 0;
    }
    else if ( y_from_n(n) == ymin || y_from_n(n) == ymax || y_from_n(n+N) == ymin || y_from_n(n-N) == ymax)
    {
        return 0;
    }
    else
    {
        return ( function[n+N] - 2* function[n]  + function[n-N] ) / dy*dy;
    }
}

