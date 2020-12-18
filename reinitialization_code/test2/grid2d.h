#ifndef GRID2D_H
#define GRID2D_H

#include <iostream>
#include <vector>


class Grid2D
{
private:
    int M;
    double xmin, xmax, ymin, ymax;

public:
    double dx, dy;
public:
    Grid2D();
    int N;
    // underscore so we don't mix up input and class parameters
    Grid2D(int N_, int M_, double xmin_, double xmax_, double ymin_, double ymax_);
    // because dx and dy are private we need function
    double get_dx() const;
    double get_dy() const;
    double get_xmin() const;
    double get_ymin() const;
    int get_N() const;
    int get_M() const;
    // these functionss move between vector and 2d space
    int i_from_n(int n) const;
    int j_from_n(int n) const;
    int n_from_ij(int i, int j) const;
    // these function give us phyical location on the grid
    double x_from_n(int n) const;
    double y_from_n(int n) const;
    // these are derivatives 1st and 2nd order (also forward and backward)
    double dx_forward(std::vector <double> & function, int n);
    double dx_backward(std::vector <double> & function, int n);
    double dy_forward(std::vector <double> & function, int n);
    double dy_backward(std::vector <double> & function, int n);
    double dx_2(std::vector <double> & function, int n);
    double dy_2(std::vector <double> & function, int n);


    void initialize_VTK_file(std::string file_name);
    void print_VTK_Format( std::vector<double> &F, std::string data_name, std::string file_name );
};

#endif // GRID2D_H
