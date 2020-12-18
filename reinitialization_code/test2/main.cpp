#include <iostream>
#include <fstream>
#include <math.h>
#include <cf_2.h>
#include <grid2d.h>
#include <vector>
#include <ls_reinitialization.h>

using namespace std;

double three_pedals(double x, double y)
{
    double r = sqrt(x * x + y * y);
    double theta = atan2(y, x);
    return r - 0.05 * (cos(3 * theta)) - 0.15;
}

int main()
{

//    int N = 107;
//    double xmin = -0.207843;
//    double xmax = 0.207843;
//    double ymin = -0.207843;
//    double ymax = 0.207843;

    int N = 256;
    double xmin = -0.5;
    double xmax = 0.5;
    double ymin = -0.5;
    double ymax = 0.5;

    double x0 = 0.0;
    double y0 = 0.0;

    Grid2D grid(N, N, xmin, xmax, ymin, ymax);
    int reinit_steps = 10;
    double dt = grid.get_dx() * grid.get_dx();

    vector<double> solution(N*N);
    for(int i = 0 ; i < N * N ; i++)
    {
        double x = grid.x_from_n(i);
        double y = grid.y_from_n(i);

        solution[i] = three_pedals(x,y);
    }

    // ============= reinitialization =============
    LS_reinitialization lsr;
    lsr.set_grid(grid);
    lsr.set_steps(reinit_steps);

    lsr.set_solution(solution);
    lsr.reinit();

    vector<double> reinit_solution(N*N);
    lsr.get_solution(reinit_solution);

    for (int i = 0 ; i < N*N ; i++){
    cout << reinit_solution[i] << " " ;
    }

    cout << lsr.norm_grad();

    // ============= save data =============

    ofstream outdata;
    outdata.open("solution.txt");
    for(int i = 0 ; i < N*N ; i++)
    {
        outdata << reinit_solution[i] << endl;
    }
    outdata.close();

//    cout << "Hello World!" << endl;
    return 0;
}
