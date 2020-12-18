#ifndef CF_2_H
#define CF_2_H

#include <math.h>

class CF_2
{
public:
    virtual double operator()(double   x,double   y) const = 0;
};


class velocity_X : CF_2{
public:
    double operator()(double x, double y) const{
        return -y;
    }
};
class velocity_Y : CF_2{
public:
    double operator()(double x, double y) const{
        return x;
    }
};
class test_function : CF_2{
public:
    double operator() (double x, double y) const{
        return cos(x) + sin(y);
    }
};

#endif // CF_2_H
