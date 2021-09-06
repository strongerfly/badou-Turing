#include "guasscacu.h"



ostream& operator<<(ostream& os, const Matrix& value)
{
    value.Print(os);
    return os;
}


//void Matrix::AverageColum()

//GuassCacu::GuassCacu(QObject *parent) : QObject(parent)
//{

//}
//double* Matrix::CacuCovarianceMatrix()

