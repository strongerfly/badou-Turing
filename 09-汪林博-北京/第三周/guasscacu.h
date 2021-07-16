#ifndef GUASSCACU_H
#define GUASSCACU_H

#include <QObject>
#include <cassert>
#include <iostream>
#include <string>
#include <QDebug>
using namespace std;
#define TOL 1e-3f
#define MAX_SWEEPS 50
#define  NONZERO(x)    ( (x)>0.0001 || (x)<-0.0001 )

class Matrix
{
public:
    Matrix(int row, int col):m_row(row),m_col(col)
    {
        assert(m_row>0 && m_col >0);
        m_data= new double[m_row*m_col];
        for (int i=0 ; i<m_row*m_col; ++i)
        {
            m_data[i] = 0;
        }
    }

    virtual ~Matrix()
    {
        delete[] m_data;
    }

    int ROW() const
    {
        return m_row;
    }
    int COL() const
    {
        return m_col;
    }


    //basic element operation
    double GetElement(int row, int col) const
    {
        assert(row<m_row && col<m_col);
        return m_data[row*m_col+col];
    }
    double operator()(int row,int col) const
    {
        return GetElement(row,col);
    }

    double& GetElement(int row, int col)
    {
        assert(row<m_row && col<m_col);
        return m_data[row*m_col+col];
    }
    double& operator()(int row,int col)
    {
        return m_data[row*m_col+col];
    }
    // fill data//
    void FillData(double *Ddata,int nLength)
    {
        if (nLength > m_row*m_col) nLength = m_row*m_col;
        if(nLength <=0 )return;
        for(int i = 0; i < nLength; i++)
            m_data[i] = Ddata[i];
    }
    // average colum//
    void AverageColum()
    {
        double *pDatatmp = new double [m_col];
        for (int i=0 ; i<m_col; ++i)
        {
            pDatatmp[i] = 0;
        }
        // cacu averge //
        int nOffset = 0;
        for(int i = 0; i < m_row ; i++)
        {
            for(int j = 0; j < m_col; j++)
            {
                pDatatmp [j] += m_data[nOffset];
                nOffset ++;
            }
        }
        //    // sub average//
        for (int i=0 ; i<m_col; ++i)
        {
            //        strPrintInfo += QString("%1,").arg(pDatatmp[i]);
            pDatatmp[i] = pDatatmp[i]/m_row;
        }
        nOffset = 0;
        for(int i = 0; i < m_row ; i++)
        {
            for(int j = 0; j < m_col; j++)
            {
                m_data[nOffset] -= pDatatmp [j] ;
                //            strPrintInfo += QString("%1,").arg(m_data[nOffset]);
                //              qDebug()<< m_data[nOffset];
                //              std:cout<<  m_data[nOffset]<< std::endl;
                nOffset ++;
            }
            //         strPrintInfo += "\r\n";
            //        qDebug()<< "\r\n ";
        }
        delete pDatatmp;
    }
    //
    double* CacuCovarianceMatrix()
    {
        //    协方差矩阵
        //    第i列所有元素-i列均值
        //    * 第j列所有元素 -j列均值 / col-1//
        int nLength = m_col*m_col;
        double *pDatatmp = new double [nLength];
        AverageColum();
        int nOffset = 0;
        for(int i = 0 ; i < m_col;i++ )
        {
            for(int j = 0 ; j < m_col;j++ )
            {
                pDatatmp[nOffset]  =0;
                //遍历 i列 和 j 列//
                int mOffset = 0;
                for(int k = 0; k < m_row; k++)
                {
                    pDatatmp[nOffset]+=  m_data[mOffset+i]*m_data[mOffset+j];
                    mOffset += m_col;
                }
                pDatatmp[nOffset] /= m_col-1;
                strPrintInfo += QString("%1,").arg(pDatatmp[nOffset]);

                //                    qDebug()<< pDatatmp[nOffset];
                ++nOffset;
            }
            strPrintInfo += "\r\n";
            //                qDebug()<< "\r\n ";
        }
        return pDatatmp;
    }
    //basic row/col operation
    void SwitchRows(int row1,int row2)
    {
        assert(row1<m_row && row2<m_row);
        double tmp;
        for (int i=0; i<m_col; ++i)
        {
            tmp = GetElement(row1,i);
            GetElement(row1,i) = GetElement(row2,i);
            GetElement(row2,i)= tmp;
        }
    }
    void SwitchCols(int col1,int col2)
    {
        assert(col1<m_col && col2<m_col);
        double tmp;
        for (int i=0; i<m_row; ++i)
        {
            tmp = GetElement(i,col1);
            GetElement(i,col1) = GetElement(i,col2);
            GetElement(i,col2)= tmp;
        }
    }

    void TimesRow(int row,double times)
    {
        assert(row<m_row);
        for (int i=0; i<m_col; ++i)
        {
            GetElement(row,i) = GetElement(row,i)*times;
        }
    }

    void TimesCol(int col,double times)
    {
        assert(col<m_col);
        for (int i=0; i<m_row; ++i)
        {
            GetElement(i,col) = GetElement(i,col)*times;
        }
    }

    //row[newrow] = row[newrow] + times* row[row];
    void AddTimesRow(int newrow, int row,double times)
    {
        assert(newrow<m_row && row <m_row);
        for (int i=0; i<m_col; ++i)
        {
            GetElement(newrow,i) += GetElement(row,i)*times ;
        }
    }

    void AddTimesCol(int newcol,int col,double times)
    {
        assert(newcol<m_col && col<m_col);

        for (int i=0; i<m_row; ++i)
        {
            GetElement(i,newcol) += GetElement(i,col)*times;
        }
    }

    void Print(ostream& os) const
    {
        os<<endl;
        for (int i=0; i<m_row; ++i)
        {
            for (int j=0; j<m_col; ++j)
            {
                if (j==0)
                {
                    os<<'[';
                }
                else
                {
                    os<<',';
                }

                os.width(8);
                os<<GetElement(i,j);
                if (j==m_col -1 )
                {
                    os<<']';
                }

            }
            os<<endl;
        }
        os<<endl;
    }
    //    a:二维数组(n*n),只用到对角线以上的元素
    //    n:阶数
    //    d:输出参数，存放特征值
    //    v:输出参数，存放特征向量
    void jacobi(unsigned int n, double *a, double *d, double *v)
    {
        double onorm, dnorm;
        double b, dma, q, t, c, s;
        double atemp, vtemp, dtemp;
        register int i, j, k, l;


        // Set v to the identity matrix, set the vector d to contain the
        // diagonal elements of the matrix a
        d[0] = a[0];
        d[1] = a[4];
        d[2] = a[8];



        for (l = 1; l <= MAX_SWEEPS; l++)
        {
            // Set dnorm to be the maximum norm of the diagonal elements, set
            // onorm to the maximum norm of the off-diagonal elements

            dnorm = (double)fabs(d[0]) + (double)fabs(d[1]) + (double)fabs(d[2]);
            onorm = (double)fabs(a[1]) + (double)fabs(a[2]) + (double)fabs(a[5]);
            // Normal end point of this algorithm.
            if((onorm/dnorm) <= TOL)
            {
                return;
            }


            for (j = 1; j < static_cast<int>(n); j++)
            {
                for (i = 0; i <= j - 1; i++)
                {


                    b = a[n*i+j];
                    if(fabs(b) > 0.0f)
                    {
                        dma = d[j] - d[i];
                        if((fabs(dma) + fabs(b)) <= fabs(dma))
                            t = b / dma;
                        else
                        {
                            q = 0.5f * dma / b;
                            t = 1.0f/((double)fabs(q) + (double)sqrt(1.0f+q*q));
                            if (q < 0.0)
                                t = -t;
                        }


                        c = 1.0f/(double)sqrt(t*t + 1.0f);
                        s = t * c;
                        a[n*i+j] = 0.0f;


                        for (k = 0; k <= i-1; k++)
                        {
                            atemp = c * a[n*k+i] - s * a[n*k+j];
                            a[n*k+j] = s * a[n*k+i] + c * a[n*k+j];
                            a[n*k+i] = atemp;
                        }


                        for (k = i+1; k <= j-1; k++)
                        {
                            atemp = c * a[n*i+k] - s * a[n*k+j];
                            a[n*k+j] = s * a[n*i+k] + c * a[n*k+j];
                            a[n*i+k] = atemp;
                        }


                        for (k = j+1; k < static_cast<int>(n); k++)
                        {
                            atemp = c * a[n*i+k] - s * a[n*j+k];
                            a[n*j+k] = s * a[n*i+k] + c * a[n*j+k];
                            a[n*i+k] = atemp;
                        }


                        for (k = 0; k < static_cast<int>(n); k++)
                        {
                            vtemp = c * v[n*k+i] - s * v[n*k+j];
                            v[n*k+j] = s * v[n*k+i] + c * v[n*k+j];
                            v[n*k+i] = vtemp;
                        }


                        dtemp = c*c*d[i] + s*s*d[j] - 2.0f*c*s*b;
                        d[j] = s*s*d[i] + c*c*d[j] + 2.0f*c*s*b;
                        d[i] = dtemp;
                    } /* end if */
                } /* end for i */
            } /* end for j */
        } /* end for l */
    }
    //    * 利用雅格比(Jacobi)方法求实对称矩阵的所有特征值及特征向量
    //    * @param pMatrix				长度为n*n的数组。存放实对称矩阵
    //    * @param nDim					矩阵的阶数
    //    * @param pdblVects				长度为n*n的数组，返回特征向量(按列存储)
    //    * @param dbEps					精度要求
    //    * @param nJt					整型变量。控制最大迭代次数
    //    * @param pdbEigenValues			特征值数组
    bool JacbiCor(double * pMatrix,int nDim, double *pdblVects, double *pdbEigenValues, double dbEps,int nJt)
    {
        for(int i = 0; i < nDim; i ++)
        {
            pdblVects[i*nDim+i] = 1.0f;
            for(int j = 0; j < nDim; j ++)
            {
                if(i != j)
                    pdblVects[i*nDim+j]=0.0f;
            }
        }

        int nCount = 0;		//迭代次数
        while(1)
        {
            //在pMatrix的非对角线上找到最大元素
            double dbMax = pMatrix[1];
            int nRow = 0;
            int nCol = 1;
            for (int i = 0; i < nDim; i ++)			//行
            {
                for (int j = 0; j < nDim; j ++)		//列
                {
                    double d = fabs(pMatrix[i*nDim+j]);

                    if((i!=j) && (d> dbMax))
                    {
                        dbMax = d;
                        nRow = i;
                        nCol = j;
                    }
                }
            }

            if(dbMax < dbEps)     //精度符合要求
                break;

            if(nCount > nJt)       //迭代次数超过限制
                break;

            nCount++;

            double dbApp = pMatrix[nRow*nDim+nRow];
            double dbApq = pMatrix[nRow*nDim+nCol];
            double dbAqq = pMatrix[nCol*nDim+nCol];

            //计算旋转角度
            double dbAngle = 0.5*atan2(-2*dbApq,dbAqq-dbApp);
            double dbSinTheta = sin(dbAngle);
            double dbCosTheta = cos(dbAngle);
            double dbSin2Theta = sin(2*dbAngle);
            double dbCos2Theta = cos(2*dbAngle);

            pMatrix[nRow*nDim+nRow] = dbApp*dbCosTheta*dbCosTheta +
                    dbAqq*dbSinTheta*dbSinTheta + 2*dbApq*dbCosTheta*dbSinTheta;
            pMatrix[nCol*nDim+nCol] = dbApp*dbSinTheta*dbSinTheta +
                    dbAqq*dbCosTheta*dbCosTheta - 2*dbApq*dbCosTheta*dbSinTheta;
            pMatrix[nRow*nDim+nCol] = 0.5*(dbAqq-dbApp)*dbSin2Theta + dbApq*dbCos2Theta;
            pMatrix[nCol*nDim+nRow] = pMatrix[nRow*nDim+nCol];

            for(int i = 0; i < nDim; i ++)
            {
                if((i!=nCol) && (i!=nRow))
                {
                    int u = i*nDim + nRow;	//p
                    int w = i*nDim + nCol;	//q
                    dbMax = pMatrix[u];
                    pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta;
                    pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta;
                }
            }

            for (int j = 0; j < nDim; j ++)
            {
                if((j!=nCol) && (j!=nRow))
                {
                    int u = nRow*nDim + j;	//p
                    int w = nCol*nDim + j;	//q
                    dbMax = pMatrix[u];
                    pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta;
                    pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta;
                }
            }

            //计算特征向量
            for(int i = 0; i < nDim; i ++)
            {
                int u = i*nDim + nRow;		//p
                int w = i*nDim + nCol;		//q
                dbMax = pdblVects[u];
                pdblVects[u] = pdblVects[w]*dbSinTheta + dbMax*dbCosTheta;
                pdblVects[w] = pdblVects[w]*dbCosTheta - dbMax*dbSinTheta;
            }

        }

        //对特征值进行排序以及又一次排列特征向量,特征值即pMatrix主对角线上的元素
        std::map<double,int> mapEigen;
        for(int i = 0; i < nDim; i ++)
        {
            pdbEigenValues[i] = pMatrix[i*nDim+i];

            mapEigen.insert(make_pair( pdbEigenValues[i],i ) );
        }

        double *pdbTmpVec = new double[nDim*nDim];
        std::map<double,int>::reverse_iterator iter = mapEigen.rbegin();
        for (int j = 0; iter != mapEigen.rend(),j < nDim; ++iter,++j)
        {
            for (int i = 0; i < nDim; i ++)
            {
                pdbTmpVec[i*nDim+j] = pdblVects[i*nDim + iter->second];
            }

            //特征值又一次排列
            pdbEigenValues[j] = iter->first;
        }

        //设定正负号
        for(int i = 0; i < nDim; i ++)
        {
            double dSumVec = 0;
            for(int j = 0; j < nDim; j ++)
                dSumVec += pdbTmpVec[j * nDim + i];
            if(dSumVec<0)
            {
                for(int j = 0;j < nDim; j ++)
                    pdbTmpVec[j * nDim + i] *= -1;
            }
        }

        memcpy(pdblVects,pdbTmpVec,sizeof(double)*nDim*nDim);
        delete []pdbTmpVec;

        return 1;
    }

    // tmp..
    //
    QString strPrintInfo = "";
protected:
private:
    Matrix& operator = (const Matrix& right);
    Matrix(const Matrix& right);
    int m_row,m_col;
    double* m_data;
};
class GuassCacu /*: public QObject*/
{
    //    Q_OBJECT
public:
    //    explicit GuassCacu(QObject *parent = nullptr);
    GuassCacu(Matrix& matrix):m_matrix(matrix)
    {
        assert(m_matrix.ROW() == m_matrix.COL() - 1);
        TraceInit();
    }
    ~GuassCacu()
    {
        TraceUnInit();
    }

    void Calculate()
    {
        int i = 0;
        for( ; i<m_matrix.ROW() ; ++i)
        {
            if (FindFirstNotZero(i))
            {
                CleanMatrix(i);
            }
            else
                break;
        }

        FinalCleanMatrix();


        //这里是最终结果:
        if (i = m_matrix.ROW() - 1)
        {
            resultinfo = "the abs(matrix) isn't zero, so only one result ";
            for(i=0;  i<m_matrix.ROW(); ++i)
            {
                //this is the value of result
                m_result[m_coltrace[i]] = m_matrix(i,m_matrix.COL());
            }
        }
        else
        {
            resultinfo = "the abs(matrix0 == 0, so need check elements";
        }

    }
    bool FindFirstNotZero(int colstart)
    {
        for (int col= colstart; col<m_matrix.ROW(); ++col)
        {
            for (int row = colstart; row<m_matrix.ROW(); ++row)
            {
                if (NONZERO(m_matrix(row,col) ))
                {
                    if (col != colstart)
                    {
                        m_matrix.SwitchCols(col,colstart);
                        TraceSwitchCol(colstart,col);
                    }
                    m_matrix.SwitchRows(colstart,row);
                    return true;
                }
            }
        }
        return false;
    }

    void CleanMatrix(int colstart)
    {
        assert(NONZERO(m_matrix(colstart,colstart)));
        for (int row=0; row<m_matrix.ROW(); ++row)
        {
            if (row !=colstart)
            {
                m_matrix.AddTimesRow(row,colstart,(-1)*m_matrix(row,colstart)/m_matrix(colstart,colstart));
            }
            else
            {
                m_matrix.TimesRow(row,1/m_matrix(colstart,colstart));
            }
        }
    }

    void FinalCleanMatrix()
    {
        for (int row=0; row<m_matrix.ROW(); ++row)
            for(int col =0; col<m_matrix.COL(); ++col)
                m_matrix(row,col) = NONZERO(m_matrix(row,col)) ? m_matrix(row,col) : 0;
    }

    string GetResultInfo()
    {
        return resultinfo;
    }

    double *GetResultPointer()
    { return m_result;}




private:
    void TraceSwitchCol(int col1,int col2)
    {
        int tmp = m_coltrace[col1];
        m_coltrace[col1] = m_coltrace[col2];
        m_coltrace[col2] = tmp;
    }
    void TraceInit()
    {
        m_coltrace = new int[m_matrix.COL()];
        m_result = new double[m_matrix.COL()];
        int i=0;
        while(i<m_matrix.COL())
            m_coltrace[i] = i++;
    }
    void TraceUnInit()
    {
        delete [] m_coltrace;
        delete [] m_result;
    }
    Matrix& m_matrix;
    int* m_coltrace;
    double* m_result;
    ostream* pObserver;
private:
    string resultinfo;
    GuassCacu& operator = (const GuassCacu& right);
    GuassCacu(const GuassCacu& right);
    //signals:

    //public slots:
};

#endif // GUASSCACU_H
