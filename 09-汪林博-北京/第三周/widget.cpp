#include "widget.h"
#include "ui_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <QPainter>
#include <QDir>
#include <QMessageBox>
#include  <iostream>
#include "guasscacu.h"
using namespace std;
Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    m_rawpic = m_resultpic = NULL;
    QString file = QDir::currentPath()+"//lenna.png";
    //    m_picLenna.load(file);
    m_picLenna.load("lenna.png");

    if(!m_picLenna.isNull())m_rawpic = &m_picLenna;
    ui->setupUi(this);
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QHBoxLayout *topLayout = new QHBoxLayout;
    QHBoxLayout *bottomLayout = new QHBoxLayout;
    topLayout->addWidget(ui->pushButtonhist);
    topLayout->addWidget(ui->pushButtonconvolution);
    topLayout->addWidget(ui->pushButtonpca);
    ui->widgetLeft->setGeometry(0,20,300,300);
    ui->widgetRight         ->setGeometry(300,20,300,300);

    bottomLayout->addWidget(ui->widgetLeft);
    bottomLayout->addWidget(ui->widgetRight);
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(bottomLayout);

    setLayout(mainLayout);
    connect(ui->pushButtonhist, SIGNAL(clicked()), this, SLOT(btnHist()));
    connect(ui->pushButtonconvolution, SIGNAL(clicked()), this, SLOT(btnConvolution()));
    connect(ui->pushButtonpca, SIGNAL(clicked()), this, SLOT(btnPca ()));
}

Widget::~Widget()

{
    delete ui;
}
void Widget::turnGrid()
{
    int outWidth, outHeight;
    outHeight = m_picLenna.height();
    outWidth = m_picLenna.width();
    QRgb *line;
    QImage *newimage = new QImage(outWidth,outHeight,QImage::Format_ARGB32);
    QImage origin = m_picLenna.toImage();
    for(int y = 0; y < newimage->height();y++)
    {
        QRgb * line = (QRgb *)origin.scanLine(y);

        for(int x = 0; x<newimage->width(); x++){
            int average = (int)(qRed(line[x])*0.3 + qGreen(line[x])*0.59 + qBlue(line[x])*0.11);
            newimage->setPixel(x,y, qRgb(average, average, average));
        }

    }
    m_picOut = QPixmap::fromImage(*newimage );
    m_resultpic = &m_picOut;
}

void Widget:: Hist()
{//灰度图直方图均衡化//
    turnGrid();
    int outWidth, outHeight;
    outHeight = m_picLenna.height();
    outWidth = m_picLenna.width();
    QRgb *line;
    QImage *newimage = new QImage(outWidth,outHeight,QImage::Format_ARGB32);

    QImage origin = m_picOut.toImage();
    // cacu cnt //
    int grad[256] = {0} ;
    int hist[256] = {0} ;
    int cnt  =0 ;
    for(int y = 0; y < newimage->height();y++)
    {
        QRgb * line = (QRgb *)origin.scanLine(y);

        for(int x = 0; x<newimage->width(); x++){
            grad   [qRed(line[x   ])]++;
            cnt++;
        }
    }
    //     std::cout <<  "total cnt"<< cnt << std::endl;
    float cnt1 = 0.000001;
    for(int i = 0; i < 256; i++)
    {

        hist[i] =(int) (((grad[i]+cnt1)*256)/cnt)-1;
        if(hist[i] <0)hist[i] =0;
        //                std::cout <<  hist[i] << std::endl;
        cnt1+= grad[i];
    }
    for(int y = 0; y < newimage->height();y++)
    {
        QRgb * line = (QRgb *)origin.scanLine(y);

        for(int x = 0; x<newimage->width(); x++){
            int average =hist[qRed(line[x])];
            newimage->setPixel(x,y, qRgb(average, average, average));
        }
    }
    m_picOut = QPixmap::fromImage(*newimage );
    m_resultpic = &m_picOut;
}
void Widget::convolution(char *nuclear,int matrixsize,int pad)
{//先测试pad= 0、、matrixsize =3//
    matrixsize =3;pad= 1;
    int nMatrixlength = matrixsize *matrixsize;
    int outWidth, outHeight;
    outHeight = m_picLenna.height();
    outWidth = m_picLenna.width();
    QRgb *line;
    QImage *rawimage = new QImage(outWidth+pad+pad,outHeight+pad+pad,QImage::Format_ARGB32);
    QImage origin = m_picLenna.toImage();
    // add pad//
    for(int y = 0; y < rawimage->height();y++)
    {
        line = NULL;
        if(y >=pad&& y < outHeight+pad)
            line = (QRgb *)origin.scanLine(y-pad);
        for(int x = 0; x<rawimage->width(); x++)
        {
            int r ,g,b;
            if(x< pad ||y < pad || x>=outWidth+pad || y >= outHeight+pad)
                rawimage->setPixel(x,y, qRgb(0, 0, 0));
            else
            {
                if(line !=NULL && x>=pad && x < outWidth+pad)
                    rawimage->setPixel(x,y, qRgb(qRed(line[x-pad]), qGreen(line[x-pad]), qBlue(line[x-pad])));
                else rawimage->setPixel(x,y, qRgb(0, 0, 0));
            }
        }
    }
    QImage *newimage = new QImage(outWidth,outHeight,QImage::Format_ARGB32);

    //    m_picOut = QPixmap::fromImage(*rawimage );
    //    m_resultpic = &m_picOut;
    QRgb * line0 = (QRgb *)rawimage->scanLine(0);
    QRgb * line1 = (QRgb *)rawimage->scanLine(1);
    QRgb * line2 = (QRgb *)rawimage->scanLine(2);
    //    QRgb * lineTemp = NULL;
    for(int y = 0; y < newimage->height();y++)
    {
        //        QRgb * line = (QRgb *)origin.scanLine(y);
        for(int x = 0; x<newimage->width(); x++)
        {
            int r ,g,b;
            //            r = qRed(line0[x]);
            //            g = qRed(line0[x+1]);
            //              b = qRed(line0[x+2]);
            //            if(x+2 < rawimage->width())
            //            {
            r =(int)( nuclear[0]*qRed(line0[x])+ nuclear[1]*qRed(line0[x+1])+nuclear[2]*qRed(line0[x+ 2 ])
                    +nuclear[3]*qRed(line1[x])+ nuclear[4]*qRed(line1[x+1])+nuclear[5]*qRed(line1[x+ 2 ])
                    +nuclear[6]*qRed(line2[x])+ nuclear[7]*qRed(line2[x+1])+nuclear[8]*qRed(line2[x+ 2 ]));
            g = (int)(nuclear[0]*qGreen(line0[x])+ nuclear[1]*qGreen(line0[x+1])+nuclear[2]*qGreen(line0[x+ 2 ])
                    +nuclear[3]*qGreen(line1[x])+ nuclear[4]*qGreen(line1[x+1])+nuclear[5]*qGreen(line1[x+ 2 ])
                    +nuclear[6]*qGreen(line2[x])+ nuclear[7]*qGreen(line2[x+1])+nuclear[8]*qGreen(line2[x+ 2 ]));
            b = (int)(nuclear[0]*qBlue(line0[x])+ nuclear[1]*qBlue(line0[x+1])+nuclear[2]*qBlue(line0[x+ 2 ])
                    +nuclear[3]*qBlue(line1[x])+ nuclear[4]*qBlue(line1[x+1])+nuclear[5]*qBlue(line1[x+ 2 ])
                    +nuclear[6]*qBlue(line2[x])+ nuclear[7]*qBlue(line2[x+1])+nuclear[8]*qBlue(line2[x+ 2 ]));
            newimage->setPixel(x,y, qRgb(r, g, b));
            //            }

            //                                newimage->setPixel(x,y, qRgb(0,0,0));
        }
        line0 = line1; line1 = line2;

        line2 = (QRgb *)rawimage->scanLine(y+3);
    }
    m_picOut = QPixmap::fromImage(*newimage );
    m_resultpic = &m_picOut;

}
void Widget::pca(int dim)
{


    //     0 均值化
//    协方差矩阵
//    特征值 特征向量
//    特征值排序 剔除部分特征值 剩下特征向量组成转换矩阵
//    原数据乘以转换矩阵实现降维
//        [2, 0, -1.4], [2.2, 0.2, -1.5], [2.4, 0.1, -1], [1.9, 0, -1.2]

//        [ 3.   1.  4.  1.  5.]
//         [ 1.   3.  5.  7.  9.]
//         [ 0.  -1.  2.  0.  5.]
//         [ 9.   1. 10.   5. -4.
//    QString strInfo  = "";
//    strLeftInfo += "初始矩阵为";
    Matrix matrix(4,3);
    double _data[12] =    { 2,0,-1.4,
                            2.2,0.2,-1.5,
                            2.4,0.1,-1,
                            1.9,0,-1.2};
    //   double _data[20] =    { 3,1,4,1,5,
    //                        1,3,5,7,9,
    //                        0,-1,2,0,5,
    //                        9,1,10,5,-4};
    matrix.FillData(_data,12);
    int nOffset = 0;
strLeftInfo += "\r\n原始矩阵为 \r\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j <4; j++)
        {
            strLeftInfo += QString("%1,").arg(_data[nOffset]);
            nOffset ++;
        }
        strLeftInfo += "\r\n";
    }
    // 0 均值化

    matrix.AverageColum();
    //协方差矩阵
    double * _convaria ;
    _convaria = matrix.CacuCovarianceMatrix();
    strLeftInfo += "协方差矩阵为\r\n";
     nOffset = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j <3; j++)
        {
            strLeftInfo += QString("%1,").arg(_convaria[nOffset]);
            nOffset ++;
        }
        strLeftInfo += "\r\n";
    }
    //特征值 特征向量
    double dfValue[3];
    double dfMatrix[9];
    matrix.JacbiCor(_convaria,3,dfMatrix,dfValue,0.00001,50);
    //特征值排序 剔除部分特征值 剩下特征向量组成转换矩阵
    strLeftInfo += "\r\n特征值为 \r\n";
    for(int i = 0; i < 3; i++)
    {
            strLeftInfo += QString("%1,").arg(dfValue[i]);
    }
    //取最大值 和对应的特征向量//
    int nSel = 0;
    double selVect[3];
    for(int i = 0; i < 2; i++)
    {
        if(dfValue[nSel] <dfValue[i+1])nSel = i+1;
    }
    //对应的特征向量//
    for(int i =0; i < 3; i++)
        selVect [i] = dfMatrix[i*3+nSel];
    strLeftInfo += "\r\n选中特征值和向量为\r\n";

            strLeftInfo += QString("%1\r\n").arg(dfValue[nSel]);
    for(int i = 0; i < 3; i++)
    {
            strLeftInfo += QString("%1,").arg(dfMatrix[i]);
    }
    //原数据乘以转换矩阵实现降维
    //_data * selVect 出最终结果//
    double result[4];
    int _offset =0;
    for(int i = 0; i < 4; i++)
    {
        result[i] =0;
        for(int j = 0; j<3; j++)
            result[i] += _data[_offset+j]*selVect[j];
        _offset +=3;
    }
    strLeftInfo += "\r\n结果矩阵\r\n";
for(int i = 0; i < 4; i++)
{
    strLeftInfo += QString("%1,").arg(result[i]);
}
    //   QMessageBox::information(NULL,"",matrix.strPrintInfo);
//    QMessageBox::information(NULL,"",strInfo);
    //   QMessageBox::information(NULL,"",QString::fromStdString(strResult));
}
void Widget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    if(   strBtnInfo == "pca")
    {
        painter.drawText(ui->widgetLeft->geometry(),Qt::TextWordWrap, strLeftInfo);
    }else
    {
        if(NULL!= m_rawpic)
            painter.drawPixmap(ui->widgetLeft->geometry(), *m_rawpic);//绘制图片 横坐标、纵坐标、宽度、高度
        if(NULL!=m_resultpic)
            painter.drawPixmap(ui->widgetRight->geometry(), *m_resultpic);//绘制图片 横坐标、纵坐标、宽度、高度

    }
}
void Widget::btnHist()
{
    strBtnInfo = "hist";
    Hist();
    update();
}
void Widget::btnConvolution()
{
    strBtnInfo = "convolution";
    char nuclear[9] = {1,1,1,0,0,0,-1,-1,-1};
    convolution(nuclear,3,1);
    update();
}
void Widget::btnPca()
{
    strBtnInfo = "pca";
    pca(3);
     update();
}
