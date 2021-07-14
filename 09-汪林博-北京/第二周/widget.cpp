#include "widget.h"
#include "ui_widget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>

#include <QPainter>
#include <QDir>
#include <QMessageBox>
//////////////////
/// \brief Widget::Widget
/// \param parent
/// Qimage img = pixmap.toimage();
/// Qpixmap pxmap = Qpixmap::fromimage(img);


Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    m_rawpic = m_resultpic = NULL;//":/images/block.png"//

    QString file = QDir::currentPath()+"//lenna.png";
//     m_picLenna.load(file);
       m_picLenna.load("lenna.png");
    if(!m_picLenna.isNull())m_rawpic = &m_picLenna;
    ui->setupUi(this);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    QHBoxLayout *topLayout = new QHBoxLayout;
    QHBoxLayout *bottomLayout = new QHBoxLayout;
    topLayout->addWidget(ui->turngrad);
    topLayout->addWidget(ui->nearChange);
    topLayout->addWidget(ui->lineChange);
    ui->widgetRaw->setGeometry(0,20,300,300);
    ui->widgetResult->setGeometry(300,20,300,300);

    bottomLayout->addWidget(ui->widgetRaw);
    bottomLayout->addWidget(ui->widgetResult);
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(bottomLayout);

    setLayout(mainLayout);
    connect(ui->turngrad, SIGNAL(clicked()), this, SLOT(btnTurnGrad()));
    connect(ui->nearChange, SIGNAL(clicked()), this, SLOT(btnNearChange()));
    connect(ui->lineChange, SIGNAL(clicked()), this, SLOT(btnLineChange()));

}

Widget::~Widget()
{
    delete ui;
}
void Widget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);

    //    QPixmap pix;
    //    painter.translate(0, 0);
    //    pix.load("D:\\wlbcode\\qt\\rgb2grad\\lenna.png");//加载图片地址 可以是资源文件
    if(NULL!= m_rawpic)
        painter.drawPixmap(ui->widgetRaw->geometry(), *m_rawpic);//绘制图片 横坐标、纵坐标、宽度、高度
    if(NULL!=m_resultpic)
        painter.drawPixmap(ui->widgetResult->geometry(), *m_resultpic);//绘制图片 横坐标、纵坐标、宽度、高度

    //    if(NULL!= m_rawpic)
    //    painter.drawPixmap(QRect(0,20,300,300) ,QPixmap(QLatin1String(":/lenna.png")),QRect(0,0,300,300));

    // painter.drawPixmap(ui->widgetRaw->geometry() ,*m_rawpic,m_rawpic->rect());
    //    ui->widgetRaw->setAutoFillBackground(true);
    //    QPalette palette;
    //    palette.setBrush(QPalette::Background,QBrush(m_rawpic));
    //    ui->widgetRaw->setPalette(palette);
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
void Widget::lineChange(int outWidth,int outHeight)
{
    m_resultpic = NULL;
    QImage *newimage = new QImage(outWidth,outHeight,QImage::Format_ARGB32);
    QRgb *line;
    QImage origin = m_picLenna.toImage();
    float xmuti,ymuti;
    float xnear,ynear;
    float xu,x_u,yv,y_v;
    double a11,a12,a21,a22;
    int cacuXnear,cacuYnear;
    int XNear1,YNear1;
    xmuti = (m_picLenna.width()+0.0000001)/outWidth;
    ymuti = (m_picLenna.height()+0.0000001)/outHeight;
    for(int y = 0; y < newimage->height();y++)
    {
        ynear= (y+0.5)*ymuti-0.5;
        if(ynear <0)ynear = 0;
        cacuYnear = (int)ynear;
        YNear1 = cacuYnear+1;
        yv = ynear -cacuYnear;
        y_v = 1-yv;
        if(cacuYnear >= newimage->height())cacuYnear = newimage->height()-1;
        if(YNear1 >= newimage->height())YNear1 = newimage->height()-1;
        QRgb * line = (QRgb *)origin.scanLine(cacuYnear);
        QRgb * lineNext = (QRgb *)origin.scanLine(YNear1);

        for(int x = 0; x<newimage->width(); x++)
        {
            xnear= (x+0.5)*xmuti-0.5;
            if(xnear<0)xnear = 0;
            cacuXnear = (int)xnear;
            XNear1 = cacuXnear+1;
            xu = xnear -cacuXnear;
            x_u = 1-xu;
            if(XNear1 >= newimage->width())XNear1 = newimage->width()-1;
            if(cacuXnear >= newimage->width())cacuXnear = newimage->width()-1;
            a11 = x_u*y_v;a12 =x_u*yv;
            a21 = xu*y_v;a22 = xu*yv;
            int r = a11*qRed(line[cacuXnear])+a12*qRed(line[XNear1])+a21*qRed(lineNext[cacuXnear])+a22*qRed(lineNext[XNear1]);
            int g = a11*qGreen(line[cacuXnear])+a12*qGreen(line[XNear1])+a21*qGreen(lineNext[cacuXnear])+a22*qGreen(lineNext[XNear1]);
            int b = a11*qBlue(line[cacuXnear])+a12*qBlue(line[XNear1])+a21*qBlue(lineNext[cacuXnear])+a22*qBlue(lineNext[XNear1]);

         //   newimage->setPixel(x,y, qRgb(qRed(line[cacuXnear]), qGreen(line[cacuXnear]), qBlue(line[cacuXnear])));
            newimage->setPixel(x,y, qRgb(r,g,b));

        }

    }
    m_picOut = QPixmap::fromImage(*newimage );
    m_resultpic = &m_picOut;
}
void Widget::nearChange(int outWidth,int outHeight)
{
    //= (1-u) * (1-v) * f(i, j) + (1-u) * v * f(i, j+1) + u * (1-v) * f(i+1, j) + u * v * f(i+1, j+1)//
    m_resultpic = NULL;
    QImage *newimage = new QImage(outWidth,outHeight,QImage::Format_ARGB32);
    QRgb *line;
    QImage origin = m_picLenna.toImage();
    float xmuti,ymuti;
    float xnear,ynear;
    int cacuXnear,cacuYnear;
    xmuti = (m_picLenna.width()+0.0000001)/outWidth;
    ymuti = (m_picLenna.height()+0.0000001)/outHeight;
    for(int y = 0; y < newimage->height();y++)
    {
        ynear= (y)*ymuti;
        cacuYnear = (int)ynear;
        if(cacuYnear >= newimage->height())cacuYnear = newimage->height()-1;

        QRgb * line = (QRgb *)origin.scanLine(cacuYnear);

        for(int x = 0; x<newimage->width(); x++)
        {
            xnear= (x)*xmuti;
            cacuXnear = (int)xnear;
            if(cacuXnear >= newimage->width())cacuXnear = newimage->width()-1;

            newimage->setPixel(x,y, qRgb(qRed(line[cacuXnear]), qGreen(line[cacuXnear]), qBlue(line[cacuXnear])));
        }

    }
    m_picOut = QPixmap::fromImage(*newimage );
    m_resultpic = &m_picOut;
}
void  Widget::btnTurnGrad()
{
    turnGrid();
    update();
}

void  Widget::btnLineChange()
{
    lineChange(700,600);
    update();
}
void  Widget::btnNearChange()
{
    nearChange(700,600);
    update();
}
