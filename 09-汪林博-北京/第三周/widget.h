#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QGraphicsLayoutItem>
#include <QGraphicsItem>
#include <string>
namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget();

private:
    Ui::Widget *ui;
protected:
    void paintEvent(QPaintEvent *event) override;
private:
    std::string strBtnInfo = "hist";
    QString strLeftInfo = "";
    QString strRightInfo = "";
    QPixmap *m_rawpic,*m_resultpic;
    QPixmap m_picLenna,m_picOut;
    void Hist();
    void convolution(char *nuclear,int matrixsize,int pad);
    void pca(int dim);
    void turnGrid();
    public
    slots:
    void btnHist();
    void btnConvolution();
    void btnPca();
};

#endif // WIDGET_H
