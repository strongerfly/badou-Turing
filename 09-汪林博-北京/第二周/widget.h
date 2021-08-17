#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QGraphicsLayoutItem>
#include <QGraphicsItem>
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
public                                                                                                                                                                                                                                                                                                                                                  :

protected:
    void paintEvent(QPaintEvent *event) override;
private:
    QPixmap *m_rawpic,*m_resultpic;
    QPixmap m_picLenna,m_picOut;
    void turnGrid();
    void lineChange(int outWidth,int outHeight);
    void nearChange(int outWidth,int outHeight);
    public
    slots:
    void btnTurnGrad();
    void btnLineChange();
    void btnNearChange();
};

#endif // WIDGET_H
