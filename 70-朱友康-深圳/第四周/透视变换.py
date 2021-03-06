import cv2
import numpy as np

def src_code():
    img = cv2.imread('../images/photo1.jpg')

    result3 = img.copy()

    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    print(img.shape)
    # 生成透视变换矩阵；进行透视变换
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    result = cv2.warpPerspective(result3, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

def my_calibration_demo():
    # 定义基本参数
    pixsize = 0.1  # 假设像素精度是0.1mm
    imgw = 640
    imgh = 512
    Zc = 150  # 物距
    f = 15  # 像距
    row = 7  # 棋盘点行数
    col = 9  # 棋盘点列数
    space = 5  # 棋盘点间隔
    u0 = imgw / 2  # 光轴中心
    v0 = imgh / 2
    k1 = -2e-2  # 径向畸变系数(这里不考虑切向畸变)
    k2 = -5e-4
    k3 = -1e-6

    chipw = imgw * pixsize * f / Zc  # 传感器尺寸
    chiph = imgh * pixsize * f / Zc
    dx = chipw / imgw  # 每个像素的硬件物理尺寸
    dy = chiph / imgh

    # 构建棋盘世界坐标点(齐次坐标)
    obj_point = np.zeros((4, row * col), np.float32)
    for idx in range(row * col):
        obj_point[:3, idx] = [idx % col * space, idx // col * space, 0]
    obj_point[3, :] = 1
    each_obj_point = obj_point[:3, :].T

    # 小孔成像转换矩阵
    prjmat = np.array([[f, 0, 0, 0],
                       [0, f, 0, 0],
                       [0, 0, 1, 0]], np.float32)
    # 相机内参矩阵
    kmat = np.array([[1 / dx, 0, u0],
                     [0, 1 / dy, v0],
                     [0, 0, 1]], np.float32)

    init_rvecs = []  # 旋转矩阵参数集合
    init_tvecs = []
    obj_points = []
    img_points = []
    picnum = 10  # 模拟用于标定的图片数,这里简化每张图世界坐标系仅仅绕Z轴旋转Gama角度
    Testimg = (np.ones((imgh, imgw, 3)) * 255).astype(np.uint8)
    for idx in range(picnum):
        angle = np.pi / 3 * (np.random.rand() - 0.5)  # -30~30度
        if idx == 0:
            angle = 0
        init_rvecs.append([0, 0, angle])
        src_offsetx = -(col - 1) * space / 2
        src_offsety = -(row - 1) * space / 2
        tvec = [src_offsetx * np.cos(angle) - src_offsety * np.sin(angle),  # 平移参数Tx,Ty,Tz,这里使点中心在光轴上
                src_offsetx * np.sin(angle) + src_offsety * np.cos(angle),
                Zc]
        init_tvecs.append(tvec)
        # 平移旋转齐次矩阵
        rtmat = np.eye(4, 4, dtype=np.float32)
        rtmat[0, 0] = rtmat[1, 1] = np.cos(angle)
        rtmat[0, 1] = -np.sin(angle)
        rtmat[1, 0] = np.sin(angle)
        rtmat[:3, 3] = tvec

        # 构建像素坐标点
        camera_point = np.dot(rtmat, obj_point)  # 相机坐标系
        c_point = np.dot(prjmat, camera_point) / Zc  # 图像物理坐标系

        # 由理想点生成畸变点
        r2 = c_point[:1, :] ** 2 + c_point[1:2, :] ** 2
        r2 = np.repeat(r2, 2, axis=0)
        scale = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        c_point[:2, :] = c_point[:2, :] * scale

        img_point = np.dot(kmat, c_point)  # 图像像素坐标系

        # 显示测试
        corners = img_point[:2, :].T
        # 加入一些噪声,噪声范围-0.05-0.05
        noise = (np.random.rand(corners.shape[0], corners.shape[1]) - 0.5) * 0.1
        corners += noise
        corners = np.expand_dims(corners, axis=1)
        showimg = (np.ones((imgh, imgw, 3)) * 255).astype(np.uint8)
        cv2.drawChessboardCorners(showimg, (col, row), corners, True)
        if idx == 0:
            Testimg = showimg.copy()
        # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img', showimg)
        cv2.waitKey(500)
        obj_points.append(each_obj_point)
        img_points.append(corners)

    # 对虚拟生成的数据进行相机标定
    use_init_k = True #不使用貌似不行

    if use_init_k:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (imgw, imgh), kmat, None,
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST)
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (imgw, imgh), None, None)

    if (ret):
        print('K error: f/dx:', mtx[0, 0] - f * kmat[0, 0] / Zc, ',f/dy:', mtx[1, 1] - f * kmat[1, 1] / Zc) #opencv返回的相机内参对应那个参数?
        print('D error: k1:', dist[0, 0] - k1,
              ',k2:', dist[0, 1] - k2,
              ',k3:', dist[0, 4] - k3
              )
        # undist = cv2.undistort(showimg, mtx, dist, None, mtx)
        map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (imgw, imgh), cv2.CV_32FC2)
        undist = cv2.remap(Testimg, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = np.concatenate((Testimg, undist), axis=1)
        cv2.imshow("cv remap", result)
        cv2.waitKey(0)

        #下面使用
        wsrc = np.float32([[0 , 0  , 0],
                           [space * (col - 1) , 0,0],
                           [space * (col - 1) , space * (row - 1)  , 0],
                           [0 , space * (row - 1) , 0]])
        #求四个点初始像素位置
        src = cv2.projectPoints(wsrc,rvecs[0],tvecs[0],mtx,dist)[0] #第二个返回值是什么？
        dst = cv2.undistortPoints(src, mtx, dist, P=mtx)  # 畸变校正后像素位置，求畸变校正后像素位置
        src = src.reshape((-1, 2))
        dst = dst.reshape((-1, 2))
        #测试用
        for i in range(4):
            cv2.rectangle(Testimg, (int(src[i][0] - 5), int(src[i][1] - 5)),
                          (int(src[i][0] + 5), int(src[i][1] + 5)), color=[255, 0, 0], thickness=3)
            cv2.rectangle(undist,(int(dst[i][0] - 5),int(dst[i][1] - 5)),
                          (int(dst[i][0] + 5),int(dst[i][1] + 5)),color=[255,0,0],thickness=3)
        cv2.imshow("warpPerspective src -> dst", np.concatenate((Testimg, undist), axis=1))
        cv2.waitKey(0)

        # 生成透视变换矩阵,进行透视变换
        m = cv2.getPerspectiveTransform(src, dst)
        undist = cv2.warpPerspective(Testimg, m, (imgw, imgh))
        result = np.concatenate((Testimg, undist), axis=1) #透视变换矩阵不能用于畸变校正吧？
        cv2.imshow("warpPerspective", result)
        cv2.waitKey(0)


if __name__ == '__main__':
    src_code()
    #根据相机模型生成虚拟畸变点做相机标定，复习相机模型
    #然后用cv remap 和透视变换分别实现畸变校正
    my_calibration_demo()