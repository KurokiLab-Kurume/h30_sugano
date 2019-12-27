import numpy as np
import sporco.linalg as sl
from sporco import util
from sporco import cnvrep


#係数と辞書の両方の最適化に射影勾配法を用いる
def mysolve(
    cri, Dr0, Xr, Sr,
    final_sigma,
    maxitr = 40,
    param_mu = 1,
    debug_dir = None
):
    Dr = Dr0.copy()
    Xr = Xr.copy()
    Sr = Sr.copy()

    #離散フーリエ変換
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN)
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    alpha = 1e0

    # sigma set
    first_sigma = Xr.max()*4
    # σ←cσ(c < 1)のｃの決定
    c = (final_sigma / first_sigma) ** (1/(maxitr - 1))
    print("c = %.8f" % c)
    sigma_list = []
    sigma_list.append(first_sigma)
    for i in range(maxitr - 1):
        sigma_list.append(sigma_list[i]*c)
        print(sigma_list[-1])
    
    # 辞書のクロップする領域を添え字で指定
    crop_op = []
    for l in Dr.shape:
        crop_op.append(slice(0, l))
    crop_op = tuple(crop_op)
    print(crop_op)
    
    updcnt = 0
    for sigma in sigma_list:
        print("sigma = %.8f" % sigma)
        # print("l0norm: %f" % l0norm(Xr, sigma_list[-1]))
        # print('error1: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))
        delta = Xr * np.exp(-(Xr*Xr) / (2*sigma*sigma))
        # print("l2(Xr): %.6f, l2(delta): %.6f" % (l2norm(Xr), l2norm(delta)))
        Xr = Xr - param_mu*delta# + np.random.randn(*Xr.shape)*sigma*1e-1
        Xf = sl.rfftn(Xr, cri.Nv, cri.axisN)
        # saveXhist(Xr, "./hist/%db.png" % reccnt)

        # print('error2: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))

        # if debug_dir is not None:
        #     save_reconstructed(cri, Dr, Xr, Sr, debug_dir + '/%drecA.png' % updcnt)

        # DXf = sl.inner(Df, Xf, axis=cri.axisM)
        # gamma = (np.sum(np.conj(DXf) * Sf, axis=cri.axisN, keepdims=True) + np.sum(DXf * np.conj(Sf), axis=cri.axisN, keepdims=True)) / 2 / np.sum(np.conj(DXf) * DXf, axis=cri.axisN, keepdims=True)
        # print(gamma)
        # print(gamma.shape, ' * ', Xr.shape)
        # gamma = np.real(gamma)
        # Xr = Xr * gamma
        # Xf = to_frequency(cri, Xr)

        # if debug_dir is not None:
        #     save_reconstructed(cri, Dr, Xr, Sr, debug_dir + '/%drecB.png' % updcnt)
        
        # print('error3: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))
        # print("max: ", np.max(Xr))

        # 辞書の勾配降下
        B = sl.inner(Xf, Df, axis=cri.axisM) - Sf
        derivDf = sl.inner(np.conj(Xf), B, axis=cri.axisK) # 目的関数(信号との二乗近似誤差)の勾配
        # derivDr = sl.irfftn(derivDf, s=cri.Nv, axes=cri.axisN)[crop_op]
        def func(alpha):
            Df_ = Df - alpha * derivDf
            Dr_ = sl.irfftn(Df_, s=cri.Nv, axes=cri.axisN)[crop_op]
            Df_ = sl.rfftn(Dr_, s=cri.Nv, axes=cri.axisN)
            Sf_ = sl.inner(Df_, Xf, axis=cri.axisM)
            return l2norm(Sr - sl.irfftn(Sf_, s=cri.Nv, axes=cri.axisN))
        choice = np.array([func(alpha / 2), func(alpha), func(alpha * 2)]).argmin()
        alpha *= [0.5, 1, 2][choice]
        print("alpha: ", alpha)
        Df = Df - alpha * derivDf

        # 辞書の射影
        Dr = sl.irfftn(Df, s=cri.Nv, axes=cri.axisN)
        Pcn = cnvrep.getPcn(Dr.shape, cri.Nv, cri.dimN, cri.dimCd, zm=False) # 射影関数のインスタンス化
        Dr = Pcn(Dr)
        Dr = Dr[crop_op]
        print(l2norm(Dr.T[0]))
        
        Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
        b = sl.inner(Df, Xf, axis=cri.axisM) - Sf
        c = sl.inner(Df, np.conj(Df), axis=cri.axisM)
        Xf = Xf - np.conj(Df) / c * b
        Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)
        
        # save_reconstructed(cri, Dr, Xr, Sr, debug_dir + "rec/%dd.png" % updcnt)
        # saveXhist(Xr, debug_dir + "hist/%db.png" % updcnt)
        updcnt += 1
    
    # print("l0 norm of final X: %d" % smoothedl0norm(Xr, 0.00001))
    plot.close()
    mplot.close()
    return Dr