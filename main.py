from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import functools
import operator
import matplotlib.pyplot as mplot
mplot.rcParams["axes.grid"] = False
import math
import pprint
import os
import shutil
import time

from scipy.linalg import toeplitz
from sporco.dictlrn import cbpdndl
from sporco.admm import cbpdn
from sporco import util
from sporco import plot
from sporco import cnvrep
import sporco.linalg as sl
import sporco.metric as sm
from sporco.admm import ccmod
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
plot.config_notebook_plotting()

def l2norm(A):
    l2norm = np.sum( abs(A)*abs(A) )
    return l2norm

def l0norm(A, threshold):
    return np.where(abs(A) < threshold, 0, 1).sum()

def strict_l0norm(A):
    return np.where(A == 0, 0, 1).sum()

def smoothedl0norm(A, sigma):
    N = functools.reduce(operator.mul, A.shape)
    # exp = np.sum( np.exp(-(A*A)/(2*sigma*sigma)) )
    # print(exp)
    # l0_norm = N - exp
    EPS = 0.0000001
    A_ = A.flatten()
    l0_norm = 0
    for a in A_:
        if a > EPS:
            l0_norm += 1
    return l0_norm

def getimages():
    exim = util.ExampleImages(scaled=True, zoom=0.5, gray=True)
    S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
    S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
    S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
    S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
    S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
    return np.dstack((S1, S2, S3, S4, S5))

def saveimg(img, filename, title=None):
    fig = plot.figure(figsize=(7, 7))
    plot.imview(img, fig=fig)
    fig.savefig(filename)
    plot.close()
    mplot.close()

# imgs.shape == (R, C, imgR, imgC) or (C, imgR, imgC)
def saveimg2D(imgs, filename, titles=None):
    if imgs.ndim == 3:
        imgs = np.array([imgs])
    if titles is not None and titles.ndim == 3:
        titles = np.array([titles])
    R = imgs.shape[0]
    C = imgs.shape[1]
    fig = plot.figure(figsize=(7*C, 7*R))
    for r in range(R):
        for c in range(C):
            ax = fig.add_subplot(R, C, r*C + c + 1)
            s = None
            if titles is not None:
                s = titles[r][c]
            plot.imview(imgs[r][c], title=s, fig=fig, ax=ax)
    plot.savefig(filename)
    plot.close()
    mplot.close()

# be careful of non-robust implementation
def format_sig(signal):
    return np.transpose(signal, (3, 0, 1, 2, 4)).squeeze()

def saveXimg(cri, Xr, filename):
    # print(Xr.shape)
    X = np.sum(abs(Xr), axis=cri.axisM).squeeze()
    fig = plot.figure(figsize=(7, 7))
    plot.imview(X, cmap=plot.cm.Blues, fig=fig)
    fig.savefig(filename)
    plot.close()
    mplot.close()

def saveXhist(Xr, filename):
    Xr_ = abs(Xr.flatten())
    fig = plot.figure(figsize=(7*10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(Xr_, bins=500, density=True)
    fig.savefig(filename)
    plot.close()
    mplot.close()

def save_result(D0, D, X, S, S_reconstructed, filename):
    titles = [[], []]
    r1 = []
    for k in range(S.shape[-1]):
        r1.append(S.T[k].T)
        titles[0].append('')
    r1.append(util.tiledict(D0))
    titles[0].append('')
    r2 = []
    for k in range(S.shape[-1]):
        r2.append(S_reconstructed.T[k].T)
        psnr = sm.psnr(S.T[k].T, S_reconstructed.T[k].T)
        ssim = compare_ssim(S.T[k].T, S_reconstructed.T[k].T)
        l0 = strict_l0norm(np.rollaxis(X, 2)[k])
        titles[1].append("PSNR: %.3fdb\nSSIM: %.4f\nl0norm: %d" % (psnr, ssim, l0))
    r2.append(util.tiledict(D))
    titles[1].append('')
    saveimg2D(np.array([r1, r2]), filename, np.array(titles))

def compressedXk(Xrk, size_rate):
    Xrk = Xrk.copy()
    X_flat = np.ravel(Xrk)
    n = math.ceil(X_flat.size*(1 - size_rate))
    print(str(X_flat.size) + " -> " + str(X_flat.size - n))
    for i in np.argsort(abs(X_flat))[0:n]:
        X_flat[i] = 0
    return Xrk

def to_inative(X, sigma):
    return np.where(X < sigma, 0, X)

# a specific axis to 1-length
# copied
def compress_axis(A, axis, i):
    idx = [slice(None)]*A.ndim
    idx[axis] = slice(i, i + 1)
    return A[tuple(idx)]

def compress_axis_op(A, axis, i):
    idx = [slice(None)]*A.ndim
    idx[axis] = slice(i, i + 1)
    return tuple(idx)

def reconstruct(cri, Dr, Xr):
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    return sl.irfftn(sl.inner(Df, Xf, axis=cri.axisM), s=cri.Nv, axes=cri.axisN)

def save_reconstructed(cri, Dr, Xr, Sr, filename, Sr_add=None):
    Sr_ = reconstruct(cri, Dr, Xr)
    if Sr_add is None:
        Sr_add = np.zeros_like(Sr)
    img = np.stack((format_sig(Sr + Sr_add), format_sig(Sr_ + Sr_add)), axis=1)
    saveimg2D(img, filename)

def compressedX(cri, Xr, Sr, size_rate):
    Xr_cmp = Xr.copy()
    for k in range(cri.K):
        s = compress_axis_op(Xr_cmp, cri.axisK, k)
        Xr_cmp[s] = compressedXk(Xr_cmp[s], (Sr.size / Xr.size)*size_rate)
    return Xr_cmp

def calcXr(cri, Dr, Sr, lmbda=5e-2):
    opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                                  'RelStopTol': 5e-3, 'AuxVarObj': False})
    b = cbpdn.ConvBPDN(Dr.squeeze(), Sr.squeeze(), lmbda, opt, dimK=cri.dimK, dimN=cri.dimN)
    Xr = b.solve()
    return Xr

def evaluate_result(cri, Dr0, Dr, Sr, Sr_add=None, lmbda=5e-2, title='result.png'):
    Xr_ = calcXr(cri, Dr, Sr, lmbda)
    print("strict l0 norm", strict_l0norm(Xr_))
    print("l2norm: ", l2norm(Xr_))
    for k in range(cri.K):
        print("image %d: strict l0 norm %f" % (k, strict_l0norm(compress_axis(Xr_, cri.axisK, k))))
    if Sr_add is None:
        Sr_add = np.zeros_like(Sr)
    save_result(Dr0.squeeze(), Dr.squeeze(), Xr_.squeeze(), (Sr + Sr_add).squeeze(), (reconstruct(cri, Dr, Xr_) + Sr_add).squeeze(), title)

def l2norm_minimize(cri, Dr, Sr):
    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding
    Xf = np.conj(Df) / sl.inner(Df, np.conj(Df), axis=cri.axisM) * Sf
    Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)

    Sr_ = sl.irfftn(sl.inner(Df, Xf, axis=cri.axisM), s=cri.Nv, axes=cri.axisN)
    # print(l2norm(np.random.randn(*Xr.shape)))
    # print(l2norm(Xr))
    # print(l2norm(Sr - Sr_))
    po = np.stack((format_sig(Sr), format_sig(Sr_)), axis=1)
    saveimg2D(po, 'l2norm_minimization_test.png') # the right side is Sr_
    return Xr

def convert_to_Df(D):
    Dr = np.asarray(D.reshape(cri.shpD), dtype=S.dtype)
    Df = sl.rfftn(Dr, cri.Nv, cri.axisN)
    return Df

def convert_to_Sf(S):
    Sr = np.asarray(S.reshape(cri.shpS), dtype=S.dtype)
    Sf = sl.rfftn(Sr, None, cri.axisN)
    return Sf

def convert_to_S(Sf):
    S = sl.irfftn(Sf, cri.Nv, cri.axisN).squeeze()
    return S

def convert_to_Xf(X):
    Xr = np.asarray(X.reshape(cri.shpX), dtype=S.dtype)
    Xf = sl.rfftn(Xr, cri.Nv, cri.axisN)
    return Xf

def convert_to_X(Xf):
    X = sl.irfftn(Xf, cri.Nv, cri.axisN).squeeze()
    return X


def derivD_spdomain(cri, Xr, Sr, Df, Xf, dict_Nv):
    B = sl.irfftn(sl.inner(Df, Xf, axis=cri.axisM), s=cri.Nv, axes=cri.axisN) - Sr
    B = B[np.newaxis, np.newaxis,]
    Xshifted = np.ones(dict_Nv + Xr.shape) * Xr
    
    N1 = 0
    N2 = 1
    I = 2
    J = 3

    print("start shifting")
    for n1 in range(dict_Nv[0]):
        for n2 in range(dict_Nv[1]):
            Xshifted[n1][n2] = np.roll(Xshifted[n1][n2], (n1, n2), axis=(I, J))
            # print("shifted ", (n1, n2))
    ret = np.sum(np.conj(B) * Xshifted, axis=(I, J, 2 + cri.axisK), keepdims=True)
    print(ret.shape)
    ret = ret[:, :, 0, 0]
    print(ret.shape)
    return ret

def goldenRatioSearch(function, rng, cnt):
    # 黄金探索法によるステップ幅の最適化
    gamma = (-1+np.sqrt(5))/2
    a = rng[0]
    b = rng[1]
    p = b-gamma*(b-a)
    q = a+gamma*(b-a)
    Fp = function(p)
    Fq = function(q)
    width = 1e8
    for i in range(cnt):
        if Fp <= Fq:
            b = q
            q = p
            Fq = Fp
            p = b-gamma*(b-a)
            Fp = function(p)
        else:
            a = p
            p = q
            Fp = Fq
            q = a+gamma*(b-a)
            Fq = function(q)
            width = abs(b-a)/2
    alpha = (a+b)/2
    return alpha

# 下に凸
def ternary_search(f, rng, cnt):
    left = rng[0]
    right = rng[1]
    for i in range(cnt):
        if f((left * 2 + right) / 3) > f((left + right * 2) / 3):
            left = (left * 2 + right) / 3
        else:
            right = (left + right * 2) / 3
    return (left + right) / 2

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    return (x - min) / (max - min)

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

def to_frequency(cri, Ar):
    return sl.rfftn(Ar, s=cri.Nv, axes=cri.axisN)

def to_spatial(cri, Af):
    return sl.irfftn(Af, s=cri.Nv, axes=cri.axisN)

def update_dict(cri, Pcn, crop_op, Xr, Gr, Hr, Sf, param_rho):
    # D step
    Xf = to_frequency(cri, Xr)
    Gf = to_frequency(cri, Gr)
    Hf = to_frequency(cri, Hr)
    XSf = sl.inner(np.conj(Xf), Sf, cri.axisK)
    b = XSf + param_rho * (Gf - Hf)
    Df = sl.solvemdbi_ism(Xf, param_rho, b, cri.axisM, cri.axisK)
    Dr = to_spatial(cri, Df)
    # G step
    Gr = Pcn(Dr + Hr)
    # H step
    Hr = Hr + Dr - Gr
    return Gr[crop_op], Hr

def nakashizuka_solve(
    cri, Dr0, Xr, Sr,
    final_sigma,
    maxitr = 40,
    param_mu = 1,
    param_lambda = 1e-2,
    debug_dir = None
):
    
    param_rho = 0.5

    Xr = Xr.copy()
    Sr = Sr.copy()
    Dr = Dr0.copy()
    Hr = np.zeros_like(cnvrep.zpad(Dr, cri.Nv))

    Sf = to_frequency(cri, Sr)

    # sigma set
    # sigma_list = []
    # sigma_list.append(Xr.max()*4)
    # for i in range(7):
    #     sigma_list.append(sigma_list[i]*0.5)
    first_sigma = Xr.max()*4
    c = (final_sigma / first_sigma) ** (1/(maxitr - 1))
    print("c = %.8f" % c)
    sigma_list = []
    sigma_list.append(first_sigma)
    for i in range(maxitr - 1):
        sigma_list.append(sigma_list[i]*c)
    
    crop_op = []
    for l in Dr.shape:
        crop_op.append(slice(0, l))
    crop_op = tuple(crop_op)
    Pcn = cnvrep.getPcn(Dr.shape, cri.Nv, cri.dimN, cri.dimCd, zm=False)

    updcnt = 0
    dictcnt = 0
    for sigma in sigma_list:
        print("sigma = %.8f" % sigma)
        # Xf_old = sl.rfftn(Xr, cri.Nv, cri.axisN)
        for l in range(1):
            # print("l0norm: %f" % l0norm(Xr, sigma_list[-1]))
            # print('error1: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))
            # print("l2(Xr): %.6f, l2(delta): %.6f" % (l2norm(Xr), l2norm(delta)))
            delta = Xr * np.exp(-(Xr*Xr) / (2*sigma*sigma))
            Xr = Xr - param_mu*delta# + np.random.randn(*Xr.shape)*sigma*1e-1
            Xf = to_frequency(cri, Xr)

            # print('error2: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))

            Df = to_frequency(cri, Dr)
            b = Xf / param_lambda + np.conj(Df) * Sf
            Xf = sl.solvedbi_sm(Df, 1/param_lambda, b, axis=cri.axisM)
            Xr = to_spatial(cri, Xf).astype(np.float32)
            
            # print('error3: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))

            # save_reconstructed(cri, Dr, Xr, Sr, "./rec/%da.png" % reccnt)
            # saveXhist(Xr, "./hist/%da.png" % reccnt)

            Dr, Hr = update_dict(cri, Pcn, crop_op, Xr, Dr, Hr, Sf, param_rho)
            Df = to_frequency(cri, Dr)
            
            # print('error4: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))

            # # project X to solution space
            # b = sl.inner(Df, Xf, axis=cri.axisM) - Sf
            # c = sl.inner(Df, np.conj(Df), axis=cri.axisM)
            # Xf = Xf - np.conj(Df) / c * b
            # Xr = sl.irfftn(Xf, s=cri.Nv, axes=cri.axisN)

            # print('error5: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))
            
            if debug_dir is not None:
                saveimg(util.tiledict(Dr.squeeze()), debug_dir + "/dict/%d.png" % updcnt)

            updcnt += 1

        # saveXhist(Xr, "Xhist_sigma=" + str(sigma) + ".png")
    
    # print("l0 norm of final X: %d" % smoothedl0norm(Xr, 0.00001))
    plot.close()
    mplot.close()
    return Dr

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

    Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)
    Sf = sl.rfftn(Sr, s=cri.Nv, axes=cri.axisN)
    Xf = sl.rfftn(Xr, s=cri.Nv, axes=cri.axisN)
    alpha = 1e0

    # sigma set
    first_sigma = Xr.max()*4
    c = (final_sigma / first_sigma) ** (1/(maxitr - 1))
    print("c = %.8f" % c)
    sigma_list = []
    sigma_list.append(first_sigma)
    for i in range(maxitr - 1):
        sigma_list.append(sigma_list[i]*c)
        print(sigma_list[-1])
    
    crop_op = []
    for l in Dr.shape:
        crop_op.append(slice(0, l))
    crop_op = tuple(crop_op)
    Pcn = cnvrep.getPcn(Dr.shape, cri.Nv, cri.dimN, cri.dimCd, zm=False)

    updcnt = 0
    for sigma in sigma_list:
        print("sigma = %.8f" % sigma)
        # Xf_old = sl.rfftn(Xr, cri.Nv, cri.axisN)
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

        B = sl.inner(Xf, Df, axis=cri.axisM) - Sf
        derivDf = sl.inner(np.conj(Xf), B, axis=cri.axisK)
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
        Dr = Pcn(sl.irfftn(Df, s=cri.Nv, axes=cri.axisN))[crop_op]
        # print(l2norm(Dr.T[0]))
        # Dr = normalize(Dr, axis=cri.axisN)
        print(l2norm(Dr.T[0]))
        Df = sl.rfftn(Dr, s=cri.Nv, axes=cri.axisN)

        if debug_dir is not None:
            saveimg(util.tiledict(Dr.squeeze()), debug_dir + "/dict/%d.png" % updcnt)
        # if debug_dir is not None:
        #     save_reconstructed(cri, Dr, Xr, Sr, debug_dir + '/%drecC.png' % updcnt)
        # dictcnt += 1

        # print('error4: ', l2norm(Sr - reconstruct(cri, Dr, Xr)))

        # save_reconstructed(cri, Dr, Xr, Sr, debug_dir + "/rec/%dc.png" % updcnt)

        # project X to solution space
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

def sporcosolve(cri, Dr0, Sr, maxitr=200):
    Dr0 = Dr0.copy()
    Sr = Sr.copy()
    lmbda = 0.2
    opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': maxitr,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns')
    d = cbpdndl.ConvBPDNDictLearn(Dr0.squeeze(), Sr.squeeze(), lmbda, opt, dmethod='cns')
    Dr = d.solve()
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
    return Dr

def testdict(cri, Dr0, Dr, Slr, Shr, dir,
    lambdas = [
        '1e-2',
        '2e-2',
        '5e-2',
        '1e-1',
        '2e-1',
        '5e-1',
    ]):
    S = (Slr + Shr).squeeze()
    ret = [[] for k in range(cri.K)]
    for s in lambdas:
        print('==========================================')
        print('test dictionary (lambda = %s)' % s)
        print('==========================================')
        lmbda = float(s)
        Xr = calcXr(cri, Dr, Shr, lmbda=lmbda)
        X = Xr.squeeze()
        S_ = (reconstruct(cri, Dr, Xr) + Slr).squeeze()
        for k in range(cri.K):
            d = {
                'lambda': lmbda,
                'psnr': sm.psnr(S.T[k].T, S_.T[k].T),
                'ssim': compare_ssim(S.T[k].T, S_.T[k].T),
                'l0norm': strict_l0norm(np.rollaxis(X, 2)[k]),
            }
            pprint.pprint(d)
            ret[k].append(d)

        save_result(Dr0.squeeze(), Dr.squeeze(), X, (Slr + Shr).squeeze(), S_, dir + '/result_lambda=%s.png' % s)
    return ret

def test_mysolve(cri_train, Dr0, Shr_train, cri_test, Slr_test, Shr_test, outdir='.'):
    itrs = [5, 10, 20, 30, 40, 50]
    # itrs = [50]
    data = [[] for k in range(cri_test.K)]
    times = []

    # dummy (for memory allocate on google colab)
    Xr = l2norm_minimize(cri_train, Dr0, Shr_train)
    Dr = mysolve(cri_train, Dr0, Xr, Shr_train, 1e-4, maxitr=2)

    for maxitr in itrs:
        start = time.time()
        Xr = l2norm_minimize(cri_train, Dr0, Shr_train)
        Dr = mysolve(cri_train, Dr0, Xr, Shr_train, 1e-4, maxitr=maxitr)
        end = time.time()
        times.append({'maxitr': maxitr, 'time': end - start})

        dir = outdir + '/mysolve_itr=%d' % maxitr
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        res = testdict(cri_test, Dr0, Dr, Slr_test, Shr_test, dir,
            lambdas = [
                '1e-3',
                '3e-3',
                '1e-2',
                '3e-2',
                '1e-1',
                '3e-1',])
        for k in range(cri_test.K):
            for d in res[k]:
                d['maxitr'] = maxitr
            data[k] += res[k]
    return data, times

def test_nakashizuka_solve(cri_train, Dr0, Shr_train, cri_test, Slr_test, Shr_test, outdir='.'):
    itrs = [20, 40, 60, 100, 200, 300]
    data = [[] for k in range(cri_test.K)]
    times = []
    
    # dummy (for memory allocate on google colab)
    Xr = l2norm_minimize(cri_train, Dr0, Shr_train)
    Dr = nakashizuka_solve(cri_train, Dr0, Xr, Shr_train, 1e-4, maxitr=2)

    for maxitr in itrs:
        start = time.time()
        Xr = l2norm_minimize(cri_train, Dr0, Shr_train)
        Dr = nakashizuka_solve(cri_train, Dr0, Xr, Shr_train, 1e-4, maxitr=maxitr)
        end = time.time()
        times.append({'maxitr': maxitr, 'time': end - start})

        dir = outdir + '/nakashizuka_solve_itr=%d' % maxitr
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        res = testdict(cri_test, Dr0, Dr, Slr_test, Shr_test, dir)
        for k in range(cri_test.K):
            for d in res[k]:
                d['maxitr'] = maxitr
            data[k] += res[k]
    return data, times

def test_sporcosolve(cri_train, Dr0, Shr_train, cri_test, Slr_test, Shr_test, outdir='.'):
    itrs = [20, 40, 80, 120, 160, 200]
    # itrs = [200]
    data = [[] for k in range(cri_test.K)]
    times = []

    # dummy (for memory allocate on google colab)
    Dr = sporcosolve(cri_train, Dr0, Shr_train, maxitr=2)

    for maxitr in itrs:
        start = time.time()
        Dr = sporcosolve(cri_train, Dr0, Shr_train, maxitr=maxitr)
        end = time.time()
        times.append({'maxitr': maxitr, 'time': end - start})

        dir = outdir + '/sporcosolve_itr=%d' % maxitr
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        res = testdict(cri_test, Dr0, Dr, Slr_test, Shr_test, dir)
        for k in range(cri_test.K):
            for d in res[k]:
                d['maxitr'] = maxitr
            data[k] += res[k]
    return data, times

np.random.seed(12345)

S = getimages().astype(np.float32)
# saveimg2D(np.transpose(S, (2, 0, 1)), 'S.png')
# exit()
print(S.shape)
print("%d images, each is %dx%d." % (S.shape[2], S.shape[0], S.shape[1]))
# Sl, Sh = util.tikhonov_filter(S, 5, 16)
Sl = np.zeros_like(S)
Smean = np.mean(S*2, axis=(0, 1))
print(Smean)
Sh = S*2 - Smean

# TODO: explicitly zero-padding (for me, foolish)
D = np.random.randn(12, 12, 256)
# D = np.random.rand(12, 12, 64)*2 - 1
cri = cnvrep.CSC_ConvRepIndexing(D, S)
Dr0 = np.asarray(D.reshape(cri.shpD), dtype=S.dtype)
Slr = np.asarray(Sl.reshape(cri.shpS), dtype=S.dtype)
Shr = np.asarray(Sh.reshape(cri.shpS), dtype=S.dtype)
Shf = sl.rfftn(Shr, s=cri.Nv, axes=cri.axisN) # implicitly zero-padding

crop_op = []
for l in Dr0.shape:
    crop_op.append(slice(0, l))
crop_op = tuple(crop_op)
Dr0 = cnvrep.getPcn(Dr0.shape, cri.Nv, cri.dimN, cri.dimCd, zm=False)(cnvrep.zpad(Dr0, cri.Nv))[crop_op]
# Dr = normalize(Dr, axis=cri.axisM)

# Xr = l2norm_minimize(cri, Dr0, Shr)
# Dr = mysolve(cri, Dr0, Xr, Shr, 1e-4, maxitr=50, debug_dir='./debug')
# # Dr = nakashizuka_solve(cri, Dr0, Xr, Shr, debug_dir='./debug')
# # Dr = sporcosolve(cri, Dr, Shr)
# # fig = plot.figure(figsize=(7, 7))
# # plot.imview(util.tiledict(Dr.squeeze()), fig=fig)
# # fig.savefig('dict.png')
# # # evaluate_result(cri, Dr0, Dr, Shr, Sr_add=Slr)


exim1 = util.ExampleImages(scaled=True, zoom=0.5, pth='./')
S1_test = exim1.image('couple.tiff')
exim2 = util.ExampleImages(scaled=True, zoom=1, pth='./')
S2_test = exim2.image('LENNA.bmp')
S_test = np.dstack((S1_test, S2_test))
cri_test = cnvrep.CSC_ConvRepIndexing(D, S_test)
Sl_test, Sh_test = util.tikhonov_filter(S_test, 5, 16)
Slr_test = np.asarray(Sl_test.reshape(cri_test.shpS), dtype=S_test.dtype)
Shr_test = np.asarray(Sh_test.reshape(cri_test.shpS), dtype=S_test.dtype)

# evaluate_result(cri, Dr0, Dr, Shr_test, Sr_add=Slr_test, lmbda=5e-3)

outdir = './no_low-pass'


#--------実験を行う手法を指定-------
prefix = 'nakashizuka_solve' # 中静先生の論文
# prefix = 'mysolve' # 提案手法
# prefix = 'sporcosolve' # sporcoライブラリに実装された方法（B. Wohlbergによる）
#----------------------------------

if prefix == 'nakashizuka_solve':
    data, times = test_nakashizuka_solve(cri, Dr0, Shr, cri_test, Slr_test, Shr_test, outdir=outdir)
if prefix == 'mysolve':
    data, times = test_mysolve(cri, Dr0, Shr, cri_test, Slr_test, Shr_test, outdir=outdir)
if prefix == 'sporcosolve':
    data, times = test_sporcosolve(cri, Dr0, Shr, cri_test, Slr_test, Shr_test, outdir=outdir)

# 実験結果をgnuplotに都合の良い形式で出力
for k in range(cri_test.K):
    data_path = outdir + '/' + prefix + '_data%d.txt' % k
    with open(data_path, mode='w') as f:
        f.write('# %20s  %20s  %20s  %20s  %20s\n' % ('iterations', 'lambda', 'L0 Norm', 'SSIM', 'PSNR'))
        for d in data[k]:
            t = (d['maxitr'], d['lambda'], d['l0norm'], d['ssim'], d['psnr'])
            f.write('  %20f  %20f  %20f  %20f  %20f\n' % t)
with open(outdir + '/' + prefix + '_times.txt', mode='w') as f:
    f.write('# %10s  %20s\n' % ('iterations', 'time'))
    for d in times:
        t = (d['maxitr'], d['time'])
        f.write('  %10d  %20f\n' % t)
