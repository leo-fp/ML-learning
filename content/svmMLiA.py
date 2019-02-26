# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:57:36 2019

@author: Administrator
"""

# --*-- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag


def loadDataSet(fileName):
    """loadDataSet£¨¶ÔÎÄ¼þ½øÐÐÖðÐÐ½âÎö£¬´Ó¶øµÃµ½µÚÐÐµÄÀà±êÇ©ºÍÕû¸öÊý¾Ý¾ØÕó£©
    Args:
        fileName ÎÄ¼þÃû
    Returns:
        dataMat  Êý¾Ý¾ØÕó
        labelMat Àà±êÇ©
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    Ëæ»úÑ¡ÔñÒ»¸öÕûÊý
    Args:
        i  µÚÒ»¸öalphaµÄÏÂ±ê
        m  ËùÓÐalphaµÄÊýÄ¿
    Returns:
        j  ·µ»ØÒ»¸ö²»ÎªiµÄËæ»úÊý£¬ÔÚ0~mÖ®¼äµÄÕûÊýÖµ
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """clipAlpha(µ÷ÕûajµÄÖµ£¬Ê¹aj´¦ÓÚ L<=aj<=H)
    Args:
        aj  Ä¿±êÖµ
        H   ×î´óÖµ
        L   ×îÐ¡Öµ
    Returns:
        aj  Ä¿±êÖµ
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def calcEk(oS, k):
    """calcEk£¨Çó EkÎó²î£ºÔ¤²âÖµ-ÕæÊµÖµµÄ²î£©
    ¸Ã¹ý³ÌÔÚÍêÕû°æµÄSMOËã·¨ÖÐÅã³öÏÖ´ÎÊý½Ï¶à£¬Òò´Ë½«Æäµ¥¶À×÷ÎªÒ»¸ö·½·¨
    Args:
        oS  optStruct¶ÔÏó
        k   ¾ßÌåµÄÄ³Ò»ÐÐ
    Returns:
        Ek  Ô¤²â½á¹ûÓëÕæÊµ½á¹û±È¶Ô£¬¼ÆËãÎó²îEk
    """
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    """selectJ£¨·µ»Ø×îÓÅµÄjºÍEj£©
    ÄÚÑ­»·µÄÆô·¢Ê½·½·¨¡£
    Ñ¡ÔñµÚ¶þ¸ö(ÄÚÑ­»·)alphaµÄalphaÖµ
    ÕâÀïµÄÄ¿±êÊÇÑ¡ÔñºÏÊÊµÄµÚ¶þ¸öalphaÖµÒÔ±£Ö¤Ã¿´ÎÓÅ»¯ÖÐ²ÉÓÃ×î´ó²½³¤¡£
    ¸Ãº¯ÊýµÄÎó²îÓëµÚÒ»¸öalphaÖµEiºÍÏÂ±êiÓÐ¹Ø¡£
    Args:
        i   ¾ßÌåµÄµÚiÒ»ÐÐ
        oS  optStruct¶ÔÏó
        Ei  Ô¤²â½á¹ûÓëÕæÊµ½á¹û±È¶Ô£¬¼ÆËãÎó²îEi
    Returns:
        j  Ëæ»úÑ¡³öµÄµÚjÒ»ÐÐ
        Ej Ô¤²â½á¹ûÓëÕæÊµ½á¹û±È¶Ô£¬¼ÆËãÎó²îEj
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # Ê×ÏÈ½«ÊäÈëÖµEiÔÚ»º´æÖÐÉèÖÃ³ÉÎªÓÐÐ§µÄ¡£ÕâÀïµÄÓÐÐ§ÒâÎ¶×ÅËüÒÑ¾­¼ÆËãºÃÁË¡£
    oS.eCache[i] = [1, Ei]

    # print 'oS.eCache[%s]=%s' % (i, oS.eCache[i])
    # print 'oS.eCache[:, 0].A=%s' % oS.eCache[:, 0].A.T
    # """
    # # ·µ»Ø·Ç0µÄ£ºÐÐÁÐÖµ
    # nonzero(oS.eCache[:, 0].A)= (
    #     ÐÐ£º array([ 0,  2,  4,  5,  8, 10, 17, 18, 20, 21, 23, 25, 26, 29, 30, 39, 46,52, 54, 55, 62, 69, 70, 76, 79, 82, 94, 97]), 
    #     ÁÐ£º array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0])
    # )
    # """
    # print 'nonzero(oS.eCache[:, 0].A)=', nonzero(oS.eCache[:, 0].A)
    # # È¡ÐÐµÄlist
    # print 'nonzero(oS.eCache[:, 0].A)[0]=', nonzero(oS.eCache[:, 0].A)[0]
    # ·ÇÁãEÖµµÄÐÐµÄlistÁÐ±í£¬Ëù¶ÔÓ¦µÄalphaÖµ
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # ÔÚËùÓÐµÄÖµÉÏ½øÐÐÑ­»·£¬²¢Ñ¡ÔñÆäÖÐÊ¹µÃ¸Ä±ä×î´óµÄÄÇ¸öÖµ
            if k == i:
                continue  # don't calc for i, waste of time

            # Çó EkÎó²î£ºÔ¤²âÖµ-ÕæÊµÖµµÄ²î
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # Èç¹ûÊÇµÚÒ»´ÎÑ­»·£¬ÔòËæ»úÑ¡ÔñÒ»¸öalphaÖµ
        j = selectJrand(i, oS.m)

        # Çó EkÎó²î£ºÔ¤²âÖµ-ÕæÊµÖµµÄ²î
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    """updateEk£¨¼ÆËãÎó²îÖµ²¢´æÈë»º´æÖÐ¡££©
    ÔÚ¶ÔalphaÖµ½øÐÐÓÅ»¯Ö®ºó»áÓÃµ½Õâ¸öÖµ¡£
    Args:
        oS  optStruct¶ÔÏó
        k   Ä³Ò»ÁÐµÄÐÐºÅ
    """

    # Çó Îó²î£ºÔ¤²âÖµ-ÕæÊµÖµµÄ²î
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """innerL
    ÄÚÑ­»·´úÂë
    Args:
        i   ¾ßÌåµÄÄ³Ò»ÐÐ
        oS  optStruct¶ÔÏó
    Returns:
        0   ÕÒ²»µ½×îÓÅµÄÖµ
        1   ÕÒµ½ÁË×îÓÅµÄÖµ£¬²¢ÇÒoS.Cacheµ½»º´æÖÐ
    """

    # Çó EkÎó²î£ºÔ¤²âÖµ-ÕæÊµÖµµÄ²î
    Ei = calcEk(oS, i)

    # Ô¼ÊøÌõ¼þ (KKTÌõ¼þÊÇ½â¾ö×îÓÅ»¯ÎÊÌâµÄÊ±ÓÃµ½µÄÒ»ÖÖ·½·¨¡£ÎÒÃÇÕâÀïÌáµ½µÄ×îÓÅ»¯ÎÊÌâÍ¨³£ÊÇÖ¸¶ÔÓÚ¸ø¶¨µÄÄ³Ò»º¯Êý£¬ÇóÆäÔÚÖ¸¶¨×÷ÓÃÓòÉÏµÄÈ«¾Ö×îÐ¡Öµ)
    # 0<=alphas[i]<=C£¬µ«ÓÉÓÚ0ºÍCÊÇ±ß½çÖµ£¬ÎÒÃÇÎÞ·¨½øÐÐÓÅ»¯£¬ÒòÎªÐèÒªÔö¼ÓÒ»¸öalphasºÍ½µµÍÒ»¸öalphas¡£
    # ±íÊ¾·¢Éú´íÎóµÄ¸ÅÂÊ£ºlabelMat[i]*Ei Èç¹û³¬³öÁË toler£¬ ²ÅÐèÒªÓÅ»¯¡£ÖÁÓÚÕý¸ººÅ£¬ÎÒÃÇ¿¼ÂÇ¾ø¶ÔÖµ¾Í¶ÔÁË¡£
    '''
    # ¼ìÑéÑµÁ·Ñù±¾(xi, yi)ÊÇ·ñÂú×ãKKTÌõ¼þ
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # Ñ¡Ôñ×î´óµÄÎó²î¶ÔÓ¦µÄj½øÐÐÓÅ»¯¡£Ð§¹û¸üÃ÷ÏÔ
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # LºÍHÓÃÓÚ½«alphas[j]µ÷Õûµ½0-CÖ®¼ä¡£Èç¹ûL==H£¬¾Í²»×öÈÎºÎ¸Ä±ä£¬Ö±½Óreturn 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0

        # etaÊÇalphas[j]µÄ×îÓÅÐÞ¸ÄÁ¿£¬Èç¹ûeta==0£¬ÐèÒªÍË³öforÑ­»·µÄµ±Ç°µü´ú¹ý³Ì
        # ²Î¿¼¡¶Í³¼ÆÑ§Ï°·½·¨¡·Àîº½-P125~P128<ÐòÁÐ×îÐ¡×îÓÅ»¯Ëã·¨>
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0

        # ¼ÆËã³öÒ»¸öÐÂµÄalphas[j]Öµ
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # ²¢Ê¹ÓÃ¸¨Öúº¯Êý£¬ÒÔ¼°LºÍH¶ÔÆä½øÐÐµ÷Õû
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # ¸üÐÂÎó²î»º´æ
        updateEk(oS, j)

        # ¼ì²éalpha[j]ÊÇ·ñÖ»ÊÇÇáÎ¢µÄ¸Ä±ä£¬Èç¹ûÊÇµÄ»°£¬¾ÍÍË³öforÑ­»·¡£
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0

        # È»ºóalphas[i]ºÍalphas[j]Í¬Ñù½øÐÐ¸Ä±ä£¬ËäÈ»¸Ä±äµÄ´óÐ¡Ò»Ñù£¬µ«ÊÇ¸Ä±äµÄ·½ÏòÕýºÃÏà·´
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # ¸üÐÂÎó²î»º´æ
        updateEk(oS, i)

        # ÔÚ¶Ôalpha[i], alpha[j] ½øÐÐÓÅ»¯Ö®ºó£¬¸øÕâÁ½¸öalphaÖµÉèÖÃÒ»¸ö³£Êýb¡£
        # w= ¦²[1~n] ai*yi*xi => b = yj ¦²[1~n] ai*yi(xi*xj)
        # ËùÒÔ£º  b1 - b = (y1-y) - ¦²[1~n] yi*(a1-a)*(xi*x1)
        # ÎªÊ²Ã´¼õ2±é£¿ ÒòÎªÊÇ ¼õÈ¥¦²[1~n]£¬ÕýºÃ2¸ö±äÁ¿iºÍj£¬ËùÒÔ¼õ2±é
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    ÍêÕûSMOËã·¨ÍâÑ­»·£¬ÓësmoSimpleÓÐÐ©ÀàËÆ£¬µ«ÕâÀïµÄÑ­»·ÍË³öÌõ¼þ¸ü¶àÒ»Ð©
    Args:
        dataMatIn    Êý¾Ý¼¯
        classLabels  Àà±ð±êÇ©
        C   ËÉ³Ú±äÁ¿(³£Á¿Öµ)£¬ÔÊÐíÓÐÐ©Êý¾Ýµã¿ÉÒÔ´¦ÓÚ·Ö¸ôÃæµÄ´íÎóÒ»²à¡£
            ¿ØÖÆ×î´ó»¯¼ä¸ôºÍ±£Ö¤´ó²¿·ÖµÄº¯Êý¼ä¸ôÐ¡ÓÚ1.0ÕâÁ½¸öÄ¿±êµÄÈ¨ÖØ¡£
            ¿ÉÒÔÍ¨¹ýµ÷½Ú¸Ã²ÎÊý´ïµ½²»Í¬µÄ½á¹û¡£
        toler   ÈÝ´íÂÊ
        maxIter ÍË³öÇ°×î´óµÄÑ­»·´ÎÊý
    Returns:
        b       Ä£ÐÍµÄ³£Á¿Öµ
        alphas  À­¸ñÀÊÈÕ³Ë×Ó
    """

    # ´´½¨Ò»¸ö optStruct ¶ÔÏó
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # Ñ­»·±éÀú£ºÑ­»·maxIter´Î ²¢ÇÒ £¨alphaPairsChanged´æÔÚ¿ÉÒÔ¸Ä±ä or ËùÓÐÐÐ±éÀúÒ»±é£©
    # Ñ­»·µü´ú½áÊø »òÕß Ñ­»·±éÀúËùÓÐalphaºó£¬alphaPairs»¹ÊÇÃ»±ä»¯
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  µ±entireSet=true or ·Ç±ß½çalpha¶ÔÃ»ÓÐÁË£»¾Í¿ªÊ¼Ñ°ÕÒ alpha¶Ô£¬È»ºó¾ö¶¨ÊÇ·ñÒª½øÐÐelse¡£
        if entireSet:
            # ÔÚÊý¾Ý¼¯ÉÏ±éÀúËùÓÐ¿ÉÄÜµÄalpha
            for i in range(oS.m):
                # ÊÇ·ñ´æÔÚalpha¶Ô£¬´æÔÚ¾Í+1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # ¶ÔÒÑ´æÔÚ alpha¶Ô£¬Ñ¡³ö·Ç±ß½çµÄalphaÖµ£¬½øÐÐÓÅ»¯¡£
        else:
            # ±éÀúËùÓÐµÄ·Ç±ß½çalphaÖµ£¬Ò²¾ÍÊÇ²»ÔÚ±ß½ç0»òCÉÏµÄÖµ¡£
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # Èç¹ûÕÒµ½alpha¶Ô£¬¾ÍÓÅ»¯·Ç±ß½çalphaÖµ£¬·ñÔò£¬¾ÍÖØÐÂ½øÐÐÑ°ÕÒ£¬Èç¹ûÑ°ÕÒÒ»±é ±éÀúËùÓÐµÄÐÐ»¹ÊÇÃ»ÕÒµ½£¬¾ÍÍË³öÑ­»·¡£
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    """
    »ùÓÚalpha¼ÆËãwÖµ
    Args:
        alphas        À­¸ñÀÊÈÕ³Ë×Ó
        dataArr       featureÊý¾Ý¼¯
        classLabels   Ä¿±ê±äÁ¿Êý¾Ý¼¯
    Returns:
        wc  »Ø¹éÏµÊý
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotfig_SVM(xArr, yArr, ws, b, alphas):
    """
    ²Î¿¼µØÖ·£º
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = mat(xArr)
    yMat = mat(yArr)

    # bÔ­À´ÊÇ¾ØÕó£¬ÏÈ×ªÎªÊý×éÀàÐÍºóÆäÊý×é´óÐ¡Îª£¨1,1£©£¬ËùÒÔºóÃæ¼Ó[0]£¬±äÎª(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ×¢ÒâflattenµÄÓÃ·¨
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x×î´óÖµ£¬×îÐ¡Öµ¸ù¾ÝÔ­Êý¾Ý¼¯dataArr[:, 0]µÄ´óÐ¡¶ø¶¨
    x = arange(-1.0, 10.0, 0.1)

    # ¸ù¾Ýx.w + b = 0 µÃµ½£¬ÆäÊ½×ÓÕ¹¿ªÎªw0.x1 + w1.x2 + b = 0, x2¾ÍÊÇyÖµ
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # ÕÒµ½Ö§³ÖÏòÁ¿£¬²¢ÔÚÍ¼ÖÐ±êºì
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


def run():
    # »ñÈ¡ÌØÕ÷ºÍÄ¿±ê±äÁ¿
    dataArr, labelArr = loadDataSet('./testSet.txt')
    # print labelArr

    # bÊÇ³£Á¿Öµ£¬ alphasÊÇÀ­¸ñÀÊÈÕ³Ë×Ó
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    print('/n/n/n')
    print('b=', b)
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # »­Í¼
    ws = calcWs(alphas, dataArr, labelArr)
    # plotfig_SVM(dataArr, labelArr, ws, b, alphas)
