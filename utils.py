#!/usr/bin/python
# -*- coding: utf-8 -*-

import random                               # Random Number
import re                                   # Regular Expression, 从数据文件读取数据
import math                                 # Math, 用于计算样本的标准差
import numpy                                # 用于比对类型(nddarray)

from sklearn import random_projection       # Random Projection
from scipy.stats import ttest_ind           # T-Test
from sklearn.svm import SVC                 # Support Vector Machine

def getSamples(filePath):
    """ 从数据文件中读取数据.

    Parameters
    ----------
    filePath : 一个字符串, 数据文件的路径

    Returns
    -------
    一个2D的List对象, 每一行表示某样本的全部基因表达值
    """
    file                = open(filePath)
    firstLine           = file.readline()
    # 获取数据规模
    numberOfSamples     = len(re.split(r'\t+', firstLine)) - 1
    numberOfGenes       = open(filePath).read().count('\n') - 1
    # 初始化
    samples             = [[None for x in range(numberOfGenes)] for y in range(numberOfSamples)]

    # 读取数据
    i = 0
    for line in file.readlines():
        geneExpression = line.split()
        # 忽略空白行
        if len(geneExpression) == 0:
            continue
        # 数据的每一行: 第1个值是基因的名称 (geneName = geneExpression[0]),
        #               第2个值开始是基因的表达值
        currentGeneExpression = geneExpression[1:]
        # 将文件中的数据读取至数组
        j = 0
        for value in currentGeneExpression:
            samples[j][i] = 'None' if value == 'null' else float(value)
            j += 1
        i += 1
    file.close()
    return samples

def getAverageAndVarianceAmongSamples(samples, index):
    """获取全体样本某个基因表达的均值和标准差.

    并使用计算所得的均值填补缺失值.

    Parameters
    ----------
    samples :  1个2D的List, 包含所有样本的基因表达值
    index : 1个int值, 表示需要计算的列的索引值

    Returns
    -------
    2个double值, 分别表示某个基因值的均值和标准差
    """
    # 所有基因表达值的和
    sum                     = 0
    # 非缺失值的数量
    numberOfExistingValue   = 0
    # 样本的均值与标准差
    average                 = 0
    stdVariance             = 0

    # 计算样本均值
    for sample in samples:
        thisValue = sample[index]

        if thisValue != 'None':
            numberOfExistingValue += 1
            sum += thisValue
    average = sum / numberOfExistingValue

    # 计算样本方差
    for dummyIndex, sample in enumerate(samples):
        thisValue = sample[index]

        if thisValue != 'None':
            stdVariance += (thisValue - average) * (thisValue - average)
        else:
            sample[index] = average
    stdVariance = math.sqrt(stdVariance / (numberOfExistingValue * 1.0))

    return average, stdVariance

def zScoreNormalization(samples):
    """Z-Score标准化.

    对所有的基因表达数据使用Z-Score进行标准化.

    Parameters
    ----------
    samples : 1个2D的List, 包含所有样本的基因表达值
    """
    averageValues       = []
    stdVarianceValues   = []

    if len(samples) == 0:
        raise Exception('No samples.')
    else:
        # 计算每个基因的均值和标准差
        # (仅计算第一行即可)
        numberOfGenes = len(samples[0])
        for index in range(0, numberOfGenes):
            average, stdVariance = getAverageAndVarianceAmongSamples(samples, index)
            averageValues.append(average)
            stdVarianceValues.append(stdVariance)

    # Z-Score Normalization
    for i, sample in enumerate(samples):
        for j, value in enumerate(sample):
            average         = averageValues[j]
            stdVariance     = stdVarianceValues[j]
            samples[i][j]   = (value - average) / stdVariance

def getTraningAndTestingSamples(samples, sampleIndexes, numberOfTraningSamples, numberOfTestingSamples):
    """从样本中采样生成训练数据和测试数据.

    Parameters
    ----------
    samples : 1个2D的List, 包含全部样本(已降维)的基因表达值
    sampleIndexes : 1个1DList, 表示可以采样的样本索引值
    numberOfTraningSamples : 1个int值, 采样作为训练样本的数量
    numberOfTestingSamples : 1个int值, 采样作为测试样本的数量

    Returns
    -------
    2个1D的List, 分别存储(已降维的)训练样本和(已降维的)测试样本
    """
    trainingSampleIndexes    = random.sample(sampleIndexes, numberOfTraningSamples)
    testingSampleIndexes     = random.sample(list(set(sampleIndexes) - set(trainingSampleIndexes)), numberOfTestingSamples)

    trainingSamples          = []
    testingSamples           = []
    
    for index in trainingSampleIndexes:
        trainingSamples.append(samples[index])
    for index in testingSampleIndexes:
        testingSamples.append(samples[index])

    return trainingSamples, testingSamples

def getClassificationErrorSamples(trainingSamples, trainingSampleLabels, testingSamples, testingSampleLabels):
    """评估降维后的数据的分类精度.

    使用降维后的数据训练一个SVM分类器, 并测试分类精度.

    Parameters
    ----------
    trainingSamples : 一个2D的List, 包含训练样本的基因表达值
    trainingSampleLabels : 一个1D的List, 包含训练样本的Class Labels
    testingSamples : 一个2D的List, 包含测试样本的基因表达值
    testingSampleLabels : 一个1D的List, 包含测试样本的Class Labels

    Returns
    -------
    5个int类型数据:
    - 第1个参数表示训练集上的误分类样本数
    - 第2个参数表示TP (True Positive)
    - 第3个参数表示FP (False Positive, Predicted Positive)
    - 第4个参数表示FN (False Negative, Predicted Negative)
    - 第5个参数表示TN (False Negative)
    """
    # Create SVM
    clf = SVC()
    clf.fit(trainingSamples, trainingSampleLabels)
    predictedTrainingSampleLabels   = clf.predict(trainingSamples)
    predictedTestingSampleLabels    = clf.predict(testingSamples)

    # Compare Classification Result
    numberOfTrainingSamples         = len(trainingSampleLabels)
    numberOfErrorTrainingSamples    = 0
    numberOfTestingSamples          = len(testingSampleLabels)
    tp                              = 0
    fp                              = 0
    fn                              = 0
    tn                              = 0

    # Training Error Statistics
    for i in range(0, numberOfTrainingSamples):
        if trainingSampleLabels[i] != predictedTrainingSampleLabels[i]:
            numberOfErrorTrainingSamples = numberOfErrorTrainingSamples + 1

    # Testing Error Statistics
    # 0 stands for Nomal, 1 stands for Tumor
    for i in range(0, numberOfTestingSamples):
        if testingSampleLabels[i] == 0 and predictedTestingSampleLabels[i] == 0:
            # TP
            tp += 1
        elif testingSampleLabels[i] == 1 and predictedTestingSampleLabels[i] == 0:
            # FP
            fp += 1
        elif testingSampleLabels[i] == 0 and predictedTestingSampleLabels[i] == 1:
            # FN
            fn += 1
        elif testingSampleLabels[i] == 1 and predictedTestingSampleLabels[i] == 1:
            # TN
            tn += 1

    return numberOfErrorTrainingSamples, tp, fp, fn, tn

def gaussianRandomProjection(samples, targetDimensionality):
    """Gaussian Random Projection.

        Parameters
        ----------
        samples : 1个2D的List, 包含所有样本的基因表达值
        targetDimensionality : 降维后的目标维度(选取基因的数量)

        Returns
        -------
        2个2D的List.
        - 第1个List表示所有样本的(已降维的)基因表达值,
        - TODO
    """
    transformer             = random_projection.GaussianRandomProjection(n_components = targetDimensionality)

    return transformer.fit_transform(samples), transformer

def sparseRandomProjection(samples, targetDimensionality):
    """Sparse Random Projection.

        Parameters
        ----------
        samples : 1个2D的List, 包含所有样本的基因表达值
        targetDimensionality : 降维后的目标维度(选取基因的数量)

        Returns
        -------
        2个2D的List.
        - 第1个List表示所有样本的(已降维的)基因表达值,
        - TODO

    """
    transformer             = random_projection.SparseRandomProjection(n_components = targetDimensionality)
    
    return transformer.fit_transform(samples), transformer

def featureSelection(samples, labels, numberOfGenes, targetDimensionality):
    """Feature Selection.

    从全部基因中选取p-Value较大的基因.

    Parameters
    ----------
    samples : 1个2D的List, 包含样本的基因表达值
    labels : 1个1D的List, 包含样本的标签
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionality : 降维后的目标维度(选取基因的数量)

    Returns
    -------
    1个2D的List和1个1D的List
    - 第1个返回值表示训练样本的(已降维的)基因表达值
    - 第2个返回值表示被选中的基因的索引值
    """
    normalSamples   = []
    tumorSamples    = []

    for index, label in enumerate(labels):
        if label == 0:
            normalSamples.append(samples[index])
        else:
            tumorSamples.append(samples[index])

    # Select Genes with Bigger P-Value
    selectedGeneIndexes     = getSelectedGeneWithBiggerPValues(normalSamples, tumorSamples, numberOfGenes, targetDimensionality)
    reducedSamples          = getReducedSamples(samples, selectedGeneIndexes)

    return reducedSamples, selectedGeneIndexes

def getSelectedGeneWithBiggerPValues(normalSamples, tumorSamples, numberOfGenes, targetDimensionality):
    """选取p-Value较大的基因.


    Parameters
    ----------
    normalSamples : 1个2D的List, 正常人样本的基因表达值
    tumorSamples : 1个2D的List, 癌症病人样本的基因表达值
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionality : 降维后的目标维度(选取基因的数量)

    Returns
    -------
    1个1D的List, 表示被选取的基因的索引值
    """
    pValues = dict()
    
    # Calculate p-Value for all Genes
    for geneIndex in range(0, numberOfGenes):
        normalGeneValues    = []
        tumorGeneValues     = []

        for normalSample in normalSamples:
            normalGeneValues.append(normalSample[geneIndex])
        for tumorSample in tumorSamples:
            tumorGeneValues.append(tumorSample[geneIndex])
        # Calculate p-Value for this Gene
        pValues[geneIndex] = getPValueOfGene(normalGeneValues, tumorGeneValues)

    # Select Top-k Genes
    geneIndexesOrderByPValues = sorted(pValues, key=pValues.get)
    return geneIndexesOrderByPValues[:targetDimensionality]

def getPValueOfGene(samplesA, samplesB):
    """获取两类样本的P-Value值.

    Parameters
    ----------
    samplesA : 1个1D的List, 表示样本A的表达值数据
    samplesB : 1个1D的List, 表示样本B的表达值数据

    Returns
    -------
    1个double值, 表示两类样本的P-Value值
    """
    t, p = ttest_ind(samplesA, samplesB, equal_var=False)
    return p


def getSelectedGeneWithBiggerAverageDifferences(normalSamples, tumorSamples, numberOfGenes, targetDimensionality):
    """选取基因均值差较大的基因.


    Parameters
    ----------
    normalSamples : 1个2D的List, 正常人样本的基因表达值
    tumorSamples : 1个2D的List, 癌症病人样本的基因表达值
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionality : 降维后的目标维度(选取基因的数量)

    Returns
    -------
    1个1D的List, 表示被选取的基因的索引值
    """
    averageDifferences = dict()

    # Calculate p-Value for all Genes
    for geneIndex in range(0, numberOfGenes):
        normalGeneValues    = []
        tumorGeneValues     = []

        for normalSample in normalSamples:
            normalGeneValues.append(normalSample[geneIndex])
        for tumorSample in tumorSamples:
            tumorGeneValues.append(tumorSample[geneIndex])
        # Calculate p-Value for this Gene
        averageDifferences[geneIndex] = getAverageDifferencesOfGene(normalGeneValues, tumorGeneValues)

    # Select Top-k Genes
    geneIndexesOrderByDifferences = sorted(averageDifferences, key=averageDifferences.get, reverse=False)
    return geneIndexesOrderByDifferences[:targetDimensionality]

def getAverageDifferencesOfGene(normalGeneValues, tumorGeneValues):
    return abs(numpy.average(normalGeneValues) - numpy.average(tumorGeneValues))

def greedyFeatureSelection(samples, sampleLabels, numberOfGenes, targetDimensionality):
    """贪婪的 Feature Selection(特征选择).

    从(随即投影后的)样本中获取若干(targetDimensionality)个最有利于分类的基因.
    对于每一个基因训练一个SVM, 选取分类精度最高的基因.

    Parameters
    -------
    samples : 1个2D的List, 包含样本的基因表达值
    sampleLabels : 1个1D的List, 包含样本的标签
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionality : 1个int值, 降维后的目标维度(选取基因的数量)

    Returns
    -------
    1个2D的List和1个1D的List
    - 第1个返回值表示训练样本的(已降维的)基因表达值
    - 第2个返回值表示被选中的基因的索引值
    """
    # 使用放回的取样方式选择30个样本作为训练集, 30个样本作为测试集
    normalSampleIndexes                         = []
    tumorSampleIndexes                          = []

    for index, sampleLabel in enumerate(sampleLabels):
        if sampleLabel == 0:
            normalSampleIndexes.append(index)
        else:
            tumorSampleIndexes.append(index)

    numberOfSamples                             = 30

    normalTrainingSampleIndexes                 = random.sample(normalSampleIndexes, numberOfSamples)
    normalTestingSampleIndexes                  = random.sample(normalSampleIndexes, numberOfSamples)
    normalTrainingSamples                       = []
    normalTestingSamples                        = []
    for index in normalTrainingSampleIndexes:
        normalTrainingSamples.append(samples[index])
    for index in normalTestingSampleIndexes:
        normalTestingSamples.append(samples[index])
    
    tumorTrainingSampleIndexes                  = random.sample(tumorSampleIndexes, numberOfSamples)
    tumorTestingSampleIndexes                   = random.sample(tumorSampleIndexes, numberOfSamples)
    tumorTrainingSamples                        = []
    tumorTestingSamples                         = []
    for index in tumorTrainingSampleIndexes:
        tumorTrainingSamples.append(samples[index])
    for index in tumorTestingSampleIndexes:
        tumorTestingSamples.append(samples[index])

    trainingSamples                             = normalTrainingSamples + tumorTrainingSamples
    testingSamples                              = normalTestingSamples + tumorTestingSamples

    trainingSampleLabels                        = [0] * numberOfSamples + [1] * numberOfSamples
    testingSampleLabels                         = [0] * numberOfSamples + [1] * numberOfSamples

    # Get Genes with Lower Classification Error
    selectedGeneIndexes                         = getSelectedGeneWithLowerClassificationError(trainingSamples, trainingSampleLabels, testingSamples, testingSampleLabels, numberOfGenes, targetDimensionality)
    reducedSamples                              = getReducedSamples(samples, selectedGeneIndexes)

    return reducedSamples, selectedGeneIndexes

def getSelectedGeneWithLowerClassificationError(trainingSamples, trainingSampleLabels, testingSamples, testingSampleLabels, numberOfGenes, targetDimensionality):
    """根据单个基因的分类误差选择较低分类误差的基因.

    Parameters
    ----------
    trainingSamples : 1个2D的List, 包含正常人样本的基因表达值
    trainingSampleLabels : 1个1D的List, 包含正常人样本的标签
    testingSamples : 1个2D的List, 包含癌症病人样本的基因表达值
    testingSampleLabels : 1个1D的List, 包含癌症病人样本的标签
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionality : 1个int值, 降维后的目标维度(选取基因的数量)

    Returns
    -------
    1个1D的List, 表示被选中的基因的索引值
    """
    classificationError = dict()

    # Calculate Classification Accuracy for all Genes
    for geneIndex in range(0, numberOfGenes):
        trainingSampleGeneValues    = []
        testingSampleGeneValues     = []

        for trainingSample in trainingSamples:
            trainingSampleGeneValues.append([trainingSample[geneIndex]])
        for testingSample in testingSamples:
            testingSampleGeneValues.append([testingSample[geneIndex]])
        
        trainingErrorSamples, tp, fp, fn, tn        = getClassificationErrorSamples(trainingSampleGeneValues, trainingSampleLabels, testingSampleGeneValues, testingSampleLabels)
        classificationError[geneIndex]              = fp + fn

    # Select Top-k Genes
    geneIndexesOrderByClassificationError = sorted(classificationError, key = classificationError.get)
    return geneIndexesOrderByClassificationError[:targetDimensionality]

def getReducedSamples(samples, selectedGeneIndexes):
    """根据选定基因的索引值对基因表达矩阵进行维度规约.

    Parameters
    -------
    samples : 1个2D的List, 包含样本的基因表达值
    selectedGeneIndexes : 1个1D的List, 表示被选中的基因的索引值

    Returns
    -------
    1个2D的List, 降维后的样本的基因表达值
    """
    numberOfSamples         = len(samples)
    targetDimensionality    = len(selectedGeneIndexes)
    reducedSamples          = [[None for x in range(targetDimensionality)] for y in range(numberOfSamples)]

    for i in range(numberOfSamples):
        j = 0
        for selectedGeneIndex in selectedGeneIndexes:
            reducedSamples[i][j] = samples[i][selectedGeneIndex]
            j += 1
    
    return reducedSamples
