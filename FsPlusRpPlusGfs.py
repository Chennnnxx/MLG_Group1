#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import dot                           # Multiply two matrices
from sys import argv                            # Get Parameters for the Script
from sys import exit                            # Terminate the Script

from utils import getSamples                    # Read Data from Files
from utils import zScoreNormalization           # Data Preprocessing
from utils import featureSelection              # Feature Selection
from utils import getReducedSamples             # Feature Selection
from utils import gaussianRandomProjection      # Gaussian Random Projection
from utils import sparseRandomProjection        # Sparse Random Projection
from utils import greedyFeatureSelection
from utils import getTraningAndTestingSamples   # Generate Traning and Testing Data
from utils import getClassificationErrorSamples # Get Classification Error After Dimensionality Reducation

def featureSelectionPlusRandomProjectionPlusGreedyFeatureSelectionProceduce(
        samples, normalSampleIndexes, tumorSampleIndexes, numberOfGenes, targetDimensionalityFS, 
        targetDimensionalityRP, targetDimensionalityGFS, numberOfNormalSamples, numberOfTumorSamples):
    """Random Projection, Feature Selection and Greedy Feature Selection.

    在执行Feature Selection之后, 执行Random Projection, 接着执行Greedy Feature Selection.

    Parameters
    ----------
    samples : 1个2D的List, 包含所有样本的基因表达值
    normalSampleIndexes : 1个1D的List, 正常人样本的索引值
    tumorSampleIndexes : 1个ID的List, 癌症病人样本的索引值
    numberOfGenes : 1个int值, 表示所有基因的数量
    targetDimensionalityFS : Feature Selection降维后的目标维度(选取基因的数量)
    targetDimensionalityRP : Random Projection降维后的目标维度(选取基因的数量)
    targetDimensionalityGFS : Greedy Feature Selection降维后的目标维度(选取基因的数量)
    numberOfNormalSamples : 采样时正常人样本的数量
    numberOfTumorSamples : 采样时癌症病人样本的数量

    Returns
    ----------
    runningTime : 1个double值, Feature Selection阶段所用的时间
    trainingErrorSamples : 1个int值, SVM分类器在训练集上的误分类样本数
    tp : 1个int值, TP (True Positive)
    fp : 1个int值, FP (False Positive, Predicted Positive)
    fn : 1个int值, FN (False Negative, Predicted Negative)
    tn : 1个int值, TN (False Negative)
    """
    """Generating Traning Data and Testing Data"""
    # Generateing Traning Data and Testing Data for Normal People
    normalTrainingSamples, normalTestingSamples = getTraningAndTestingSamples(samples, normalSampleIndexes, numberOfNormalSamples, numberOfNormalSamples)
    # Generateing Traning Data and Testing Data for Tumor People
    tumorTrainingSamples, tumorTestingSamples   = getTraningAndTestingSamples(samples, tumorSampleIndexes, numberOfTumorSamples, numberOfTumorSamples)
    # Genereating Traning Data and Testing Data
    trainingSamples                             = normalTrainingSamples + tumorTrainingSamples
    testingSamples                              = normalTestingSamples + tumorTestingSamples
    # Generate Labels for Samples
    trainingSampleLabels                        = [0] * numberOfNormalSamples + [1] * numberOfTumorSamples
    testingSampleLabels                         = [0] * numberOfNormalSamples + [1] * numberOfTumorSamples
    

    """Feature Selection + Random Projection + Greedy Feature Selection"""
    trainingSamples, selectedGeneIndexesFS      = featureSelection(trainingSamples, trainingSampleLabels, numberOfGenes, targetDimensionalityFS)
    trainingSamples, transformer                = sparseRandomProjection(trainingSamples, targetDimensionalityRP)
    trainingSamples, selectedGeneIndexesGFS     = greedyFeatureSelection(trainingSamples, trainingSampleLabels, targetDimensionalityRP, targetDimensionalityGFS)
    # Apply Random Projection to Testing Data
    testingSamples                              = getReducedSamples(testingSamples, selectedGeneIndexesFS)
    testingSamples                              = transformer.transform(testingSamples)
    testingSamples                              = getReducedSamples(testingSamples, selectedGeneIndexesGFS)
    print('[Result] Selected Gene Index after Feature Selection: {}'.format(repr(selectedGeneIndexesFS)))
    print('[Result] Selected Gene Index after Random Projection and Greedy Feature Selection: {}'.format(repr(selectedGeneIndexesGFS)))
    """Test Classification Accuracy"""
    trainingErrorSamples, tp, fp, fn, tn        = getClassificationErrorSamples(trainingSamples, trainingSampleLabels, testingSamples, testingSampleLabels)


    return trainingErrorSamples, tp, fp, fn, tn





"""
    PLEASE NOTE: THIS IS THE END OF THE FEATURE SELECTION AND RANDOM PROJECTION AND GREEDY FEATURE SELECTION PROCEDURE 
                 ALSO THE BEGINNING OF THE SCRIPT
"""
def main():
    if len(argv) != 8:
        print('Usage: FsPlusRpPlusGfs NormalFilePath TumorFilePath TargetDimensionalityFS TargetDimensionalityRP TargetDimensionalityGFS NumberOfNormalSamples NumberOfTumorSamples')
        exit()

    """Parameters for executing the script"""
    # File Path of Normal People's Data
    normalFilePath                  = argv[1]
    # File Path of Tumor People's Data
    tumorFilePath                   = argv[2]
    # The Dimensionality of Subspace after Feature Selection Stage
    targetDimensionalityFS          = int(argv[3])
    # The Dimensionality of Subspace after Random Projection Stage
    targetDimensionalityRP          = int(argv[4])
    # The Dimensionality of Subspace after Greedy Feature Selection Stage
    targetDimensionalityGFS         = int(argv[5])
    # Number of Samples for Normal People
    numberOfNormalTestingSamples    = int(argv[6])
    # Number of Samples for Tumor People
    numberOfTumorTestingSamples     = int(argv[7])

    """Read data from files"""
    # Sample of Normal People
    normalSamples                   = getSamples(normalFilePath)
    numberOfNormalSamples           = len(normalSamples) 
    # Samples of Tumor People
    tumorSamples                    = getSamples(tumorFilePath)
    numberOfTumorSamples            = len(tumorSamples)
    # All Samples
    samples                         = normalSamples + tumorSamples
    numberOfSamples                 = len(samples)
    # Gene Indexes in List: samples
    normalSampleIndexes             = range(0, numberOfNormalSamples)
    tumorSampleIndexes              = range(numberOfNormalSamples, numberOfSamples)

    numberOfGenes                   = len(samples[0]) if len(samples) != 0 else 0
    print('Original Data Matrix: {} Samples with {} Genes'.format(numberOfSamples, numberOfGenes))

    """Data preprocessing"""
    zScoreNormalization(samples)


    """Runtime Result"""
    trainingErrorSamples, tp, fp, fn, tn = \
        featureSelectionPlusRandomProjectionPlusGreedyFeatureSelectionProceduce(
            samples, normalSampleIndexes, tumorSampleIndexes, numberOfGenes, targetDimensionalityFS, 
            targetDimensionalityRP, targetDimensionalityGFS, numberOfNormalTestingSamples, numberOfTumorTestingSamples)

    
   #print('[Result] Training Error Samples: {}'.format(repr(trainingErrorSamples)))
   #print('[Result] Testing Error Samples: {}'.format(repr(fp + fn)))
   #print('[Result] TP = {}, FP = {}, FN = {}, TN = {}'.format(repr(tp), repr(fp), repr(fn), repr(tn)))

if __name__ == "__main__":
    main()
