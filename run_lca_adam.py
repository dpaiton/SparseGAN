#import matplotlib
#matplotlib.use('Agg')
from lca_adam import LCA_ADAM
import numpy as np
import pdb

params = {
    "inputFile":       "/media/tbell/datasets/natural_images.txt",
    "inputShape":      [256, 256, 3],
    #Base output directory
    'outDir':          "/home/slundqui/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/sparse_code/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       10,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundqui/mountData/tfSparseCode/saved/fista_cifar_nf256.ckpt",
    #Device to run on
    'device':          '/gpu:0',
    #####FISTA PARAMS######
    'numIterations':   10,
    'displayPeriod':   400,
    #Batch size
    'batchSize':       16,
    #Learning rate for optimizer
    'learningRateA':   5e-3,
    'learningRateW':   1,
    #Lambda in energy function
    'thresh':          .01,
    #Number of features in V1
    'numV':            256,
    #Stride of V1
    'VStrideY':        2,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      12,
    'patchSizeX':      12,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

