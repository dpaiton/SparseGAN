import pdb
import numpy as np
import tensorflow as tf
from base import base
from utils import *
#Using pvp files for saving
#import pvtools as pv
from load_data import createQueueReader

class LCA_ADAM(base):
    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(LCA_ADAM, self).loadParams(params)
        self.learningRateA = params['learningRateA']
        self.learningRateW = params['learningRateW']
        self.thresh = params['thresh']
        self.numV = params['numV']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.inputFile = params["inputFile"]

    def runModel(self):
        # Coordinator manages threads, checks for stopping requests
        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(self.sess, coord=coord, start=True)
        #file_list = sess.run(filenames)
        #print(file_list)
        #print("\n-----\n")

        #Normalize weights to start
        self.normWeights()

        with coord.stop_on_exception():
          for i in range(self.numIterations):
            if(i%self.savePeriod == 0):
              self.trainA(True)
            else:
              self.trainA(False)
            #Train
            self.trainW()
            self.normWeights()
            #This function is responsible for determining when to plot per iteration
            #self.plot()
        coord.request_stop()
        coord.join(enqueue_threads)

        #Training

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params):
        super(LCA_ADAM, self).__init__(params)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        assert(inputShape[0] % self.VStrideY == 0)
        assert(inputShape[1] % self.VStrideX == 0)
        V_Y = int(inputShape[0]/self.VStrideY)
        V_X = int(inputShape[1]/self.VStrideX)
        self.imageShape = (self.batchSize, inputShape[0], inputShape[1], inputShape[2])
        self.WShape = (self.patchSizeY, self.patchSizeX, inputShape[2], self.numV)
        self.VShape = (self.batchSize, V_Y, V_X, self.numV)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                #self.inputImage = tf.placeholder("float", shape=self.imageShape, name="inputImage")
                (filename_batch, image_batch) = createQueueReader(self.inputFile, self.batchSize, self.inputShape)
                self.inputImage = tf.Variable(tf.zeros(self.imageShape, dtype=tf.float32), name="inputImage", trainable=False, dtype=tf.float32)
                self.updateImageOp = tf.assign(self.inputImage, image_batch)

                #self.zeros = tf.zeros(self.imageShape)
                #self.log_inputImage = tf.log(tf.abs(self.inputImage)) * tf.sign(self.inputImage)
                #self.select_inputImage = tf.select(tf.is_nan(self.log_inputImage), self.zeros, self.log_inputImage)

                #self.scaled_inputImage = self.scaled_inputImage/np.sqrt(self.patchSizeX*self.patchSizeY*inputShape[2])
                #Scale inputImage
                self.scaled_inputImage = self.inputImage/(np.sqrt(self.patchSizeX*self.patchSizeY*inputShape[2]))
                #self.checked_inputImage = tf.check_numerics(self.scaled_inputImage, "scaled_input error", name=None)

            with tf.name_scope("Dictionary"):
                self.V1_W = weight_variable(self.WShape, "V1_W", 1e-3)

            with tf.name_scope("weightNorm"):
                self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2], keep_dims=True))
                self.normVals = tf.verify_tensor_all_finite(self.normVals, 'V1W error', name=None)
                self.normalize_W = self.V1_W.assign(self.V1_W/(self.normVals + 1e-8))

            with tf.name_scope("LCA_ADAM"):
                self.V1_U = uniform_weight_variable(self.VShape, "V1_U", 0.0, 1.25*self.thresh)
                self.V1_A = weight_variable(self.VShape, "V1_A", 1e-3)

            with tf.name_scope("Recon"):
                assert(self.VStrideY >= 1)
                assert(self.VStrideX >= 1)
                #We build index tensor in numpy to gather
                self.recon = tf.nn.conv2d_transpose(self.V1_A, self.V1_W, self.imageShape, [1, self.VStrideY, self.VStrideX, 1], padding='SAME', name="recon")
                #self.recon = tf.check_numerics(self.recon, 'recon error', name=None)

            with tf.name_scope("Error"):
                self.error = self.scaled_inputImage - self.recon

            with tf.name_scope("Loss"):
                self.reconError = tf.reduce_mean(tf.reduce_sum(tf.square(self.error), reduction_indices=[1, 2, 3]))
                self.l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A), reduction_indices=[1, 2, 3]))
                #self.reconError = tf.reduce_mean(tf.square(self.error))
                #self.l1Sparsity = tf.reduce_mean(tf.abs(self.V1_A))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

            with tf.name_scope("Opt"):
                #Calculate A from U
                self.optimizerA0 = self.V1_A.assign(tf.nn.relu(self.V1_U - self.thresh))

                self.optimizerA1 = tf.train.AdamOptimizer(self.learningRateA)

                #Find gradient wrt A
                self.lossGrad = self.optimizerA1.compute_gradients(self.reconError, [self.V1_A])
                #self.checkGrad = tf.check_numerics(self.lossGrad[0][0], "grad error", name=None)
                self.dU = [(self.lossGrad[0][0] - self.V1_A + self.V1_U, self.V1_U)];

                #TODO add momentum or ADAM here
                self.optimizerA = self.optimizerA1.apply_gradients(self.dU)

                self.optimizerW = tf.train.AdadeltaOptimizer(self.learningRateW, epsilon=1e-6).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

            with tf.name_scope("stats"):
                self.nnz = tf.reduce_mean(tf.cast(tf.not_equal(self.V1_A, 0), tf.float32))

                self.imageStd = tf.sqrt(tf.reduce_mean(tf.square(self.scaled_inputImage - tf.reduce_mean(self.scaled_inputImage))))
                self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))/self.imageStd
                self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))

                self.weightImages = tf.squeeze(tf.transpose(self.V1_W, [3, 0, 1, 2]))

                #For log of activities
                self.log_V1_A = tf.log(tf.abs(self.V1_A)+1e-13)

        #Summaries
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('recon error', self.reconError)
        tf.summary.scalar('errorStd', self.errorStd)
        tf.summary.scalar('l1_sparsity', self.l1Sparsity)
        tf.summary.scalar('l1_mean', self.l1_mean)
        tf.summary.scalar('nnz', self.nnz)

        tf.summary.histogram('input', self.inputImage)
        tf.summary.histogram('scale_input', self.scaled_inputImage)
        tf.summary.histogram('recon', self.recon)
        tf.summary.histogram('V1_W', self.V1_W)
        tf.summary.histogram('V1_U', self.V1_U)
        tf.summary.histogram('V1_A', self.V1_A)
        tf.summary.histogram('Log_V1_A', self.log_V1_A)

        tf.summary.image("inputImage", self.inputImage)
        tf.summary.image("reconstruction", self.recon)

        #self.h_normVals = tf.histogram_summary('normVals', self.normVals, name="normVals")

    def encodeImage(self):
        #try:
            #Update to new image
            self.sess.run(self.updateImageOp)
            for i in range(self.displayPeriod):
                #Run optimizer
                #This calculates A
                self.sess.run(self.optimizerA0)
                #This updates U based on loss function wrt A
                self.sess.run(self.optimizerA)
                self.timestep+=1
                if((i+1)%self.writeStep == 0):
                    summary = self.sess.run(self.mergedSummary)
                    self.train_writer.add_summary(summary, self.timestep)
                if((i+1)%self.progress == 0):
                    print("Timestep ", self.timestep)
        #except:
        #    print("Error")
        #    pdb.set_trace()

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        #feedDict = {self.inputImage: self.currImg}
        self.encodeImage()

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        self.sess.run(self.normalize_W)


    #def plot(self):
    #    #Visualization
    #    if (self.plotTimestep % self.plotPeriod == 0):
    #        np_V1_W = self.sess.run(self.weightImages)
    #        np_V1_A = self.sess.run(self.V1_A)

    #        #plot_weights(rescaled_V1_W, self.plotDir+"dict_"+str(self.timestep), activity=np_V1_A)

    #        plotStr = self.plotDir + "dict_"+str(self.timestep)
    #        if(np_V1_W.ndim == 3):
    #            plot_1d_weights(np_V1_W, plotStr, activity=np_V1_A)
    #        else:
    #            plot_weights(V1_W, plotStr)

    #        np_inputImage = self.currImg
    #        feedDict = {self.inputImage: self.currImg}
    #        np_recon = self.sess.run(self.recon, feed_dict=feedDict)

    #        #Draw recons
    #        if(np.squeeze(np_recon).ndim == 2):
    #            rescaled_inputImage = np_inputImage * np.sqrt(self.patchSizeX * self.patchSizeY)
    #            rescaled_recon = np_recon * np.sqrt(self.patchSizeX * self.patchSizeY)

    #            exp_inputImage = np.squeeze(np.exp(np.abs(rescaled_inputImage) - 1e-10) * np.sign(np_inputImage))
    #            exp_recon = np.squeeze(np.exp(np.abs(rescaled_recon) - 1e-10) * np.sign(np_recon))

    #            plotRecon1d(exp_recon, exp_inputImage, self.plotDir+"recon_"+str(self.timestep), r=range(4))
    #        else:
    #            plotRecon(np_recon, np_inputImage, self.plotDir+"recon_"+str(self.timestep), r=range(4))

    #    self.plotTimestep += 1

    def trainW(self):
        #Update weights
        self.sess.run(self.optimizerW)


    #Finds sparse encoding of inData
    #inData must be in the shape of the image
    #[batch, nY, nX, nF]
    def evalData(self, inData):
        (nb, ny, nx, nf) = inData.shape
        #Check size
        assert(nb == self.batchSize)
        assert(ny == self.inputShape[0])
        assert(nx == self.inputShape[1])
        assert(nf == self.inputShape[2])

        #feedDict = {self.inputImage: inData}
        self.encodeImage()
        #Get thresholded v1 as an output
        outVals = self.V1_A.eval(session=self.sess)
        return outVals

    #def evalSet(self, evalDataObj, outFilename):
    #    numImages = evalDataObj.numImages
    #    #skip must be 1 for now
    #    assert(evalDataObj.skip == 1)
    #    numIterations = int(np.ceil(float(numImages)/self.batchSize))

    #    pvFile = pvpOpen(outFilename, 'w')
    #    for it in range(numIterations):
    #        print(str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")")
    #        #Evaluate
    #        npV1_A = self.evalData(self.currImg)
    #        v1Sparse = convertToSparse4d(npV1_A)
    #        time = range(it*self.batchSize, (it+1)*self.batchSize)
    #        data = {"values":v1Sparse, "time":time}
    #        pvFile.write(data, shape=(self.VShape[1], self.VShape[2], self.VShape[3]))
    #        self.currImg = self.dataObj.getData(self.batchSize)
    #    pvFile.close()

