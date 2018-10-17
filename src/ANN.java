import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class ANN {

    public ANN(int trainingDataSize) {

        this.trainingDataSize = trainingDataSize;

        this.inputLayer = new Double[trainingDataSize];
        this.hiddenLayer = new Double[trainingDataSize];

        this.input2HiddenWeights = new Double[trainingDataSize][trainingDataSize];
        this.hidden2OutputWeights = new Double[trainingDataSize];
        initialize();
    }

    private int trainingDataSize;

    private Double[] inputLayer;
    private Double[] hiddenLayer;
    private Double outputLayer;

    private Double[][] input2HiddenWeights;
    private Double[] hidden2OutputWeights;

    private Double biasInput2Hidden, biasHidden2Output;

    public void trainAnn(List<InstanceEntry> trainingDataSet, Double learningRate, Double numEpochs) {

        for (int iteration = 0; iteration < numEpochs.intValue(); iteration++) {

            // Refer here for a brief explanation of the algorithm
            // https://www.nnwj.de/backpropagation.html
            for (InstanceEntry trainingDataEntry : trainingDataSet) {

                /* ****************************************************************** */

                // Step 1: Forward Propagation phase

                // Step 1.1: Initialize input layer with training data
                inputLayer = trainingDataEntry.getFeatureValues();

                // Step 1.2: Compute node values of hidden layer
                for (int i = 0; i < trainingDataSize; i++) {
                    double currentInput2HiddenLayer = 0d;
                    for (int j = 0; j < trainingDataSize; j++) {
//                        currentInput2HiddenLayer += (inputLayer[j] * input2HiddenWeights[i][j]);
                        currentInput2HiddenLayer += (inputLayer[j] * input2HiddenWeights[j][i]);
                    }
                    currentInput2HiddenLayer += biasInput2Hidden;
                    hiddenLayer[i] = 1 / (1 + Math.exp(-currentInput2HiddenLayer));
                }

                // Step 1.3: Compute node value of output layer
                Double input2OutputLayer = 0d;
                for (int i = 0; i < trainingDataSize; i++) {
                    input2OutputLayer += (hiddenLayer[i] * hidden2OutputWeights[i]);
                }
                input2OutputLayer += biasHidden2Output;
                outputLayer = 1 / (1 + Math.exp(-input2OutputLayer));

                /* ****************************************************************** */
                /* ****************************************************************** */

                // Step 2: Error computation

                double error = 0d;
                int actual = 0;
                // cross-entropy, L2 loss: -ylog(yp)-(1-y)log(1-yp)
                if (trainingDataEntry.getClassLabel().equals(trainingDataEntry.getAllClassLabels()[0])) {
//                    error = 1;
//                    error = 1 / (1 - outputLayer);
//                    error = 0 - outputLayer;
                    // Output Layer value should be < 0.5 in error free case.
//                    if (outputLayer >= 0.5) {
//                        error = 0 - outputLayer;
////                        error = 1d;
////                    error = -Math.log(1-outputLayer); // y = 0
//                    }
                } else {
                    actual = 1;
//                    error = -1;
//                    error = - 1 / (outputLayer);
//                    error = 1 - outputLayer;
                    // Output Layer value should be >= 0.5 in error free case.
//                    if (outputLayer < 0.5) {
//                        error = 1 - outputLayer;
////                        error = -1d;
////                    error = -Math.log(outputLayer); // y = 1
//                    }
                }

                /* ****************************************************************** */
                /* ****************************************************************** */

                // Step 3: Back propagation
                // Refer here for mathematical expression:
                // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

                // Step 3.1: Correction in hidden2OutputWeights
                Double[] correctedHidden2OutputWeights = new Double[trainingDataSize];
                for (int i = 0; i < trainingDataSize; i++) {
//                    Double valueChangeForWeight_i
//                            = learningRate                           // learning rate
//                            * error                             // - (target - outputLayer)
//                            * outputLayer
//                            ;
                    double valueChangeForWeight_i;
                    if (actual == 0) {
//                        error = 1 / (1 - outputLayer);
                        valueChangeForWeight_i
                                = learningRate                           // learning rate
                                * hiddenLayer[i]                         // output_h_i
                                * outputLayer       // output_o * (1 - output_o)
                                ;
                    } else  {
//                        error = - 1 / (outputLayer);
                        valueChangeForWeight_i
                                = learningRate                           // learning rate
                                * hiddenLayer[i]                         // output_h_i
                                * (outputLayer - 1)        // output_o * (1 - output_o)
                                ;
                    }
//                    Double valueChangeForWeight_i
//                            = learningRate                           // learning rate
//                            * error                             // - (target - outputLayer)
//                            * hiddenLayer[i]                         // output_h_i
//                            * outputLayer * (1 - outputLayer)        // output_o * (1 - output_o)
//                            ;
                    correctedHidden2OutputWeights[i] = hidden2OutputWeights[i] - valueChangeForWeight_i;
//                    hidden2OutputWeights[i] = hidden2OutputWeights[i] + valueChangeForWeight_i;
                }

                // Step 3.2: Correction in input2HiddenWeights
                Double[][] correctedInput2HiddenWeights = new Double[trainingDataSize][trainingDataSize];
                for (int i = 0; i < trainingDataSize; i++) {
                    for (int j = 0; j < trainingDataSize; j++) {
//                        Double valueChangeForWeight_i_j
//                                = learningRate                           // learning rate
//                                * error                             // - (target - outputLayer)
//                                * hiddenLayer[j];                         // input_i

                        double valueChangeForWeight_i_j;
                        if (actual == 0) {
                            valueChangeForWeight_i_j =
                                    learningRate
                                    * outputLayer
                                    * hidden2OutputWeights[j]
                                    * hiddenLayer[j] * (1 - hiddenLayer[j])
                                    * inputLayer[i];
                        } else {
                            valueChangeForWeight_i_j =
                                    learningRate
                                    * -1
                                    * (1 - outputLayer)
                                    * hidden2OutputWeights[j]
                                    * hiddenLayer[j] * (1 - hiddenLayer[j])
                                    * inputLayer[i];
                        }
//                        Double valueChangeForWeight_i_j
//                                = learningRate                           // learning rate
//                                * error                             // - (target - outputLayer)
//                                * hiddenLayer[j] * (1 - hiddenLayer[j])  // output_h_j * (1 - output_h_j)
//                                * inputLayer[i];                         // input_i
//                        input2HiddenWeights[i][j] = input2HiddenWeights[i][j] + valueChangeForWeight_i_j;
                        correctedInput2HiddenWeights[i][j] = input2HiddenWeights[i][j] - valueChangeForWeight_i_j;
                    }
                }

                input2HiddenWeights = correctedInput2HiddenWeights;
                hidden2OutputWeights = correctedHidden2OutputWeights;

                /* ****************************************************************** */
                /* ****************************************************************** */

            }

        }
    }

    public void evaluate(InstanceEntry testEntry) {

        // Step 1.1: Initialize input layer with training data
        Double[] inputLayer = testEntry.getFeatureValues().clone();

        Double[] hiddenLayer = new Double[trainingDataSize];
        double outputLayer;

        // Step 1.2: Compute node values of hidden layer
        for (int i = 0; i < trainingDataSize; i++) {
            Double currentInput2HidderLayer = 0d;
            for (int j = 0; j < trainingDataSize; j++) {
                currentInput2HidderLayer += (inputLayer[j] * input2HiddenWeights[j][i]);
            }
            currentInput2HidderLayer += biasInput2Hidden;
            hiddenLayer[i] = 1 / (1 + Math.exp(-currentInput2HidderLayer));
        }

        // Step 1.3: Compute node value of output layer
        Double input2OutputLayer = 0d;
        for (int i = 0; i < trainingDataSize; i++) {
            input2OutputLayer += (hiddenLayer[i] * hidden2OutputWeights[i]);
        }
        input2OutputLayer += biasHidden2Output;
        outputLayer = 1 / (1 + Math.exp(-1*input2OutputLayer));

        testEntry.setPredictionConfidence(outputLayer);
        if (outputLayer < 0.5) {
            testEntry.setPredictedClassLabel(testEntry.getAllClassLabels()[0]);
        } else {
            testEntry.setPredictedClassLabel(testEntry.getAllClassLabels()[1]);
        }
    }

    public void initialize() {

        for (int i = 0; i < trainingDataSize; i++) {
            hidden2OutputWeights[i] = ThreadLocalRandom.current().nextDouble(-1d, 1d);
            for (int j = 0; j < trainingDataSize; j++) {
                input2HiddenWeights[i][j] = ThreadLocalRandom.current().nextDouble(-1d, 1d);
            }
        }

        biasInput2Hidden = ThreadLocalRandom.current().nextDouble(-1d, 1d);
        biasHidden2Output = ThreadLocalRandom.current().nextDouble(-1d, 1d);
//        biasInput2Hidden = 0d;
//        biasHidden2Output = 0d;
    }
}
