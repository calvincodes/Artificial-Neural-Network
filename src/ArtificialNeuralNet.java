import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class ArtificialNeuralNet {

    public ArtificialNeuralNet(int trainingDataSize) {

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
    private double outputLayer;

    private Double[][] input2HiddenWeights;
    private Double[] hidden2OutputWeights;

    private Double biasInput2Hidden, biasHidden2Output;

    public void trainAnn(List<InstanceEntry> trainingDataSet, Double learningRate, Double numEpochs) {

        for (int iteration = 0; iteration < numEpochs.intValue(); iteration++) {

            if (trainingDataSet == null) {
                System.out.println("dafaq");
            }

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
                // Using Cross Entropy, L2 loss: -ylog(p)-(1-y)log(1-p)
                int actual
                        = trainingDataEntry.getClassLabel().equals(trainingDataEntry.getAllClassLabels()[0]) ? 0 : 1;

                /* ****************************************************************** */
                /* ****************************************************************** */

                // Step 3: Back propagation
                // Refer here for mathematical expression:
                // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

                // Step 3.1: Correction in hidden2OutputWeights
                Double[] correctedHidden2OutputWeights = new Double[trainingDataSize];
                for (int i = 0; i < trainingDataSize; i++) {

                    double valueChangeForWeight_i =
                            learningRate                    // learning rate
                                * hiddenLayer[i]            // output_h_i
                                * (outputLayer - actual)    // output_o * (1 - output_o)
                    ;
                    correctedHidden2OutputWeights[i] = hidden2OutputWeights[i] - valueChangeForWeight_i;
                }

                // Step 3.2: Correction in input2HiddenWeights
                Double[][] correctedInput2HiddenWeights = new Double[trainingDataSize][trainingDataSize];
                for (int i = 0; i < trainingDataSize; i++) {
                    for (int j = 0; j < trainingDataSize; j++) {

                        double valueChangeForWeight_i_j =
                                learningRate
                                        * (outputLayer - actual)
                                        * hidden2OutputWeights[j]
                                        * hiddenLayer[j] * (1 - hiddenLayer[j])
                                        * inputLayer[i];
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
            hidden2OutputWeights[i] = -1 + (2*Math.random());
            for (int j = 0; j < trainingDataSize; j++) {
                input2HiddenWeights[i][j] = -1 + (2*Math.random());
            }
        }

        biasInput2Hidden = -1 + (2*Math.random());
        biasHidden2Output = -1 + (2*Math.random());

//        for (int i = 0; i < trainingDataSize; i++) {
//            hidden2OutputWeights[i] = ThreadLocalRandom.current().nextDouble(-1d, 1d);
//            for (int j = 0; j < trainingDataSize; j++) {
//                input2HiddenWeights[i][j] = ThreadLocalRandom.current().nextDouble(-1d, 1d);
//            }
//        }
//
//        biasInput2Hidden = ThreadLocalRandom.current().nextDouble(-1d, 1d);
//        biasHidden2Output = ThreadLocalRandom.current().nextDouble(-1d, 1d);
    }
}
