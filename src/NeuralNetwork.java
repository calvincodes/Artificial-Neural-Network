//import java.util.Collections;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
//public class NeuralNetwork {
//
//    private Node[] inputLayer;
//    private Node[] hiddenLayer;
//    private Node outputLayer;
//    private int fold;
//    private int epoch;
//    private double learningRate;
//    Node hiddenLayerBias;
//    Node outputLayerBias;
//
//    public NeuralNetwork(int inputNodeCount, int hiddenNodeCount, int fold, int epoch, double learningRate) {
//        this.inputLayer = new Node[inputNodeCount];
//        this.hiddenLayer = new Node[hiddenNodeCount];
//        this.fold = fold;
//        this.epoch = epoch;
//        this.learningRate = learningRate;
//    }
//
////    -1+2*Math.random()
//
//    void randomizeWeightOfAlNodes() {
//        for (int i = 0; i < inputLayer.length; i++) {
//            this.inputLayer[i] = new Node();
//
//        }
//        this.hiddenLayerBias = new Node(1d);
//        this.outputLayerBias = new Node(1d);
//        for (int i = 0; i < hiddenLayer.length; i++) {
//            this.hiddenLayer[i] = new Node();
//            this.hiddenLayer[i].addIncomingConnections(inputLayer, true);
//            this.hiddenLayer[i].addBiasConnection(hiddenLayerBias, true);
//        }
//
//        outputLayer = new Node();
//        outputLayer.addIncomingConnections(hiddenLayer, true);
//        outputLayer.addBiasConnection(outputLayerBias, true);
//    }
//
//    public void run(List<Instance> trainingInstance) {
//        Map<String, List<Instance>> splitInstances = Helper.splitInstances(trainingInstance);
//        Map<Integer, List<Instance>> finalFold = Helper.generateKFold(splitInstances,this.fold);
//        for (int i = 0; i < this.fold; i++) {
//            for (int j = 0; j < this.fold; j++) {
//                if( i!=j) {
//                    for (int iteration = 0; iteration < this.epoch; iteration++) {
//                        List<Instance> instances = finalFold.get(j);
//                        for (Instance instance : instances) {
//                            transformInstanceIntoInputLayer(instance);
//                            for (Node node : this.hiddenLayer) {
//                                double net = 0d;
//                                for (Connection connection : node.incomingConnection) {
//                                    net += connection.weight * connection.leftNeuron.weight;
//                                }
//                                node.weight = Helper.Sigmoid(net);
//                            }
//
//                            double net = 0;
//                            for (Connection connection : outputLayer.incomingConnection) {
//                                net += connection.weight * connection.leftNeuron.weight;
//                            }
//                            outputLayer.weight = Helper.Sigmoid(net);
//                            double actualOutput = 0;
//                            if (instance.final_class.equals(Driver.allHeaders.get(instance.ft_names[instance.ft_names.length - 1]).get(1))) {
//                                actualOutput = 1;
//                            }
//                            double diff = actualOutput - outputLayer.weight;
//
//
//                            // Changing weight of hidden to output layer
//
//                            for (Node hiddenNode : this.hiddenLayer) {
//                                double valueForChangingWeight = this.learningRate * diff * hiddenNode.weight * outputLayer.weight * (1 - outputLayer.weight);
//                                for (Connection connection : outputLayer.incomingConnection) {
//                                    if (connection.leftNeuron.id == hiddenNode.id) {
//                                        connection.weight = connection.weight + valueForChangingWeight;
//                                    }
//                                }
//                            }
//
//                            // Changing weight of input layer to Hidden layer
//
//                            for (Node inputNode : this.inputLayer) {
//                                double valueForChangingWeight = this.learningRate * diff * inputNode.weight;
//                                for (Node hiddenNode : this.hiddenLayer) {
//                                    valueForChangingWeight = valueForChangingWeight * hiddenNode.weight * (1 - hiddenNode.weight);
//                                    for (Connection connection : hiddenNode.incomingConnection) {
//                                        if (connection.leftNeuron.id == hiddenNode.id) {
//                                            connection.weight = connection.weight + valueForChangingWeight;
//                                        }
//                                    }
//                                }
//                            }
//                        }
//
//                    }
//                }
//            /*
//            For evaluating the model, you average the predictions accuracy, not the weight matrix.
//            For K times training, you will get K models, you find the accuracy for each model on
//            it's corresponding test data and average over all model.
//
//             */
//            }
//            // Testing the fold.
//            List<Instance> toBeTested = finalFold.get(i);
//            for (Instance instance : toBeTested) {
//                transformInstanceIntoInputLayer(instance);
//                for (Node node : this.hiddenLayer) {
//                    double net = 0d;
//                    for (Connection connection : node.incomingConnection) {
//                        net += connection.weight * connection.leftNeuron.weight;
//                    }
//                    node.weight = Helper.Sigmoid(net);
//                }
//                double net = 0;
//                for (Connection connection : outputLayer.incomingConnection) {
//                    net += connection.weight * connection.leftNeuron.weight;
//                }
//                outputLayer.weight = Helper.Sigmoid(net);
//                instance.fold = i;
//                instance.confidenceOfPrediction = outputLayer.weight;
//                if(outputLayer.weight < 0.5){
//                    instance.predictedOutput = Driver.allHeaders.get(instance.ft_names[instance.ft_names.length - 1]).get(0);
//                } else {
//                    instance.predictedOutput = Driver.allHeaders.get(instance.ft_names[instance.ft_names.length - 1]).get(1);
//                }
//
//            }
//            for(Integer key : finalFold.keySet()){
//                Collections.shuffle(finalFold.get(key));
//            }
//        }
//    }
//
//    private void transformInstanceIntoInputLayer(Instance instance){
//        for(int i=0;i<this.inputLayer.length;i++){
//            this.inputLayer[i].weight = Double.valueOf(instance.ft_values[i]);
//        }
//    }
//}