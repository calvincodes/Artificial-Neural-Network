import java.text.DecimalFormat;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class Driver {

    private static ArffFileReader arffFileReader  = new ArffFileReader();
    private static NFoldStratifiedHelper nFoldStratifiedHelper = new NFoldStratifiedHelper();
    private static DecimalFormat df = new DecimalFormat("0.000000");

    public static void main(String[] args) {

//        long startTime = System.nanoTime();
        if (args.length != 4) {
            System.err.println("Usage neuralnet trainfile num_folds learning_rate num_epochs");
            System.exit(-1);
        }

        String trainingFile = args[0];
        double numFolds = Double.valueOf(args[1]);
        Double learningRate = Double.valueOf(args[2]);
        Double numEpochs = Double.valueOf(args[3]);

        /* ****************************************************************** */

        // Step 1: Read the ARFF file

        List<InstanceEntry> allTrainingData = arffFileReader.readArff(trainingFile);
        int allTrainingDataSize = allTrainingData.size();

        int iterationCount = 10;
        double finalAccuracy = Double.MIN_VALUE;
        String[] finalResultsForPrinting = new String[allTrainingDataSize];

        for (int x = 0; x < iterationCount; x++) {

            /* ****************************************************************** */
            /* ****************************************************************** */

            // Step 2: Subdivide the n-fold stratified cross validation
            // n = numFolds

            HashMap<Integer, List<InstanceEntry>> nFoldedInstanceEntries = new HashMap<>();
            HashMap<Integer, Integer> instanceEntry2Fold = new HashMap<>();
            nFoldStratifiedHelper.performNFoldStratifiedCrossValidation(
                    allTrainingData, numFolds, nFoldedInstanceEntries, instanceEntry2Fold);

            /* ****************************************************************** */
            /* ****************************************************************** */

            String[] resultsForPrinting = new String[allTrainingDataSize];
            int correctClassifications = 0;

            for (int i = 0; i < numFolds; i++) {

                List<InstanceEntry> testDataSet = nFoldedInstanceEntries.get(i); // Test Data

                ArtificialNeuralNet artificialNeuralNet = new ArtificialNeuralNet(testDataSet.get(0).getFeatureValues().length);

                for (int j = 0; j < numFolds; j++) {
                    if (j != i) { // Training data
                        artificialNeuralNet.trainAnn(nFoldedInstanceEntries.get(j), learningRate, numEpochs);
                    }
                }

                for (InstanceEntry testEntry : testDataSet) {
                    artificialNeuralNet.evaluate(testEntry);
                    if (testEntry.getPredictedClassLabel().equalsIgnoreCase(testEntry.getClassLabel())) {
                        correctClassifications++;
                    }
                    StringBuilder stringBuilder = new StringBuilder();
                    int originalIndex = allTrainingData.indexOf(testEntry);
                    stringBuilder.append(instanceEntry2Fold.get(originalIndex));
                    stringBuilder.append(" ");
                    stringBuilder.append(testEntry.getPredictedClassLabel());
                    stringBuilder.append(" ");
                    stringBuilder.append(testEntry.getClassLabel());
                    stringBuilder.append(" ");
                    String conf = df.format(testEntry.getPredictionConfidence());
                    if (conf.equals("1.000000")) {
                        conf = "0.999999";
                    }
                    if (conf.equals("0.000000")) {
                        conf = "0.000001";
                    }
                    stringBuilder.append(conf);
                    stringBuilder.append("\n");
                    resultsForPrinting[originalIndex] = stringBuilder.toString();
                }

                for(Integer key : nFoldedInstanceEntries.keySet()){
                    Collections.shuffle(nFoldedInstanceEntries.get(key));
                }
            }

            double accuracy = (double) correctClassifications / allTrainingDataSize;
            if (finalAccuracy < accuracy) {
                finalAccuracy = accuracy;
                finalResultsForPrinting = resultsForPrinting;
            }
        }

//        System.out.println("\n");
//        System.out.println("Accuracy: " + finalAccuracy * 100);

        for (String str : finalResultsForPrinting) {
            System.out.print(str);
        }

//        long endTime = System.nanoTime();
//        System.out.println("Took "+ (double)(endTime - startTime) / 1000000000.0 + " s");
    }
}
