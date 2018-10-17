import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class Driver {

    private static ArffFileReader arffFileReader  = new ArffFileReader();
    private static NFoldStratifiedHelper nFoldStratifiedHelper = new NFoldStratifiedHelper();

    public static void main(String[] args) {

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

        double totalAccuracy = 0d;
        for (int i = 0; i < numFolds; i++) {

            List<InstanceEntry> testDataSet = nFoldedInstanceEntries.get(i); // Test Data
            List<InstanceEntry> trainingData = new ArrayList<>();

            ANN ann = new ANN(testDataSet.get(0).getFeatureValues().length);

            for (int j = 0; j < numFolds; j++) {
                if (j != i) { // Training data
                    ann.trainAnn(nFoldedInstanceEntries.get(j), learningRate, numEpochs);
//                    trainingData.addAll(nFoldedInstanceEntries.get(j));
                }
            }

            int currentCorrectClassifications = 0;
            for (InstanceEntry testEntry : testDataSet) {
                ann.evaluate(testEntry);
                if (testEntry.getPredictedClassLabel().equalsIgnoreCase(testEntry.getClassLabel())) {
                    currentCorrectClassifications++;
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
                stringBuilder.append(String.format("%.6f", testEntry.getPredictionConfidence()));
                stringBuilder.append("\n");
                resultsForPrinting[originalIndex] = stringBuilder.toString();
            }

            for(Integer key : nFoldedInstanceEntries.keySet()){
                Collections.shuffle(nFoldedInstanceEntries.get(key));
            }

            double currentAccuracy = (double) currentCorrectClassifications / testDataSet.size();
            totalAccuracy += currentAccuracy;
        }

        /* ****************************************************************** */
        /* ****************************************************************** */

        for (String str : resultsForPrinting) {
            System.out.print(str);
        }

        /* ****************************************************************** */
        /* ****************************************************************** */

        System.out.println("\n");
//        double accuracy = (double) correctClassifications / allTrainingDataSize;
        double accuracy = totalAccuracy / numFolds;
        System.out.println("Accuracy: " + accuracy);
    }
}
