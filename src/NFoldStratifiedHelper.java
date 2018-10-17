import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class NFoldStratifiedHelper {

    // Note that the input parameters to this method will get modified.
    public void performNFoldStratifiedCrossValidation(List<InstanceEntry> allTrainingData, double numFolds,
                                                      HashMap<Integer, List<InstanceEntry>> nFoldedInstanceEntries,
                                                      HashMap<Integer, Integer> instanceEntry2Fold) {

        int allTrainingDataSize = allTrainingData.size();

        ArrayList<InstanceEntry>[] posAndNegSegregated = new ArrayList[2];
        int negInstances = 0;
        int posInstances = 0;
        for (InstanceEntry instanceEntry : allTrainingData) {
            if (instanceEntry.getClassLabel().equalsIgnoreCase(instanceEntry.getAllClassLabels()[0])) {
                if (posAndNegSegregated[0] == null) {
                    posAndNegSegregated[0] = new ArrayList<>();
                }
                posAndNegSegregated[0].add(instanceEntry);
                negInstances++;
            } else {
                if (posAndNegSegregated[1] == null) {
                    posAndNegSegregated[1] = new ArrayList<>();
                }
                posAndNegSegregated[1].add(instanceEntry);
                posInstances++;
            }
        }

        List<InstanceEntry> shuffledNegTrainingData = new ArrayList<>(posAndNegSegregated[0]);
        List<InstanceEntry> shuffledPosTrainingData = new ArrayList<>(posAndNegSegregated[1]);
        Collections.shuffle(shuffledNegTrainingData);
        Collections.shuffle(shuffledPosTrainingData);

        List<InstanceEntry> shuffledTrainingData = new ArrayList<>(allTrainingData);
        Collections.shuffle(shuffledTrainingData);

        int entriesPerFold = (int) (allTrainingDataSize / numFolds);
        int negEntriesPerFold = (int) Math.floor(entriesPerFold * ((double) negInstances / allTrainingDataSize));
        int posEntriesPerFold = entriesPerFold - negEntriesPerFold;
        int currentFold = 0;
        int entryCountInCurrentFold = 0;
        int negEntriesCountInCurrentFold = 0;
        int posEntriesCountInCurrentFold = 0;
        List<InstanceEntry> entriesInCurrentFold = new ArrayList<>();

        int posIndex = 0;
        int negIndex = 0;
        for (int i = 1; i <= numFolds; i++) {

            do {
                InstanceEntry shuffledTrainingEntry = shuffledPosTrainingData.get(posIndex);
                int originalIndexOfEntry = allTrainingData.indexOf(shuffledTrainingEntry);
                entriesInCurrentFold.add(shuffledTrainingEntry);
                instanceEntry2Fold.put(originalIndexOfEntry, currentFold);
                posEntriesCountInCurrentFold++;
                entryCountInCurrentFold++;
                posIndex++;
            } while (posEntriesCountInCurrentFold != posEntriesPerFold);

            do {
                InstanceEntry shuffledTrainingEntry = shuffledNegTrainingData.get(negIndex);
                int originalIndexOfEntry = allTrainingData.indexOf(shuffledTrainingEntry);
                entriesInCurrentFold.add(shuffledTrainingEntry);
                instanceEntry2Fold.put(originalIndexOfEntry, currentFold);
                negEntriesCountInCurrentFold++;
                entryCountInCurrentFold++;
                negIndex++;
            } while (negEntriesCountInCurrentFold != negEntriesPerFold);

            if (entryCountInCurrentFold == entriesPerFold
                    && i != numFolds) {
                posEntriesCountInCurrentFold = 0;
                negEntriesCountInCurrentFold = 0;
                nFoldedInstanceEntries.put(currentFold, new ArrayList<>(entriesInCurrentFold));
                entriesInCurrentFold = new ArrayList<>();
                currentFold++;
                entryCountInCurrentFold = 0;
            }

            // Add all remaining entries to last fold
            if (i == numFolds) {
                do {
                    InstanceEntry shuffledTrainingEntry = shuffledPosTrainingData.get(posIndex);
                    int originalIndexOfEntry = allTrainingData.indexOf(shuffledTrainingEntry);
                    entriesInCurrentFold.add(shuffledTrainingEntry);
                    instanceEntry2Fold.put(originalIndexOfEntry, currentFold);
                    posIndex++;
                } while (posIndex < shuffledPosTrainingData.size());

                do {
                    InstanceEntry shuffledTrainingEntry = shuffledNegTrainingData.get(negIndex);
                    int originalIndexOfEntry = allTrainingData.indexOf(shuffledTrainingEntry);
                    entriesInCurrentFold.add(shuffledTrainingEntry);
                    instanceEntry2Fold.put(originalIndexOfEntry, currentFold);
                    negIndex++;
                } while (negIndex < shuffledNegTrainingData.size());

                nFoldedInstanceEntries.put(currentFold, new ArrayList<>(entriesInCurrentFold));
            }
        }
    }
}
