import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ArffFileReader {

    public List<InstanceEntry> readArff(String fileName) {

        FileReader file = null;
        try {
            file = new FileReader(fileName);
        } catch (FileNotFoundException ex) {
            System.out.println("Unable to open file '" + fileName + "'" + Arrays.toString(ex.getStackTrace()));
            System.exit(-1);
        }

        List<InstanceEntry> arffEntries = new ArrayList<>();

        String line;
        BufferedReader bufferedReader = new BufferedReader(file);

        try {

            String[] allClassLabels = new String[2];
            while((line = bufferedReader.readLine()) != null) {

                if (line.startsWith("@attribute")) {

                    String[] header = line.split("'");
                    if (header[1].equalsIgnoreCase("class")) {
                        header[2] = header[2].replace('{', ' ');
                        header[2] = header[2].replace('}', ' ');
                        header[2] = header[2].replaceAll("\\s+", "");
                        allClassLabels = header[2].split(",");
                    }
                } else if (line.startsWith("@")) {

                    continue;
                } else {

                    List<String> featureValues = Arrays.asList(line.split(","));
                    InstanceEntry instanceEntry = new InstanceEntry();
                    int featureLen = featureValues.size();
                    instanceEntry.setClassLabel(featureValues.get(featureLen - 1)); // Extract class label.
                    // Trim class label and set feature values
                    instanceEntry.setFeatureValues(
                            featureValues.subList(0, featureLen - 1)
                                    .stream()
                                    .map(Double::valueOf)
                                    .toArray(Double[]::new)
                            );
                    instanceEntry.setAllClassLabels(allClassLabels);
                    arffEntries.add(instanceEntry);
                }
            }
            bufferedReader.close();
        } catch(IOException ex) {
            System.out.println("Error reading file '" + fileName + "'" + Arrays.toString(ex.getStackTrace()));
            System.exit(-1);
        }

        return arffEntries;
    }
}
