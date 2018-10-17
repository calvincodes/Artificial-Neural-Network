import java.util.Arrays;

public class InstanceEntry {

    private Double[] featureValues;
    private String classLabel;
    private String[] allClassLabels;
    private String predictedClassLabel;
    private double predictionConfidence;

    public Double[] getFeatureValues() {
        return featureValues;
    }

    public void setFeatureValues(Double[] featureValues) {
        this.featureValues = featureValues;
    }

    public String getClassLabel() {
        return classLabel;
    }

    public void setClassLabel(String classLabel) {
        this.classLabel = classLabel;
    }

    public String[] getAllClassLabels() {
        return allClassLabels;
    }

    public void setAllClassLabels(String[] allClassLabels) {
        this.allClassLabels = allClassLabels;
    }

    public String getPredictedClassLabel() {
        return predictedClassLabel;
    }

    public void setPredictedClassLabel(String predictedClassLabel) {
        this.predictedClassLabel = predictedClassLabel;
    }

    public double getPredictionConfidence() {
        return predictionConfidence;
    }

    public void setPredictionConfidence(double predictionConfidence) {
        this.predictionConfidence = predictionConfidence;
    }

    @Override
    public String toString() {
        return "InstanceEntry{" +
                "featureValues=" + Arrays.toString(featureValues) +
                ", classLabel='" + classLabel + '\'' +
                ", allClassLabels=" + Arrays.toString(allClassLabels) +
                ", predictedClassLabel='" + predictedClassLabel + '\'' +
                ", predictionConfidence=" + predictionConfidence +
                '}';
    }
}
