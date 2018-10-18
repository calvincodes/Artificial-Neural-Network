import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RoCPlot {

    public void plotRoc(List<InstanceEntry> testInstance) {

        StringBuilder xCoords = new StringBuilder("x = [ ");
        StringBuilder yCoords = new StringBuilder("y = [ ");

        testInstance.sort((o1, o2) -> {
            if (o1.getPredictionConfidence() < o2.getPredictionConfidence()) {
                return -1;
            } else if (o1.getPredictionConfidence() > o2.getPredictionConfidence()) {
                return 1;
            }
            return 0;
        });
        Map<String, List<InstanceEntry>> keyGroup = new HashMap<>();
        List<String> keySet = new ArrayList<>();
        keySet.add(0, testInstance.get(0).getAllClassLabels()[0]);
        keySet.add(1, testInstance.get(0).getAllClassLabels()[1]);
        for(InstanceEntry e : testInstance){
            keyGroup.putIfAbsent(e.getClassLabel(), new ArrayList<>());
            keyGroup.get(e.getClassLabel()).add(e);
        }

        int num_neg = 97;
        int num_pos = 111;
        int TP = 0, FP = 0;
        int last_TP = 0;
        double FPR = 0d;
        double TPR = 0d;

        for(int i = 1; i < testInstance.size(); i++){
            if(i>1
                    && (testInstance.get(i).getPredictionConfidence() != testInstance.get(i-1).getPredictionConfidence())
                    && (TP > last_TP)
//                    && testInstance.get(i).getClassLabel().equals(keySet.get(0))
            ){
                FPR = (double) FP/num_neg;
                TPR = (double)TP/num_pos;
                xCoords.append(String.valueOf(FPR)).append(", ");
                yCoords.append(String.valueOf(TPR)).append(", ");
//                System.out.println("series1.add("+TPR + ", " + FPR +");");
                last_TP = TP;
            }
            if(testInstance.get(i).getClassLabel().equals(keySet.get(1))
//                    && trainingInstance.get(i).getClassLabel().equals(keySet.get(1))
            )
                TP++;
            else // if(trainingInstance.get(i).getPredictedClassLabel().equals(keySet.get(1)))
                FP++;

        }
        FPR = (double)FP/num_neg;
        TPR = (double)TP/num_pos;
        xCoords.append(String.valueOf(FPR)).append("]");
        yCoords.append(String.valueOf(TPR)).append("]");
        System.out.println(xCoords);
        System.out.println(yCoords);
//        System.out.println("series1.add("+TPR + ", " + FPR+");");
    }
}
