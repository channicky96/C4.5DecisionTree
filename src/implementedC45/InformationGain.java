package implementedC45;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

public class InformationGain {

    // Variables for an Information Gain object
    double[] classArr;
    Instances train;

    // Blank constructor
    public InformationGain() {

    }

    // Generic constructor
    public InformationGain(Instances train, double[] classVec) {
        this.train = train;
        this.classArr = classVec;
    }

    /**
     *
     * @param train train data passed in to compute the entropy of the data
     * @return The entropy of the data set
     */
    public double getEntropy(Instances train) {

        int count = 0;
        double setEntropy = 0;
        double[] classArray = new double[train.numInstances()];

        // Obtain all unique class values
        double[] temp = train.attributeToDoubleArray(train.classIndex());
        Set<Double> mySet = new HashSet<>();
        for (double classVal : temp) {
            mySet.add(classVal);
        }

        double[] classVec = new double[mySet.size()];
        int k = 0;
        for (Double classVal : mySet) {
            classVec[k++] = classVal;
        }

        int[] distribution = new int[classVec.length];
        for (int i = 0; i < train.numInstances(); i++) {
            classArray[i] = train.get(i).classValue();
        }

        // Loop through each class and count the number of occurences
        for (double uniqueClassVal : classVec) {
            for (double classVal : classArray) {
                if (uniqueClassVal == classVal) {
                    distribution[count]++;
                }
            }
            count++;
        }

        // Calculate the entropy of the data set based on the number of occurences
        for (int dist : distribution) {
            setEntropy += (-((double) ((double) dist / (double) classArray.length)) * (Math.log(((double) dist / (double) classArray.length)) / Math.log(2)));
        }
        return setEntropy;
    }

    /**
     *
     * @param attCol column of the data to calculate the splitting threshold on
     * @param setEntropy entropy of the data set
     * @return the node with the highest IG
     */
    TreeNode getThreshold(int attCol, double setEntropy) {
        int splitCol = 0;
        double thisGain, thisThres, thisAtt;
        double bestGain = 0, bestThreshold = 0, infoGain = 0;

        ArrayList<Double> thresholdTable = new ArrayList<>();
        HashMap<Double, Double> threshGain = new HashMap<>();
        ArrayList<Double> classList = new ArrayList<>();

        // Obtain all unique class values
        double[] temp = train.attributeToDoubleArray(train.classIndex());
        Set<Double> mySet = new HashSet<>();
        for (double classVal : temp) {
            mySet.add(classVal);
        }

        for (Double classVal : mySet) {
            classList.add(classVal);
        }

        // Return node with highest IG from a column
        for (int j = 0; j < this.train.numInstances(); j++) {
            thisAtt = train.get(j).value(attCol);
            if (!thresholdTable.contains(thisAtt)) { // Create an array with unique values from the data
                thresholdTable.add(thisAtt);
            }
        }

        // For each unique data value, calculate the information gain at each value
        for (double value : thresholdTable) {
            infoGain = getInfoGain(value, classList, attCol, setEntropy);
            threshGain.put(value, infoGain);
        }
        
        // For each entry in the HashMap, find the entry, hence column with the highest IG
        for (Entry<Double, Double> map : threshGain.entrySet()) {
            thisGain = map.getValue();
            thisThres = map.getKey();

            if (thisGain > bestGain) {
                bestGain = thisGain;
                bestThreshold = thisThres;
                splitCol = attCol;
            }
        }

        // Create and return node with highest IG value
        TreeNode node = new TreeNode(train.attribute(attCol).name(), splitCol, null, null, null, bestThreshold, bestGain);
        return node;
    }

    /**
     *
     * @param threshVal threshold value for splitting continuous
     * @param classList list holding unique list of the class attribute
     * @param attCol attribute column to compute the info gain for
     * @param entropy entropy of the dataset
     * @return the information gain of the column
     */
    public double getInfoGain(double threshVal, ArrayList<Double> classList, int attCol, double entropy) {
        double infoGain, gainRatio = 0, splitInfo = 0;
        double numGreater = 0, numLess = 0, entropyLower = 0, entropyUpper = 0;
        ArrayList<Instance> lowerArray = new ArrayList<>();
        ArrayList<Instance> upperArray = new ArrayList<>();

        // Partition the data
        for (int i = 0; i < train.numInstances(); i++) {
            if (train.get(i).value(attCol) <= threshVal) {
                lowerArray.add(train.get(i));
            } else {
                upperArray.add(train.get(i));
            }
        }

        // Calculate entropy for the part of the data lower than threshold
        // based on the number of occurences
        for (int i = 0; i < classList.size(); i++) {
            double currentVal = classList.get(i);
            double numOccurences = 0;

            // Loop through each class and count the number of occurences
            for (int j = 0; j < lowerArray.size(); j++) {
                Instance thisIntsance = lowerArray.get(j);
                if (thisIntsance.classValue() == currentVal) {
                    numOccurences++;
                }
            }
            numLess = lowerArray.size();
            if (numOccurences != 0) {
                entropyLower += -(numOccurences / numLess) * (Math.log((numOccurences / numLess) / Math.log(2)));
            } else {
                // entropy = 0 if there are no occurence 
                entropyLower += 0;
            }
        }

        // Calculate entropy for the part of the data higher than threshold
        // based on the number of occurences          //
        for (int i = 0; i < classList.size(); i++) {
            double currentVal = classList.get(i);
            double numOccurences = 0;

            for (int j = 0; j < upperArray.size(); j++) {
                Instance thisIntsance = upperArray.get(j);
                if (thisIntsance.classValue() == currentVal) {
                    numOccurences++;
                }
            }
            numGreater = upperArray.size();
            if (numOccurences != 0) {
                entropyUpper += -(numOccurences / numGreater) * (Math.log((numOccurences / numGreater) / Math.log(2)));
            } else {
                // entropy = 0 if there are no occurence 
                entropyUpper += 0;
            }
        }

        // Calculate the information gain for the given threshold value
        infoGain = entropy - (entropyLower * numLess) / (train.numInstances()) - ((entropyUpper * numGreater) / (train.numInstances()));

        // Calculate the information gain ratio by getting the distribution
        // of each unique value in the data set
//        double[] attArray = train.attributeToDoubleArray(attCol);
//        Set<Double> uniqueSet = new HashSet<>();
//
//        for (double d : attArray) {
//            uniqueSet.add(d);
//        }
//
//        ArrayList<Double> distribution = new ArrayList<>();
//
//        for (double uniqueVal : uniqueSet) {
//            double count = 0;
//            for (int i = 0; i < train.numInstances(); i++) {
//                if (train.get(i).value(attCol) == uniqueVal) {
//                    count++;
//                }
//            }
//            distribution.add(count);
//        }
//        // Compute the intrinsic information and the gain ratio
//        for (Double count : distribution) {
//            splitInfo -= (count / train.numInstances()) * (Math.log(count / train.numInstances()) / Math.log(2));
//        }
//        gainRatio = infoGain / splitInfo;
        return infoGain;
    }
}
