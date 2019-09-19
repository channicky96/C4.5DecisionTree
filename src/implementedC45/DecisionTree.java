package implementedC45;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Collections;
import java.util.stream.Stream;
import static java.util.Map.Entry.comparingByValue;
import static java.util.stream.Collectors.toMap;

import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.WrapperSubsetEval;

public class DecisionTree {

    public static void main(String[] args) throws IOException, Exception {

        String path = "benchmarks";
        String[] datasets = {"banana", "breast-cancer", "diabetis", "german", "heart", "image",
            "ringnorm", "splice", "thyroid", "twonorm", "waveform"};

//        String[] datasets = {"thyroid"};

        ArrayList<Double> errorRateList = new ArrayList<>();
        ArrayList<Double> starndardErrorList = new ArrayList<>();

        // For each benchmark data set
        for (String dataset : datasets) {
            System.out.println("Processing dataset: " + dataset);
            int numOfFiles;
            double sumOfSE = 0.0, sumOfError = 0.0;

            try (Stream<Path> files = Files.list(Paths.get(path + "/" + dataset))) {
                numOfFiles = (int) files.count() / 2;
            }

            // For each train and test data file of the dataset
            for (int i = 1; i <= numOfFiles; i++) {

                // Load in training and testing data
                Instances train = loadDataARFF(path + "/" + dataset + "/" + dataset + "_train_data_" + i);
                Instances test = loadDataARFF(path + "/" + dataset + "/" + dataset + "_test_data_" + i);

                // ================== Feature selection methods ================
                // ====================== Wrapper methods ======================
                // Greedy Stepwise Feature Selection                            //
                // Second formal parameter option:                              //
                // true = Backward Elimination; false = Forward Selection       //
//                train = WrapperGreedyStepwise(train, true);                   //
                // ====================== Filter  methods ======================
//                train = FilterPearson(train);                                 //
//                train = FilterIG(train);                                      //
//                train = FilterGR(train);                                      //
                // =============================================================

                // If only the class attribute remains, break
                if (train.numAttributes() == 1) {
                    break;
                }

                // obtain a double array of unique classes
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

                // Calculate the entropy of the entire data set
                InformationGain IG = new InformationGain();
                double totalEntropy = IG.getEntropy(train);
                
                // Generate a C4.5 decision tree 
                TreeNode tree = buildTree(train, classVec, totalEntropy);

                // TestData object to store a single result
                TestData td = new TestData();
                // Stores all results of a single run
                ArrayList<TestData> testResult = td.testModel(tree, test);

                double correct = 0.0;
                for (TestData t : testResult) {
                    if (t.getPredicted().equals(t.getActual())) {
                        correct++;
                    }
                }

                // Compute sum of standard error for the benchmark dataset
                // i.e. sum of mean of the errors of 100 testing splits
                sumOfSE += getSE(train);

                // Compute sum of all test set error (error rates)
                sumOfError += (1 - (correct / test.numInstances()));
            }

            // Compute standard mean error for the benchmark dataset
            errorRateList.add(sumOfError / numOfFiles);

            // Compute test set error for the benchmark dataset            
            starndardErrorList.add(sumOfSE / numOfFiles);
        }
        System.out.println("\n================================");
        for (int i = 0; i < datasets.length; i++) {
            System.out.println("For dataset: " + datasets[i]
                    + "\nError rate: " + errorRateList.get(i)
                    + "\nStandard Error: " + starndardErrorList.get(i) + "\n");
        }
    }

    /**
     *
     * @param train train partitioned data
     * @param classVec vector containing classes of the train data
     * @param setEntropy the entropy of the data
     * @return A tree node representing the decision tree built
     */
    private static TreeNode buildTree(Instances train, double[] classVec, double setEntropy) {

        // If there are no data, return null
        if (train.isEmpty()) {
            return null;
        }

        TreeNode thisNode, leftChild, rightChild;
        // Default "best" node with highest information gain
        TreeNode best = new TreeNode("Best node", 0, null, null, null, 0, 0);

        int uniqueElements = 0, highestIGCol = 0;

        // Check if all instances in the data set is the same
        for (int i = 0; i < (train.numInstances() - 1); i++) {
            if (train.get(i).classValue() != train.get(i + 1).classValue()) {
                if (uniqueElements > 2) {
                    i = train.numInstances();
                } else {
                    uniqueElements++;
                }
            } else if (i == (train.numInstances() - 2) && uniqueElements == 0) {
                return new TreeNode(String.valueOf(train.get(0).classValue()), 0, null, null, null, -1, -1);
            } else if (i == (train.numInstances() - 2) && uniqueElements == 1) {
                TreeNode conNode = new TreeNode(train);
                best = conNode.continuousNode();
                return best;
            }
        }

        // Find the best node with the highest value of information gain
        for (int i = 0; i < train.numAttributes() - 1; i++) {
            train.sort(i);
            InformationGain IG = new InformationGain(train, classVec);
            thisNode = IG.getThreshold(i, setEntropy);
            if (thisNode.getInformationGain() > best.getInformationGain()) {
                best = thisNode;
            }
        }

        // Obtain the column index as it will be used to split on
        for (int i = 0; i < train.numAttributes() - 1; i++) {
            if (train.attribute(i).name().equals(best.getTitle())) {
                highestIGCol = i;
            }
        }

        // Find the thresgold for the binary split
        Double splitThresh = best.getThreshold();

        // 2 as we are doing this for left and right child
        for (int i = 0; i < 2; i++) {
            // Sort the data in accending order, used for spliting the data
            train.sort(highestIGCol);
            // Obtain the data used for creating the child node
            Instances t = getChild(train, highestIGCol, i, splitThresh);

            if (t.numInstances() <= 2 || train.lastInstance().index(highestIGCol) == splitThresh) {
                TreeNode conNode = new TreeNode(t);
                best = conNode.continuousNode();
                return best;
            }

            if (i == 0) {
                // Recursive call of method to build left side of the tree
                leftChild = buildTree(t, classVec, setEntropy);
                best.setLeftChild(leftChild);
            } else {
                // Recursive call of method to build right side of the tree
                rightChild = buildTree(t, classVec, setEntropy);
                best.setRightChild(rightChild);
            }
        }
        return best;
    }

    /**
     *
     * @param data Data passed in to be split to build the left child or the
     * right child node
     * @param splitColumn Column to split on
     * @param leftOrRight If we are building the left side or the right side of
     * the tree
     * @param splitThresh The threshold for splitting the data
     * @return An instances object used to build the left child or the right
     * child of the tree
     */
    private static Instances getChild(Instances data, int splitColumn, int leftOrRight, Double splitThresh) {
        double index;
        switch (leftOrRight) {
            case 0: {
                if (data.numInstances() <= 1 || data.lastInstance().value(splitColumn) == splitThresh) {
                    return data;
                }
                int counter = 0;
                boolean belowThresh = true;

                // loop through sorted data and count occurrence of attributes below the threshold
                while (belowThresh) {
                    if (counter == data.numInstances()) {
                        return null;
                    }
                    index = data.instance(counter).value(splitColumn);
                    if (index > splitThresh) {
                        belowThresh = false;
                    }
                    counter++;
                }
                // Partition the data lower than the threshold to make the left child
                Instances leftData = new Instances(data, 0, counter - 1);
                return leftData;
            }
            case 1: {
                int counter = 1;
                boolean belowThresh = true;

                // loop through sorted data and count occurrence of attributes below the threshold
                while (belowThresh) {
                    if (counter == data.numInstances()) {
                        return null;
                    }
                    index = data.instance(counter).value(splitColumn);
                    if (index > splitThresh) {
                        belowThresh = false;
                    }
                    counter++;
                }
                // Partition the data greater than the threshold to make the left child
                Instances rightData = new Instances(data, counter - 1, data.numInstances() - counter + 1);
                return rightData;
            }
            default:
                // Should never enter this part of the code
                System.out.println("Something went wrong");
                break;
        }
        return data;
    }

    /**
     *
     * @param data train partitioned data
     * @return the combined standard error of the data set provided
     */
    public static double getSE(Instances data) {

        double sumOfVar = 0.0;
        ArrayList<Double> meanArray = new ArrayList<>();
        ArrayList<Double> varArray = new ArrayList<>();

        // Compute mean for each attribute
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            meanArray.add(Arrays.stream(data.attributeToDoubleArray(i)).sum() / data.numInstances());
        }

        // Compute variance for each attribute
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            for (Double dataPt : data.attributeToDoubleArray(i)) {
                sumOfVar += Math.pow(dataPt - meanArray.get(i), 2);
            }
            varArray.add(sumOfVar / data.numInstances());
            sumOfVar = 0.0;
        }

        double meanOfAllAttributes = meanArray.stream().mapToDouble(Double::doubleValue).sum() / meanArray.size();

        // Compute total variance 
        for (int i = 0; i < meanArray.size(); i++) {
            sumOfVar += data.numInstances()
                    * (varArray.get(i) - Math.pow(meanArray.get(i) - meanOfAllAttributes, 2));
        }

        // Compute and return standard error of the mean
        int totalNumInst = (data.numInstances() * (data.numAttributes() - 1));
        double sd = sumOfVar / totalNumInst;
        return sd / Math.sqrt(totalNumInst);
    }

    /**
     *
     * @param train train partitioned data
     * @return optimal" feature subset obtained using the Pearson's correlation
     * coefficient
     * @throws Exception
     */
    public static Instances FilterPearson(Instances train) throws Exception {

        double[] X = train.attributeToDoubleArray(train.classIndex()), Y;
        int n = X.length;
        ArrayList<Integer> toRemove = new ArrayList<>();
        HashMap<Integer, Double> hmap = new HashMap<>();
        for (int j = 0; j < train.numAttributes() - 1; j++) {
            Y = train.attributeToDoubleArray(j);

            double sum_X = 0, sum_Y = 0, sum_XY = 0;
            double squareSum_X = 0, squareSum_Y = 0;

            for (int i = 0; i < n; i++) {
                // sum of elements of array X
                sum_X += X[i];

                // sum of elements of array Y
                sum_Y += Y[i];

                // sum of X[i] * Y[i]. 
                sum_XY += X[i] * Y[i];

                // sum of square of array elements
                squareSum_X += X[i] * X[i];
                squareSum_Y += Y[i] * Y[i];
            }

            // use formula for calculating correlation coefficient
            double corr = (float) (n * sum_XY - sum_X * sum_Y)
                    / (float) (Math.sqrt((n * squareSum_X
                            - sum_X * sum_X) * (n * squareSum_Y
                            - sum_Y * sum_Y)));

            // Take absolute value of the coefficient, as 0 is the least "useful"
            if (corr < 0) {
                corr = corr * -1;
            }
            hmap.put(j, corr);
        }

        // Sort map into accending order
        Map<Integer, Double> sorted = hmap
                .entrySet().stream().sorted(comparingByValue())
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

        // If value is less than 0.05 (threshold /cut off point), add it to a list
        for (Integer attCol : sorted.keySet()) {
            double value = sorted.get(attCol);
            if (value < 0.05) {
                toRemove.add(attCol);
            }
        }

        Collections.sort(toRemove, Collections.reverseOrder());
        // Delete attributes with Pearson's correlation coefficient less than 0.05
        // And return the data
        if (!toRemove.isEmpty()) {
            for (Integer colToRemove : toRemove) {
                train.deleteAttributeAt(colToRemove);
            }
        } else {
            return train;
        }
        return train;
    }

    /**
     *
     * @param train train partitioned data
     * @return "optimal" feature subset obtained using the information gain
     * @throws Exception
     */
    public static Instances FilterIG(Instances train) throws Exception {

        InformationGain info = new InformationGain();
        // Compute entropy of the whole data set
        double totalEntropy = info.getEntropy(train);
        TreeNode thisNode;
        HashMap<Integer, Double> colAndIG = new HashMap<>();

        // Array containing all unique class values
        double[] classVec = new double[train.classAttribute().numValues()];
        for (int j = 0; j < train.classAttribute().numValues(); j++) {
            classVec[j] = Double.valueOf(train.classAttribute().value(j));
        }

        // Obtain for each column their respective information gain 
        for (int i = 0; i < train.numAttributes() - 1; i++) {
            InformationGain IG = new InformationGain(train, classVec);
            thisNode = IG.getThreshold(i, totalEntropy);
            colAndIG.put(i, thisNode.getInformationGain());
        }

        Map<Integer, Double> sorted = colAndIG
                .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

        // Keep 10% of the original amount of features
        int toKeep = (int) (train.numAttributes() - 1 * 0.1);
        if (toKeep == 0) {
            toKeep = 1;
        }

        for (int i = sorted.size() - 1; i > toKeep; i--) {
            train.deleteAttributeAt(i);
        }
        return train;
    }

    /**
     *
     * @param train train partitioned data
     * @return "optimal" feature subset obtained using the information gain
     * ratio
     * @throws Exception
     */
    public static Instances FilterGR(Instances train) throws Exception {

        TreeNode thisNode;
        // Compute entropy of the whole data set
        InformationGain info = new InformationGain();
        double totalEntropy = info.getEntropy(train);

        HashMap<Integer, Double> colAndIG = new HashMap<>();

        // Array containing all unique class values
        double[] classVec = new double[train.classAttribute().numValues()];
        for (int j = 0; j < train.classAttribute().numValues(); j++) {
            classVec[j] = Double.valueOf(train.classAttribute().value(j));
        }

        // Obtain for each column their respective information gain ratio
        for (int i = 0; i < train.numAttributes() - 1; i++) {
            InformationGain IG = new InformationGain(train, classVec);
            thisNode = IG.getThreshold(i, totalEntropy);

            double[] attArray = train.attributeToDoubleArray(i);
            Set<Double> uniqueSet = new HashSet<>();

            // Obtaining unique data attribute values
            for (double d : attArray) {
                uniqueSet.add(d);
            }

            // And the distribution of those values
            ArrayList<Double> distribution = new ArrayList<>();
            double splitInfo = 0.0;
            for (double uniqueVal : uniqueSet) {
                double count = 0;
                for (int j = 0; j < train.numInstances(); j++) {
                    if (train.get(j).value(i) == uniqueVal) {
                        count++;
                    }
                }
                distribution.add(count);
            }

            // Computing the intrinsic information
            for (Double count : distribution) {
                splitInfo -= (count / train.numInstances()) * (Math.log(count / train.numInstances()) / Math.log(2));
            }
            colAndIG.put(i, thisNode.getInformationGain() / splitInfo);
        }

        Map<Integer, Double> sorted = colAndIG
                .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

        // Keep 10% of the original amount of features
        int toKeep = (int) (train.numAttributes() - 1 * 0.1);
        if (toKeep == 0) {
            toKeep = 1;
        }

        for (int i = sorted.size(); i > toKeep; i--) {
            train.deleteAttributeAt(i);
        }
        return train;
    }

    /**
     *
     * @param data train partitioned data
     * @param backward if set to true set search direction to backward instead
     * of forward
     * @return "optimal" feature subset obtained with greedy step wise method
     * and wrapper approach
     * @throws Exception
     */
    public static Instances WrapperGreedyStepwise(Instances data, boolean backward) throws Exception {

        AttributeSelection attSelect = new AttributeSelection();
        WrapperSubsetEval eval = new WrapperSubsetEval();
        J48 j48 = new J48();
        GreedyStepwise gs = new GreedyStepwise();
        gs.setSearchBackwards(backward);
        gs.setGenerateRanking(true);
        eval.setClassifier(j48);
        eval.setFolds(5);
        eval.setThreshold(0.01);
        eval.buildEvaluator(data);

        attSelect.setEvaluator(eval);
        attSelect.setSearch(gs);
        attSelect.SelectAttributes(data);
        return attSelect.reduceDimensionality(data);
    }

    /**
     *
     * @param fullPath directory path of data
     * @return an Instances object of data given
     * @throws Exception
     */
    public static Instances loadDataARFF(String fullPath) throws Exception {
        Instances output = new Instances(new FileReader(fullPath + ".arff"));
        output.setClassIndex(output.numAttributes() - 1);
        return output;
    }
}
