package implementedC45;

import java.util.ArrayList;
import weka.core.Instances;

public class TestData {

    // Variables of a result of a particular test instance
    private boolean result;
    private String predicted;
    private String actual;

    // Constructor used to test the data
    public TestData(boolean result, String predicted, String actual) {
        this.result = result;
        this.predicted = predicted;
        this.actual = actual;
    }

    // Blank constructor
    public TestData() {
    
    }

    /**
     *
     * @param tree the model to be trained on tested on
     * @param test the instance to be predicted
     * @return An array list containing all the results
     */
    public ArrayList<TestData> testModel(TreeNode tree, Instances test) {
        ArrayList<String> atts = new ArrayList<>();
        for (int i = 0; i < test.numAttributes() - 1; i++) {
            atts.add(test.attribute(i).toString().split(" ")[1]);
        }
        ArrayList<TestData> results = new ArrayList<>();
        int column;
        for (int i = 0; i < test.numInstances(); i++) {
            boolean finished = false;
            TreeNode thisNode = tree;
            String thisAtt, classPredict = null;
            double thisThresh;
            while (!finished) {
                thisAtt = thisNode.getTitle();
                thisThresh = thisNode.getThreshold();
                column = atts.indexOf(thisAtt);
                // Case for single nodes
                if (thisNode.getThreshold() == -1.0) {
                    classPredict = thisAtt;
                    finished = true;
                    // Case for Continuous nodes
                } else if (thisAtt.equals("Continuous")) {
                    // Check probability and assign class
                    if (thisNode.getProb1() > thisNode.getProb2()) {
                        classPredict = thisNode.getValOfClass1();
                        finished = true;
                    } else {
                        classPredict = thisNode.getValOfClass2();
                        finished = true;
                    }
                } else {
                    // Check current value against threshold at current node
                    if (test.get(i).value(column) <= thisThresh) {
                        // If the current node has no left child, 
                        // and is lower than threshold, assign current class
                        if (thisNode.getLeftChild() == null) {
                            classPredict = String.valueOf(thisNode.getClassVal());
                            finished = true;
                        } else {
                            // Traversing to left side of the tree
                            thisNode = thisNode.getLeftChild();
                        }
                    } else {
                        if (thisNode.getRightChild() == null) {
                            // If the current node has no right child, assign current class
                            classPredict = String.valueOf(thisNode.getClassVal());
                            finished = true;
                        } else {
                            // Traversing to right side of the tree
                            thisNode = thisNode.getRightChild();
                        }
                    }
                }
            }

            double classActual = test.get(i).classValue();
            boolean result = (classActual == Double.valueOf(classPredict));
            TestData currentResult = new TestData(result, classPredict, String.valueOf(classActual));
            results.add(currentResult);
        }
        return results;
    }

    // Generic accessors and mutators
    public boolean getResult() {
        return result;
    }

    public void setResultG(boolean result) {
        this.result = result;
    }

    public String getPredicted() {
        return predicted;
    }

    public void setPredicted(String predicted) {
        this.predicted = predicted;
    }

    public String getActual() {
        return actual;
    }

    public void setActual(String actual) {
        this.actual = actual;
    }

}
