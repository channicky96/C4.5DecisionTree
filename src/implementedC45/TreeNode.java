package implementedC45;

import weka.core.Instances;

public class TreeNode {

    // Variables of a tree (node)
    private String label;
    private double classVal;
    private TreeNode childR;
    private TreeNode childL;
    private double threshold;
    private double infoGain;
    private Instances threshData;
    private int numAttributes;

    // In case of continuous nodes
    private String value1;
    private String value2;
    private double prob1;
    private double prob2;

    // Generic node constructors
    public TreeNode(String label, int s, TreeNode p, TreeNode r, TreeNode l, double t, double gain) {
        this.label = label;
        this.childR = r;
        this.childL = l;
        this.threshold = t;
        this.infoGain = gain;

    }

    public TreeNode(Instances train) {
        this.threshData = train;
        this.numAttributes = train.numAttributes();
    }

    // Continuous node constructor
    public TreeNode(String title, String v1, String v2, double probability1, double probability2) {
        this.label = title;
        this.value1 = v1;
        this.value2 = v2;
        this.prob1 = probability1;
        this.prob2 = probability2;
    }

    // For continuous data attribute with the binary split
    public TreeNode continuousNode() {
        double v1 = threshData.get(0).classValue();
        double v2 = -100.0;
        boolean valFound = false;
        double total = threshData.numInstances();
        double count1 = 1;
        double count2 = 0;

        // Count the occurences for v1 and v2
        for (int i = 2; i < total; i++) {					
            if (threshData.get(i).classValue() != v1) {	
                v2 = threshData.get(i).classValue();
                valFound = true;
                count2++;									
            } else {
                count1++;
            }
        }

        // Create a Continuous node by dividing the occurences by the total
        if (valFound == true) { 
            double probability1 = count1 / total;
            double probability2 = count2 / total;
            return new TreeNode("Continuous", String.valueOf(v1), String.valueOf(v2), probability1, probability2);
        } else {
            return null;
        }
    }

    // Generic accessors and mutators
    public double getInformationGain() {
        return this.infoGain;
    }

    public String getTitle() {
        return this.label;
    }

    public double getClassVal() {
        return this.classVal;
    }

    public double getThreshold() {
        return this.threshold;
    }
    
    public Instances getThreshData() {
        return this.threshData;
    }

    public TreeNode getLeftChild() {
        return childL;
    }

    public TreeNode getRightChild() {
        return childR;
    }

    void setLeftChild(TreeNode leftChild) {
        this.childL = leftChild;
    }

    void setRightChild(TreeNode rightChild) {
        this.childR = rightChild;
    }

    public double getProb1() {
        return prob1;
    }

    public double getProb2() {
        return prob2;
    }

    public String getValOfClass1() {
        return value1;
    }

    public String getValOfClass2() {
        return value2;
    }

}
