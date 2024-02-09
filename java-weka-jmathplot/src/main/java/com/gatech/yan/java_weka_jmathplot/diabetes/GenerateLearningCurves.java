package com.gatech.yan.java_weka_jmathplot.diabetes;


import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

public class GenerateLearningCurves {
    

	public static void main(String[] args) {
        try {
        	
        	DiabetesReader diabetes = new DiabetesReader();
             
        	IBk knn = new IBk(3);
        	diabetes.doClassifier(knn, "KNN Learning Curve");
        	
        	J48 j48 = new J48();
        	String[] options = new String[]{"-C", "0.25", "-M", "2"};
    		j48.setOptions(options);
    		diabetes.doClassifier(j48, "Decision Tree J48");
    		
    		AdaBoostM1 ada = new AdaBoostM1();
    		ada.setClassifier(j48);
    		ada.setNumIterations(10);
    		diabetes.doClassifier(ada, "Decision Tree AdaBoost J48");
    		
            MultilayerPerceptron neural = new MultilayerPerceptron();
            options = new String[]{
                    "-L", "0.3",  // Learning rate
                    "-M", "0.2",  // Momentum          
                    "-N", "500"  // Number of epochs                     
                };
            neural.setOptions(options);
            diabetes.doClassifier(neural, "NN - MultilayerPerceptron");
            
            SMO smo = new SMO();           
            options = new String[]{
                "-C", "1.0",  // Set the complexity constant
                "-L", "0.001",  // Set the tolerance parameter
                "-P", "1.0E-12",  // Set the epsilon for round-off error
                "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 3"  // Set the polynomial kernel with degree 3
            };
            smo.setOptions(options);
            diabetes.doClassifier(smo, "Support Vector Machines");

            
            
            
    		
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

	
	
}



