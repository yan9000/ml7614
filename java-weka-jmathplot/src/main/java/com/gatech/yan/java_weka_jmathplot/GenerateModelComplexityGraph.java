package com.gatech.yan.java_weka_jmathplot;


import java.awt.Color;
import java.util.Arrays;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

public class GenerateModelComplexityGraph {
    
	
	static int diabetesDataSetSize = 768;
	

	
	public static void main(String[] args) {
        try {
      
        	// knn varying k
        	DiabetesReader diabetes = new DiabetesReader();
            
        	for (int k=1; k<11; k++) {        		
        		IBk knn = new IBk(k);
            	diabetes.doModelComplexity(knn);
        	}
        	
        	double[] x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}; 
    		double[] y = diabetes.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] y2 = diabetes.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(x, y, x, y2, "KNN - vary K - ModelComplexity", "K");
        
    		// knn with weighting    		
    		DiabetesReader diabetes2 = new DiabetesReader();
            for (int k=1; k<11; k++) {        		
            	IBk kyy = new IBk();
            	String[] options = new String[]{"-K", Integer.toString(k), "-I"};        		
            	kyy.setOptions(options);
                diabetes2.doModelComplexity(kyy);
        	}
    		 
            double[] xk = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}; 
    		double[] yy = diabetes2.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] yy2 = diabetes2.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xk, yy, xk, yy2, "KNN - vary K  /w weighting - ModelComplexity", "K");


    		// decision tree    		
    		DiabetesReader diabetesDt1 = new DiabetesReader();
    		for (int k=1; k<9; k++) {        		
    			J48 j48 = new J48();
    			double confidence = k * 0.1;    			
            	String[] options = new String[]{"-C", Double.toString(confidence), "-M", "2"};
        		j48.setOptions(options);
        		diabetesDt1.doModelComplexity(j48);
        	}
    		
    		double[] xdt = {.1,.2,.3,.4,.5,.6,.7,.8}; 
    		double[] ydt = diabetesDt1.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] ydt2 = diabetesDt1.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xdt, ydt, xdt, ydt2, "tree - vary confidence - ModelComplexity", "Confidence");
    		
    	
    		DiabetesReader diabetesDt2 = new DiabetesReader();
    		for (int k=1; k<8; k++) {        		
    			J48 j48 = new J48();    			    			
            	String[] options = new String[]{"-C", "0.25", "-M", Integer.toString(k)};
        		j48.setOptions(options);
        		diabetesDt2.doModelComplexity(j48);
        	}
    		
    		double[] xdtq = {1,2,3,4,5,6,7}; 
    		double[] ydtq = diabetesDt2.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] ydt2q = diabetesDt2.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xdtq, ydtq, xdtq, ydt2q, "tree - vary minNumObjects - ModelComplexity", "minNumObjects");
    		
    		// Boosted Decision Tree
        	
        	DiabetesReader diabetesBoost = new DiabetesReader();
    		for (int k=1; k<11; k++) {        		
    			AdaBoostM1 ada = new AdaBoostM1();
    			J48 j48 = new J48();
            	String[] options = new String[]{"-C", "0.25", "-M", "2"};
        		j48.setOptions(options);        		
        		ada.setClassifier(j48);
        		ada.setNumIterations(k);
        		diabetesBoost.doModelComplexity(ada);;
        	}
    		
    		double[] xboost = {1,2,3,4,5,6,7,8,9,10}; 
    		double[] yboost = diabetesBoost.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] yboost2 = diabetesBoost.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xboost, yboost, xboost, yboost2, "boost - vary numIterations - ModelComplexity", "numIterations");
    		
    		DiabetesReader diabetesBoost2 = new DiabetesReader();
    		for (int k=1; k<11; k++) {        		
    			AdaBoostM1 ada = new AdaBoostM1();
    			J48 j48 = new J48();
            	String[] options = new String[]{"-C", "0.25", "-M", "2"};
        		j48.setOptions(options);        		
        		ada.setClassifier(j48);
        		ada.setNumIterations(5);
        		ada.setWeightThreshold(k*10);
        		diabetesBoost2.doModelComplexity(ada);;
        	}
    		
    		double[] xboostq = {1,2,3,4,5,6,7,8,9,10}; 
    		double[] yboostq = diabetesBoost2.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] yboost2q = diabetesBoost2.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xboostq, yboostq, xboostq, yboost2q, "boost - vary weight threshold - ModelComplexity", "weight threshold");
    		
    		
    		DiabetesReader diabetesNeural = new DiabetesReader();
    		for (int k=1; k<8; k++) {        		
    			MultilayerPerceptron neural = new MultilayerPerceptron();
    			String learningRate = Double.toString(.1 * k);
                String[] options = new String[]{
                        "-L", learningRate,  // Learning rate
                        "-M", "0.2",  // Momentum          
                        "-N", "500"  // Number of epochs                     
                    };
                neural.setOptions(options);               
                
        		diabetesNeural.doModelComplexity(neural);
        	}
    		
    		double[] xNeural = {1,2,3,4,5,6,7}; 
    		double[] yNeural = diabetesNeural.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] yNeural2 = diabetesNeural.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xNeural, yNeural, xNeural, yNeural2, "Neural - vary learning rate - ModelComplexity", "learning rate");
    		
    		
    		DiabetesReader diabetesNeuralq = new DiabetesReader();
    		for (int k=1; k<8; k++) {        		
    			MultilayerPerceptron neural = new MultilayerPerceptron();
    			String momentum = Double.toString(.1 * k);
                String[] options = new String[]{
                        "-L", "0.3",  // Learning rate
                        "-M", momentum,  // Momentum          
                        "-N", "500"  // Number of epochs                     
                    };
                neural.setOptions(options);
                diabetesNeuralq.doModelComplexity(neural);
        	}
    		
    		double[] xNeuralq = {1,2,3,4,5,6,7}; 
    		double[] yNeuralq = diabetesNeuralq.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] yNeural2q = diabetesNeuralq.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xNeuralq, yNeuralq, xNeuralq, yNeural2q, "Neural - vary momentum - ModelComplexity", "momentum");
        	
        	DiabetesReader diabetesSupportVector = new DiabetesReader();
    		for (int k=1; k<8; k++) {        		
    			SMO smo = new SMO();           
                String[] options = new String[]{
                    "-C", Integer.toString(k*10),  // Set the complexity constant
                    "-L", "0.001",  // Set the tolerance parameter
                    "-P", "1.0E-12",  // Set the epsilon for round-off error
                    "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 3"  // Set the polynomial kernel with degree 3
                };
                smo.setOptions(options);
                diabetesSupportVector.doModelComplexity(smo);
        	}
    		
    		double[] xSup = {1,2,3,4,5,6,7}; 
    		double[] ySup = diabetesSupportVector.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] ySup2 = diabetesSupportVector.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xSup, ySup, xSup, ySup2, "SVM - vary complexity - ModelComplexity", "complexity const");
    		
    		DiabetesReader diabetesSupportVectorq = new DiabetesReader();
    		for (int k=1; k<8; k++) {        		
    			SMO smo = new SMO();           
                String[] options = new String[]{
                    "-C", "1.0",  // Set the complexity constant
                    "-L", Double.toString(k*0.001),  // Set the tolerance parameter
                    "-P", "1.0E-12",  // Set the epsilon for round-off error
                    "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 3"  // Set the polynomial kernel with degree 3
                };
                smo.setOptions(options);
                diabetesSupportVectorq.doModelComplexity(smo);
        	}
    		
    		double[] xSupq = {1,2,3,4,5,6,7}; 
    		double[] ySupq = diabetesSupportVectorq.mcTrainingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();   		
    		double[] ySup2q = diabetesSupportVectorq.mcTestingPoints.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
    		GenerateModelComplexityGraph.plot(xSupq, ySupq, xSupq, ySup2q, "SVM - vary tolerance - ModelComplexity", "tolerance");
   	
        } catch (Exception e) {
            e.printStackTrace();
        }   
	
	
	}	
	
	public static void plot(double[] x, double[] y, double[] x2, double[] y2, String title, String xLabel) {
		
		Plot2DPanel plot = new Plot2DPanel();

		plot.addLinePlot("Training Line", Color.BLUE, x, y);
		plot.addLinePlot("Validation Line", Color.GREEN, x2, y2);
		plot.setAxisLabels(xLabel, "error rate %");
		plot.addLegend("SOUTH");

		JFrame frame = new JFrame(title);
		frame.setContentPane(plot);
		frame.setSize(400, 400);
		frame.setVisible(true);

		System.out.println("TrainingPoints:");
		System.out.println(Arrays.toString(x));
		System.out.println(Arrays.toString(y));

		System.out.println("TestingPoints:");
		System.out.println(Arrays.toString(x2));
		System.out.println(Arrays.toString(y2));
	}

}