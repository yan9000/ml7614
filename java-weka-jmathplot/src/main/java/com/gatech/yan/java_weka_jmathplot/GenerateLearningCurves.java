package com.gatech.yan.java_weka_jmathplot;


import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class GenerateLearningCurves {
    
	
	static int diabetesDataSetSize = 768;
	

	
	public static void main(String[] args) {
        try {
             
        	IBk knn = new IBk(3);
        	doClassifier(knn, "KNN Learning Curve", "# instances", "% correctly classified");
        	
        	J48 j48 = new J48();
        	String[] options = new String[]{"-C", "0.25", "-M", "2"};
    		j48.setOptions(options);
    		doClassifier(j48, "Decision Tree J48", "# instances", "% correctly classified");
    		
    		AdaBoostM1 ada = new AdaBoostM1();
    		ada.setClassifier(j48);
    		ada.setNumIterations(10);
    		doClassifier(ada, "Decision Tree AdaBoost J48", "# instances", "% correctly classified");
    		
            MultilayerPerceptron neural = new MultilayerPerceptron();
            options = new String[]{
                    "-L", "0.3",  // Learning rate
                    "-M", "0.2",  // Momentum          
                    "-N", "500"  // Number of epochs                     
                };
            neural.setOptions(options);
            doClassifier(neural, "NN - MultilayerPerceptron", "# instances", "% correctly classified");
            
            SMO smo = new SMO();           
            options = new String[]{
                "-C", "1.0",  // Set the complexity constant
                "-L", "0.001",  // Set the tolerance parameter
                "-P", "1.0E-12",  // Set the epsilon for round-off error
                "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 3"  // Set the polynomial kernel with degree 3
            };
            smo.setOptions(options);
            doClassifier(smo, "Support Vector Machines", "# instances", "% correctly classified");

            
            
            
    		
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

	private static void doClassifier(AbstractClassifier c, String title, String xLabel, String yLabel) throws Exception {
		
		System.out.println("");
		System.out.println("Classifier: " + c.getClass().getSimpleName()+ java.util.Arrays.toString(c.getOptions()));	
		List<Point> pointsTraining = getTrainingPoints(c);            
		List<Point> pointsTesting= getTestingPoints(c);			
		double[] x = pointsTraining.stream().map(p -> p.getX()).mapToDouble(Double::doubleValue).toArray();
		double[] y = pointsTraining.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();            
		double[] x2 = pointsTesting.stream().map(p -> p.getX()).mapToDouble(Double::doubleValue).toArray();
		double[] y2 = pointsTesting.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();            
		plotLearningCurve(x, y, x2, y2, title, xLabel, yLabel);
	}
	
	private static List<Point> getTestingPoints(AbstractClassifier c) throws Exception {
		
		List<Point> pointsTesting= new ArrayList<>();
		Instances data;			
		Instances trainDataSet;
        int trainSize;
        
        data = new DataSource("diabetes.arff").getDataSet();           
        data.setClassIndex(data.numAttributes()-1);
                    
        trainSize = (int) Math.round(data.numInstances() * 0.8);                        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));            
        
        trainSize = (int) Math.round(data.numInstances() * 0.7);                        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));        
        
        trainSize = (int) Math.round(data.numInstances() * 0.6);                        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));
        
        trainSize = (int) Math.round(data.numInstances() * 0.5);                        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));
        
        trainSize = (int) Math.round(data.numInstances() * 0.4);
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));
        
        trainSize = (int) Math.round(data.numInstances() * 0.3);
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));
        
        trainSize = (int) Math.round(data.numInstances() * 0.2);
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes()-1);
        pointsTesting.add(getTestingPointForDiabetes(trainDataSet, c));   
        
        return pointsTesting;		
	}

	private static  List<Point> getTrainingPoints(AbstractClassifier c) throws Exception {
		
		List<Point> pointsTraining= new ArrayList<>();
		
		Instances data;
		data = new DataSource("diabetes20.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes30.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes40.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes50.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes60.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes70.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes80.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		data = new DataSource("diabetes.arff").getDataSet();           
		data.setClassIndex(data.numAttributes()-1);
		pointsTraining.add( getTrainingPoint(data, c));
		
		return pointsTraining;
	}

	
	private static Point getTrainingPoint(Instances data, AbstractClassifier c) throws Exception {		
		c.buildClassifier(data);		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(c, data);		
//		System.out.println("Classifier: " + c.getClass().getSimpleName()+ java.util.Arrays.toString(c.getOptions()));		
//		System.out.println("Eval Instances: " + eval.numInstances());
//		System.out.println("Correct % : " + eval.pctCorrect());
		return (new Point(eval.numInstances(), eval.pctCorrect()));
	}
	
	private static Point getTestingPointForDiabetes(Instances data, AbstractClassifier c) throws Exception {		   
		c.buildClassifier(data);		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(c, data);		
//		System.out.println("Classifier: " + c.getClass().getSimpleName()+ java.util.Arrays.toString(c.getOptions()));		
//		System.out.println("Eval Instances: " + eval.numInstances());
//		System.out.println("Correct % : " + eval.pctCorrect());
		return new Point(diabetesDataSetSize-eval.numInstances(), eval.pctCorrect());		
	}
	
	private static void plotLearningCurve(double[] x, double[] y, double[] x2, double[] y2, String title, String xAxis, String yAxis) {
		Plot2DPanel plot = new Plot2DPanel();

		plot.addLinePlot("Training Line", Color.BLUE, x, y);            
		plot.addLinePlot("Validation Line", Color.GREEN, x2, y2);
		plot.setAxisLabels(xAxis,yAxis);

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



