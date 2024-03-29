package com.gatech.yan.java_weka_jmathplot.diabetes;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

import com.gatech.yan.java_weka_jmathplot.Point;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DiabetesReader {

	static int diabetesDataSetSize = 4177;
	
	public List<Point> mcTrainingPoints = new ArrayList<Point>();
	public List<Point> mcTestingPoints = new ArrayList<Point>();
	
	public void doModelComplexity(AbstractClassifier c) throws Exception {

		System.out.println("");
		System.out.println("Classifier: " + c.getClass().getSimpleName() + java.util.Arrays.toString(c.getOptions()));
		mcTrainingPoints.add(getTrainingPointForFullData(c));
		mcTestingPoints.add(getTestingPointFor8020Split(c));
	}
	

	public void doClassifier(AbstractClassifier c, String title) throws Exception {

		System.out.println("");
		System.out.println("Classifier: " + c.getClass().getSimpleName() + java.util.Arrays.toString(c.getOptions()));
		List<Point> pointsTraining = getTrainingPoints(c);
		List<Point> pointsTesting = getTestingPoints(c);
		double[] x = pointsTraining.stream().map(p -> p.getX()).mapToDouble(Double::doubleValue).toArray();
		double[] y = pointsTraining.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
		double[] x2 = pointsTesting.stream().map(p -> p.getX()).mapToDouble(Double::doubleValue).toArray();
		double[] y2 = pointsTesting.stream().map(p -> p.getY()).mapToDouble(Double::doubleValue).toArray();
		plotLearningCurve(x, y, x2, y2, title);
	}
	
	
	private Point getTestingPointFor8020Split(AbstractClassifier c) throws Exception {
		Instances data;
		Instances trainSet;
		Instances testSet;	

		data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);	
		
		int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        trainSet = new Instances(data, 0, trainSize);
        trainSet.setClassIndex(data.numAttributes() - 1);
        testSet = new Instances(data, trainSize, testSize);        
        
		return (getTestingPointForDiabetes(trainSet, testSet,  c));		
	}
		
	

	private List<Point> getTestingPoints(AbstractClassifier c) throws Exception {

		List<Point> pointsTesting = new ArrayList<>();
		Instances data;
		Instances trainDataSet;
		Instances testDataSet;
		int trainSize;
		int testSize;

		data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);

		 trainSize = (int) Math.round(data.numInstances() * 0.8);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));
		
         trainSize = (int) Math.round(data.numInstances() * 0.7);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));

         trainSize = (int) Math.round(data.numInstances() * 0.6);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));

         trainSize = (int) Math.round(data.numInstances() * 0.5);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));
         
         trainSize = (int) Math.round(data.numInstances() * 0.4);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));
         
         trainSize = (int) Math.round(data.numInstances() * 0.3);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));
         
         trainSize = (int) Math.round(data.numInstances() * 0.2);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));
         
         trainSize = (int) Math.round(data.numInstances() * 0.001);
         testSize = data.numInstances() - trainSize;
         trainDataSet = new Instances(data, 0, trainSize);
         trainDataSet.setClassIndex(data.numAttributes() - 1);
         testDataSet = new Instances(data, trainSize, testSize);
         pointsTesting.add(getTestingPointForDiabetes(trainDataSet, testDataSet,  c));

		return pointsTesting;
	}
	
	private Point getTrainingPointForFullData(AbstractClassifier c) throws Exception {

		
		Instances data;		
		data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		return (getTrainingPoint(data, c));		
	}


	private List<Point> getTrainingPoints(AbstractClassifier c) throws Exception {

		List<Point> pointsTraining = new ArrayList<>();
		Instances data;		
		int trainSize;
		Instances trainDataSet;		
		
		data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.2);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.3);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.4);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.5);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.6);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.7);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(data, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 0.8);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
        
    	data = new DataSource("abalone.arff").getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		trainSize = (int) Math.round(data.numInstances() * 1);        
        trainDataSet = new Instances(data, 0, trainSize);
        trainDataSet.setClassIndex(data.numAttributes() - 1);        
        pointsTraining.add(getTrainingPoint(trainDataSet, c));
		
		
		return pointsTraining;
	}

	private Point getTrainingPoint(Instances data, AbstractClassifier c) throws Exception {
		c.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(c, data);
		return (new Point(eval.numInstances(), eval.pctIncorrect()));
	}

	private Point getTestingPointForDiabetes(Instances data, Instances testData, AbstractClassifier c) throws Exception {
		c.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(c, testData);
		return new Point(diabetesDataSetSize - eval.numInstances(), eval.pctIncorrect());
	}

	public void plotLearningCurve(double[] x, double[] y, double[] x2, double[] y2, String title) {
		Plot2DPanel plot = new Plot2DPanel();

		plot.addLinePlot("Training Line", Color.BLUE, x, y);
		plot.addLinePlot("Validation Line", Color.GREEN, x2, y2);
		plot.setAxisLabels("# instances", "error rate %");
		plot.addLegend("SOUTH");

		JFrame frame = new JFrame(title);
		frame.setContentPane(plot);
		frame.setSize(600, 500);
		frame.setVisible(true);

		System.out.println("TrainingPoints:");
		System.out.println(Arrays.toString(x));
		System.out.println(Arrays.toString(y));

		System.out.println("TestingPoints:");
		System.out.println(Arrays.toString(x2));
		System.out.println(Arrays.toString(y2));
	}

}
