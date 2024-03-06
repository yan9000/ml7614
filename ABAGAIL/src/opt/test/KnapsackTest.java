package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 1;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT = MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;  //800

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	
    	long startTime, endTime, executionTime;
    	
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        startTime = System.nanoTime();
        System.out.println("===================================================================");
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit_hc = new FixedIterationTrainer(rhc, 1000);
        fit_hc.train();
        System.out.println(ef.value(rhc.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("hill climb execution time: " + executionTime + " nanoseconds");

        startTime = System.nanoTime();
        System.out.println("===================================================================");
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        FixedIterationTrainer fit_sa = new FixedIterationTrainer(sa, 1000);
        fit_sa.train();
        System.out.println(ef.value(sa.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("simmulated annealing execution time: " + executionTime + " nanoseconds");
   
        startTime = System.nanoTime();
        System.out.println("===================================================================");
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
        FixedIterationTrainer fit_ga = new FixedIterationTrainer(ga, 100);
        fit_ga.train();
        System.out.println(ef.value(ga.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("genetic algo execution time: " + executionTime + " nanoseconds");
        
//        startTime = System.nanoTime();
//        System.out.println("===================================================================");
//        ga = new StandardGeneticAlgorithm(200, 150, 50, gap);
//        fit_ga = new FixedIterationTrainer(ga, 100);
//        fit_ga.train();
//        System.out.println(ef.value(ga.getOptimal()));
//        endTime = System.nanoTime();        
//        executionTime = endTime - startTime;
//        System.out.println("genetic algo execution time: " + executionTime + " nanoseconds");
//        
//        startTime = System.nanoTime();
//        System.out.println("===================================================================");
//        ga = new StandardGeneticAlgorithm(200, 150, 75, gap);
//        fit_ga = new FixedIterationTrainer(ga, 100);
//        fit_ga.train();
//        System.out.println(ef.value(ga.getOptimal()));
//        endTime = System.nanoTime();        
//        executionTime = endTime - startTime;
//        System.out.println("genetic algo execution time: " + executionTime + " nanoseconds");
//        
        startTime = System.nanoTime();
        System.out.println("===================================================================");
        MIMIC mimic = new MIMIC(200, 100, pop);
        FixedIterationTrainer fit_mimic = new FixedIterationTrainer(mimic, 1000);
        fit_mimic.train();
        System.out.println(ef.value(mimic.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("mimic execution time: " + executionTime + " nanoseconds");
    }

}
