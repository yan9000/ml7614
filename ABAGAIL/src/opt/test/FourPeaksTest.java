package opt.test;

import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 8;
    /** The t value */
    private static final int T = 2;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        FourPeaksEvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        long startTime, endTime, executionTime;
        FixedIterationTrainer fit;
        
        startTime = System.nanoTime();
        System.out.println("===================================================================");
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        fit = new FixedIterationTrainer(rhc, 1000);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("hill climb execution time: " + executionTime + " nanoseconds");

        startTime = System.nanoTime();
        System.out.println("===================================================================111111111111");
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 256);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("simmulated annealing execution time: " + executionTime + " nanoseconds");
        
//        startTime = System.nanoTime();
//        System.out.println("===================================================================2222222222222");
//        sa = new SimulatedAnnealing(1E11, .55, hcp);
//        fit = new FixedIterationTrainer(sa, 256);
//        fit.train();
//        System.out.println("SA: " + ef.value(sa.getOptimal()));
//        endTime = System.nanoTime();        
//        executionTime = endTime - startTime;
//        System.out.println("simmulated annealing execution time: " + executionTime + " nanoseconds");
//        
//        
//        startTime = System.nanoTime();
//        System.out.println("===================================================================3333333333333");
//        sa = new SimulatedAnnealing(1E11, .1, hcp);
//        fit = new FixedIterationTrainer(sa, 256);
//        fit.train();
//        System.out.println("SA: " + ef.value(sa.getOptimal()));
//        endTime = System.nanoTime();        
//        executionTime = endTime - startTime;
//        System.out.println("simmulated annealing execution time: " + executionTime + " nanoseconds");
        

        startTime = System.nanoTime();
        System.out.println("===================================================================");
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("genetic execution time: " + executionTime + " nanoseconds");
                
        startTime = System.nanoTime();
        System.out.println("===================================================================");
        MIMIC mimic = new MIMIC(200, 20, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
        endTime = System.nanoTime();        
        executionTime = endTime - startTime;
        System.out.println("mimic execution time: " + executionTime + " nanoseconds");
    }
}
