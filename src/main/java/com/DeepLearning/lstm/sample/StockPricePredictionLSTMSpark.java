package com.DeepLearning.lstm.sample;


import javafx.util.Pair;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

public class StockPricePredictionLSTMSpark {
	private static final Logger log = LoggerFactory.getLogger(StockPricePredictionLSTMSpark.class);

    private static int exampleLength = 22; // time series length, assume 22 working days per month
    private static int batchSizePerWorker = 16;
    
    

    public static void main (String[] args) throws IOException {
    	

    	
        String filetraining = "D:\\bigdata\\spark\\testdata\\deep\\stock-google-train.csv";
        String testfile = "D:\\bigdata\\spark\\testdata\\deep\\stock-google-test.csv";
        		//new ClassPathResource("prices-split-adjusted.csv").getFile().getAbsolutePath();
        String symbol = "GOOG"; // stock name
        int batchSize = 64; // mini-batch size


        log.info("Create dataSet iterator...");
        PriceCategory category = PriceCategory.CLOSE; // CLOSE: predict close price
        StockDataSetIterator iterator = new StockDataSetIterator(filetraining, symbol, batchSize, exampleLength, category);
        StockDataSetIterator iteratortest = new StockDataSetIterator(testfile, symbol, batchSize, exampleLength, category);
        
        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        

        
        
        
        SparkConf sparkConf = new SparkConf();
    	
    	//sparkConf.setMaster("local[*]");
    	
    	sparkConf.setAppName("DL4J Spark MLP Example");
    	JavaSparkContext sc = new JavaSparkContext(sparkConf);
    	
    	List<DataSet> trainDataList = new ArrayList<>();
    	List<DataSet> testDataList = new ArrayList<>();

    	while (iterator.hasNext()) {
    	trainDataList.add(iterator.next());
    	}
    	while (iteratortest.hasNext()) {
    		testDataList.add(iteratortest.next());
    	}

    	JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
    	JavaRDD<DataSet> testData = sc.parallelize(testDataList);
    	
    	TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
        .batchSizePerWorker(batchSizePerWorker)
        .build();
    	

    	 //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, tm);
        
        int numEpochs = 25; // training epochs 100
      //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch {}", i);
        }
        
        //Perform evaluation (distributed)
        Evaluation evaluation = sparkNet.evaluate(testData);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("Saving model...");
        
     

        //Delete the temp training files, now that we are done with them
       // tm.deleteTempFiles(sc);
        

       
      
        log.info("Done...");
    }

    /** Predict one feature of a stock one-day ahead */
    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");

    }

    private static void predictPriceMultiple (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min) {
        // TODO
    }

    /** Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead */
    private static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\tActual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "\t" + actuals[i]);
        log.info("Plot...");
        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0: name = "Stock OPEN Price"; break;
                case 1: name = "Stock CLOSE Price"; break;
                case 2: name = "Stock LOW Price"; break;
                case 3: name = "Stock HIGH Price"; break;
                case 4: name = "Stock VOLUME Amount"; break;
                default: throw new NoSuchElementException();
            }
         
        }
    }
}
