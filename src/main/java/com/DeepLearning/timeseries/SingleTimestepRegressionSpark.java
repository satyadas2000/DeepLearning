package com.DeepLearning.timeseries;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.DeepLearning.lstm.sample.StockPricePredictionLSTMSpark;

import javax.swing.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SingleTimestepRegressionSpark {
    private static final Logger LOGGER = LoggerFactory.getLogger(SingleTimestepRegressionExample.class);


    private static File baseDir = new File("D:\\bigdata\\spark\\testdata\\pythondata\\");

    public static void main(String[] args) throws Exception {

        int miniBatchSize = 32;

        // ----- Load the training data -----
        SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ";");
        trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/train_%d.csv", 0, 0));

        //For regression, numPossibleLabels is not used. Setting it to -1 here
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true);

        SequenceRecordReader testReader = new CSVSequenceRecordReader(0, ";");
        testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/test_%d.csv", 0, 0));
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true);

   

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(0.0015)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));
        
        SparkConf sparkConf = new SparkConf();     	
     	sparkConf.setAppName("DL4J Spark  Example");
     	JavaSparkContext sc = new JavaSparkContext(sparkConf);     	

     	List<DataSet> trainDataList = new ArrayList<>();

    	while (trainIter.hasNext()) {
    	trainDataList.add(trainIter.next());
    	}



    	JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
    	
    	JavaRDD<DataSet>[] splits = trainData.randomSplit(new double[] { 0.80, 0.10, 0.10 } ,1);
    	JavaRDD<DataSet> jTrainData = splits[0];
    	JavaRDD<DataSet> jValidData = splits[1];
    	JavaRDD<DataSet> jTestData = splits[2];


     	
     	TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(miniBatchSize)    //Each DataSet object: contains (by default) 32 examples
         .averagingFrequency(5)
         .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
         .batchSizePerWorker(miniBatchSize)
         .build();
     	

     	 //Create the Spark network
         SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, tm); 
         int nEpochs = 10;
         for (int i = 0; i < nEpochs; i++) {
        	 sparkNet.fit(jTrainData);       
             
         }

     //TODO TEST Evaluation
      

        // ----- Train the network, evaluating the test set performance at each epoch -----
 

        LOGGER.info("----- Example Complete -----");
    }

    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Number of passengers";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }
}
