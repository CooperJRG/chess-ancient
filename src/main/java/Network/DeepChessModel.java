package Network;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.DropoutLayer.Builder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DeepChessModel {

    public ComputationGraph createModel(int filters, int kernelSize, double learningRate) {
        int numResBlocks = 11;
        int numDenseLayers = 3;
        double dropoutRate = 0.47894320806610857;

        String lastLayerName = "input";
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(42)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(8, 8, 14));

        builder.addLayer("0_convolutional", new ConvolutionLayer.Builder(kernelSize, kernelSize)
                .nIn(14)
                .nOut(filters)
                .activation(new ActivationLReLU())
                .build(), lastLayerName);

        lastLayerName = "0_convolutional";

        for (int i = 0; i < numResBlocks; i++) {
            String resBlockName = "residual_block_" + i;
            builder.addLayer(resBlockName + "_conv1", new ConvolutionLayer.Builder(kernelSize, kernelSize)
                    .nIn(filters)
                    .nOut(filters)
                    .activation(new ActivationLReLU())
                    .build(), lastLayerName);

            builder.addLayer(resBlockName + "_conv2", new ConvolutionLayer.Builder(kernelSize, kernelSize)
                    .nIn(filters)
                    .nOut(filters)
                    .activation(new ActivationLReLU())
                    .build(), resBlockName + "_conv1");

            builder.addVertex(resBlockName + "_merge", new MergeVertex(), lastLayerName, resBlockName + "_conv2");

            lastLayerName = resBlockName + "_merge";
        }

        builder.addLayer("global_average_pooling", new GlobalPoolingLayer.Builder(PoolingType.AVG)
                .build(), lastLayerName);

        lastLayerName = "global_average_pooling";

        for (int i = 0; i < numDenseLayers; i++) {
            builder.addLayer(i + "_dense", new DenseLayer.Builder()
                    .nIn((i == 0) ? filters : (2 << (9 - i)))
                    .nOut(2 << (8 - i))
                    .activation(new ActivationLReLU())
                    .build(), lastLayerName);

            builder.addLayer(i + "_dropout", new Builder(dropoutRate)
                    .build(), i + "_dense");

            lastLayerName = i + "_dropout";
        }

        builder.addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(2)
                        .nOut(1)
                        .activation(Activation.valueOf("sigmoid"))
                        .build(), lastLayerName)
                .setOutputs("output");
        ComputationGraphConfiguration configuration = builder.build();
        ComputationGraph model = new ComputationGraph(configuration);
        model.init();
        return model;
    }


}
