import java.util.ArrayList;
import java.util.List;
// My neural net can run XOR, AND, and a digit data set downloaded from: http://yann.lecun.com/exdb/mnist/
public class Main {

    public static void main (String[] args) {
        //for(int y = 2; y <= 8; y++){
        //    for(double x = 1; x <= 10; x ++){
        //        testXOR(x, y);
                //testAND(x, y);

        //    }
        //}
        //testMnist();
        testXOR(2,3);
    }

    static void testXOR(double learningRate, int hiddenNode){
        System.out.println("Learning XOR");
        BasicExample[] examplesList = new BasicExample[4];
        examplesList[0] = new BasicExample(new double []{0,0}, 0, 2);
        examplesList[1] = new BasicExample(new double []{0,1}, 1, 2);
        examplesList[2] = new BasicExample(new double []{1,0}, 1, 2);
        examplesList[3] = new BasicExample(new double []{1,1}, 0, 2);
        NeuralNet basic = new NeuralNet(2,hiddenNode,2, learningRate);
        //basic.learn(examplesList, examplesList, examplesList, 1.0, 1000,
                //10000, 10000);
        int epoch = basic.learn_many_example(examplesList, 1.0, 1000000, 10000, 10000);
        System.out.println("HiddenNode: " + hiddenNode + " learningRate: " + learningRate + " Epoch: " + epoch);
    }

    static void testAND(double learningRate, int hiddenNode){
        System.out.println("Learning AND");
        BasicExample[] examplesList = new BasicExample[4];
        examplesList[0] = new BasicExample(new double []{0,0}, 0, 2);
        examplesList[1] = new BasicExample(new double []{0,1}, 0, 2);
        examplesList[2] = new BasicExample(new double []{1,0}, 0, 2);
        examplesList[3] = new BasicExample(new double []{1,1}, 1, 2);
        NeuralNet basic = new NeuralNet(2,hiddenNode,2,learningRate);
        int epoch = basic.learn_many_example(examplesList, 1.0, 1000000, 10000, 10000);
        System.out.println("HiddenNode: " + hiddenNode + " learningRate: " + learningRate + " Epoch: " + epoch);

        //basic.learn(examplesList, examplesList, examplesList, 1.0, 100,
                //10000, 10000);
    }


    static void testMnist(){
        System.out.println("Learning Mnist");
        BasicExample[] imageTrainingList = convertMnistExamples("train-labels.idx1-ubyte",
                "train-images.idx3-ubyte");
        BasicExample[] imageTestingList = convertMnistExamples("t10k-labels.idx1-ubyte",
                "t10k-images.idx3-ubyte");
        NeuralNet basic = new NeuralNet(numPixel(),20,10,0.9);
        basic.learn(imageTrainingList, imageTrainingList, imageTestingList, 0.9, 1,
                10000000, 100000);
        //basic.learn_one_example(imageTrainingList[0]); //Debugging
    }

    static BasicExample[] convertMnistExamples(String labelsPath, String imagePath) {
        int[] labels = MnistReader.getLabels("mnistData/" + labelsPath);
        //System.out.println("Mnist contains x examples: "+labels.length);
        List<int[][]> images = MnistReader.getImages("mnistData/" + imagePath);
        BasicExample[] result = new BasicExample[labels.length];
        for (int ii=0; ii<labels.length; ii++) {
            double[] linearImage = new double[images.get(0).length*images.get(0)[0].length];
            for (int jj=0; jj<images.get(0).length; jj++) {
                for (int kk=0; kk<images.get(0)[0].length; kk++) {
                    linearImage[images.get(0).length*jj+kk] = (double) images.get(ii)[jj][kk]/(double)255;
                }
                //Debugging
                //for (int a = 0; a < images.get(0).length*images.get(0)[0].length; a++){
                //    System.out.println(linearImage[a]);
                //}
                //Debugging
            }
            result[ii] = new BasicExample(linearImage, labels[ii], 10);
        }
        return result;
    }

    static int numPixel(){
        List<int[][]> images = MnistReader.getImages("mnistData/t10k-images.idx3-ubyte");
        return images.get(0).length*images.get(0)[0].length;
    }

    static void debuggingExample(){
        System.out.println("Annoying Debugging Stuff");
        NeuralNet basic = new NeuralNet(2,2,2,0.1);
        BasicExample[] examplesList = new BasicExample[1];
        examplesList[0] = new BasicExample(new double []{0,0}, 0, 2);
        basic.learn_one_example(examplesList[0]);
        basic.calculate_accuracy(examplesList);
        basic.initializeRandomWeights();
        basic.classify_one_example(examplesList[0]);
    }

}
