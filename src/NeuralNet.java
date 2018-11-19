public class NeuralNet {
    private int numInput;
    private int numHiddenNodes;
    private int numOutputNodes;
    private double learningRate;
    // first [] is index of hidden node, second [] is index of which child
    // (overall it is an array of weights from children)
    private double [][] inputHiddenWeights;
    // first [] is index of output node, second [] is index of which child
    // (overall it is the array of weights from children)
    private double [][] hiddenOutputWeights;
    // first [] is index of the numHiddenNodes
    private double [] recentHiddenOutput;
    // first [] is index of numOutputNodes
    private double [] recentOutputOutput;

    public NeuralNet(int numInput, int numHiddenNodes, int numOutputNodes, double learningRate) {
        this.numInput = numInput;
        this.numHiddenNodes = numHiddenNodes;
        this.numOutputNodes = numOutputNodes;
        this.learningRate = learningRate;
        this.inputHiddenWeights = new double[numHiddenNodes][numInput + 1];
        this.hiddenOutputWeights = new double[numOutputNodes][numHiddenNodes + 1];
        this.recentHiddenOutput = new double[numHiddenNodes];
        this.recentOutputOutput = new double[numOutputNodes];
        initializeRandomWeights();
    }

    public void initializeRandomWeights(){
        for (int i = 0; i < numHiddenNodes; i++){
            for (int j = 0; j <= numInput; j++){
                double random1 = (Math.random() - 0.5)/10;
                inputHiddenWeights[i][j] = random1;
            }
        }
        for (int a = 0; a < numOutputNodes; a++){
            for (int b = 0; b <= numHiddenNodes; b++){
                double random2 = (Math.random() - 0.5)/10;
                hiddenOutputWeights[a][b] = random2;
            }
        }
    }

    public int classify_one_example(Example example){
        double [] listExampleNodeValues = new double[numInput];
        for(int i = 0; i < numInput; i++){
                listExampleNodeValues[i] = example.getInput(i);
        }
        recentHiddenOutput = multiplyWeight(numHiddenNodes, numInput, inputHiddenWeights, listExampleNodeValues);
        recentOutputOutput = multiplyWeight(numOutputNodes, numHiddenNodes, hiddenOutputWeights, recentHiddenOutput);

        // Get the max
        double maximumValue = 0;
        int largestOutput = 0;
        for(int j = 0; j < recentOutputOutput.length; j++){
            if(recentOutputOutput[j] > maximumValue){
                maximumValue = recentOutputOutput[j];
                // LargestOutput is the index of the largest output
                largestOutput = j;
            }
        }
        // takes one example and returns the category that the neural net thinks it belongs to
        return largestOutput;
    }

    // Multiply the inputs by the weight to get output. Stores in list for later use.
    public double [] multiplyWeight(int numUpperLevelNodes, int numLowerLevelNodes, double [][] weights,
                                    double [] listLowerNodeValues){
        double [] listNodeValue = new double[numUpperLevelNodes];
        for(int i = 0; i < numUpperLevelNodes; i++){
            double nodeValue = 0;
            for (int j = 0; j < numLowerLevelNodes; j++){
                double nodeAnswer = weights[i][j] * listLowerNodeValues[j];
                nodeValue += nodeAnswer;
            }
            nodeValue += weights[i][numLowerLevelNodes]; //Bias
            // Activation function
            listNodeValue[i] = 1/ (1 + Math.pow(Math.E, -nodeValue));
        }

        return listNodeValue;
    }

    public double calculate_accuracy(Example [] basicExampleList) {
        int trueValues = 0;
        for(Example basic : basicExampleList){
            if(basic.getAnswer(classify_one_example(basic)) == 1){
                trueValues++;
            }
        }
        // takes a list of pre-categorized testing examples, and returns the fraction (or percent) that are correctly
        // classified by this network
        return (double)trueValues / (double)basicExampleList.length;
    }

    public void learn_one_example(Example basic){
        classify_one_example(basic);

        //Compute Output Error Signal
        double [] outputErrorSignalList = new double[numOutputNodes];
        for (int i = 0; i < numOutputNodes; i++){
            double outputErrorSignal = (basic.getAnswer(i) - recentOutputOutput[i]) * recentOutputOutput[i] *
                    (1-recentOutputOutput[i]);
            outputErrorSignalList[i] = outputErrorSignal;
            //System.out.println("Output Error Signal: " + outputErrorSignal); // Debugging
        }

        //Compute Hidden Error Signal
        double [] hiddenErrorSignalList = new double [numHiddenNodes];
        for (int a = 0; a < numHiddenNodes; a++){
            double runningTotal = 0;
            for (int b = 0; b < numOutputNodes; b++){
                double total = outputErrorSignalList[b] * hiddenOutputWeights[b][a];
                runningTotal += total;
            }
            double hiddenErrorSignal = runningTotal * (recentHiddenOutput[a] * (1 - recentHiddenOutput[a]));
            //System.out.println("Hidden Error Signal: " + hiddenErrorSignal); // Debugging
            hiddenErrorSignalList[a] = hiddenErrorSignal;
        }

        //Update Output Weights
        for (int c = 0; c < numOutputNodes; c++){
            for(int d = 0; d < numHiddenNodes; d++){
                double newOutputWeight = hiddenOutputWeights[c][d] + outputErrorSignalList[c] * recentHiddenOutput[d]
                        * learningRate;
                // Changing Weights? Debugging
                //System.out.println("Original Hidden to Output Weights: " + hiddenOutputWeights[c][d]); //Debugging
                hiddenOutputWeights[c][d] = newOutputWeight;
                //System.out.println("New Hidden to Output Weights: " + hiddenOutputWeights[c][d]); //Debugging
            }
            double newOutputWeight = hiddenOutputWeights[c][numHiddenNodes] + outputErrorSignalList[c]*learningRate;
            hiddenOutputWeights[c][numHiddenNodes] = newOutputWeight; //Bias
        }

        //Update Hidden Weights
        for(int e = 0; e < numHiddenNodes; e++){
            for(int f = 0; f < numInput; f++){
                double newHiddenWeights = inputHiddenWeights[e][f] + hiddenErrorSignalList[e] * basic.getInput(f)
                        * learningRate;
                //System.out.println("Original Input to Hidden Weights: " + inputHiddenWeights[e][f]); //Debugging
                inputHiddenWeights[e][f] = newHiddenWeights;
                //System.out.println("New Input to Hidden Weights: " + inputHiddenWeights[e][f]); //Debugging

            }
            double newHiddenWeights = inputHiddenWeights[e][numInput] + hiddenErrorSignalList[e]*learningRate;
            inputHiddenWeights[e][numInput] = newHiddenWeights; //Bias
        }
        //takes one pre-categorized example, classifies it, and then learns by modifying the network's weights
    }

    public int learn_many_example(Example [] basicExampleList, double desiredAccuracy,
                                  int maxEpoch, double timeLimit, int epochInterval){
        double startTime = System.currentTimeMillis();
        while(System.currentTimeMillis() - startTime < timeLimit){
            initializeRandomWeights();
            //System.out.println("Number of Examples is: " + basicExampleList.length);
            for(int epoch = 1; epoch < maxEpoch; epoch++){
                for (Example basic : basicExampleList) {
                    learn_one_example(basic);
                }
                double currentAccuracy = calculate_accuracy(basicExampleList);


                if(epoch%epochInterval == 0){
                    //System.out.println("Epoch: " + epoch + ", Accuracy: " + currentAccuracy +
                            //", Time: " + (-(startTime - System.currentTimeMillis())));
                }
                if (currentAccuracy >= desiredAccuracy) {
                    //System.out.println("Epoch To Learn: " + epoch);
                    return epoch;
                }
            }
            //System.out.println("Max Epoch Hit: Restarting");
        }
        //System.out.println("Failed within given time");
        return 0;

        //takes a list of pre-categorized training examples (and maybe a list of pre-categorized validation examples),
        // and a desired accuracy and/or time limit.  It repeatedly learns from the training examples until a
        // termination condition is reached.
    }

    public void learn(Example [] trainingList, Example [] validationList, Example [] testingList,
                      double desiredAccuracy, int epochInterval, int maxEpoch, double timeLimit){
        learn_many_example(trainingList, desiredAccuracy, maxEpoch, timeLimit, epochInterval);
        double validationAc = calculate_accuracy(validationList);
        double testingAc = calculate_accuracy(testingList);
        System.out.println("Validation Accuracy: " + validationAc + ", Test Accuracy: " + testingAc);
    }



}
