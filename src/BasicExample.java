public class BasicExample implements Example {
    private double [] output;
    private double [] input;

    public BasicExample(double [] inputs, int categoryNumber, int categorySize){
        this.input = inputs;
        this.output = new double [categorySize];

        if(categoryNumber < 0 || categoryNumber >= categorySize){
            throw new RuntimeException("Bad Category Number");
        }
        for (int i = 0; i < categorySize; i++){
            if(i != categoryNumber){
                output[i] = 0;
            }else{
                output[i] = 1;
            }
        }

    }

    @Override
    public double getAnswer(int a) {
        return output[a];
    }

    @Override
    public double getInput(int b) {
        return input[b];
    }

}
